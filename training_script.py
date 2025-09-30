import mlflow.config
import numpy as np
import os, mlflow 
import torch
from matplotlib import pyplot as plt
from segmentation.environment import SegmentationEnv
from segmentation.learners import DQNLearner
from segmentation.utils import get_parameters, PrioritizedReplayBuffer, ReplayBuffer
import argparse
from typing import List
from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_env(samples, gt_break_points, level_wavelet, ws, tolerance, args):
    def _init():
        return SegmentationEnv(
            samples=samples, 
            gt_break_points=gt_break_points,
            level_wavelet=level_wavelet,
            window_size=ws,
            tolerance=tolerance,
            args=args) 
    return _init

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    mlflow.set_experiment("ActiveRL-Pool_Version")
    mlflow.start_run()
    mlflow.config.enable_async_logging(True)

    N_ENVS = args.num_cpus
    
    params_dict = OmegaConf.to_container(args, resolve=True)
    mlflow.log_params(params_dict)

    data = np.load(args.dataset)
    samples = data['samples']
    gt_break_points = data['gt_break_points']

    mu = samples.mean()
    sigma = samples.std()
    samples = (samples - mu) / (3 * sigma)

    ws, tolerance, level_wavelet, budget = get_parameters(dataset=args.dataset_name)

    env = AsyncVectorEnv([make_env(samples, gt_break_points, level_wavelet, ws, tolerance, args) for _ in range(N_ENVS)], autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    state, info = env.reset()
    
    learner = DQNLearner(num_samples=samples.shape[0], gt_break_points=gt_break_points, args=args)
    # replay_buffer = ReplayBuffer(capacity=args.buffer_capacity, device=DEVICE)
    replay_buffer = PrioritizedReplayBuffer(capacity=args.buffer_capacity, alpha=0.3, beta=0.4, device=DEVICE)

    os.makedirs("outputs", exist_ok=True)

    global_ep_done = 0
    step = 0

    total_rewards = np.zeros(N_ENVS)
    total_reward_history = []
    total_losses = 0
    num_updates = 0
    checkpoint_interval = 500

    while global_ep_done < args.num_episodes:
        action = learner.get_candidate(state)
        next_state, reward, done, terminated, info = env.step(action)
        total_rewards += reward

        replay_buffer.store(state, action, reward, next_state, done)

        if replay_buffer.is_full:
            batch = replay_buffer.sample(args.batch_size)
            loss, td_errors = learner.learn(batch)
            replay_buffer.update_priorities(batch['idxs'], td_errors)
            total_losses += loss
            num_updates += 1
            avg_loss = total_losses / num_updates if num_updates > 0 else 0.0
            learner.model.scheduler.step(avg_loss)
            mlflow.log_metric("episode_avg_loss", avg_loss, step=step, synchronous=False)

        autoreset = done + terminated
        if np.any(autoreset):
            state, info = env.reset(options={'reset_mask': autoreset})
            learner.epsilon *= learner.epsilon_decay
            learner.epsilon = max(learner.epsilon, learner.epsilon_min)

            
            for i, done in enumerate(autoreset):
                if done:
                    mlflow.log_metric("episode_reward", total_rewards[i], step=global_ep_done, synchronous=False)
                    total_reward_history.append(total_rewards[i])

                    avg_total_reward = np.mean(total_reward_history[-100:]) if len(total_reward_history) >= 100 else 0.0
                    if global_ep_done > 100:
                        mlflow.log_metric("avg_total_reward", avg_total_reward, step=global_ep_done, synchronous=False)

                    print(f"Episode {global_ep_done}, Reward {total_rewards[i]}, Avg Loss {0 if not replay_buffer.is_full else avg_loss:.6f}")
                    global_ep_done += 1
                    total_rewards[i] = 0

                    if global_ep_done % checkpoint_interval == 0 and global_ep_done > 0:
                        checkpoint_path = f"outputs/dqn_ep{global_ep_done}.pt"
                        torch.save(learner.model.state_dict(), checkpoint_path)
                        mlflow.log_artifact(checkpoint_path)

            avg_total_reward = np.mean(total_reward_history[-100:])
            if global_ep_done>100:
                mlflow.log_metric("avg_total_reward", avg_total_reward, step=global_ep_done, synchronous=False)
        else:
            state = next_state

        
        step += 1

    # Save final model
    final_model_path = f"outputs/{args.output}.pt"
    torch.save(learner.model.state_dict(), final_model_path)
    mlflow.log_artifact(final_model_path)

    mlflow.end_run()

if __name__ == "__main__":
    main()