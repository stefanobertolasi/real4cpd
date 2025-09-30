from typing import List
from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from .model import GRUModel, TCNModel
from .segmenters import Segmenter


class ActiveLearner:

    @abstractmethod
    def get_candidate():
        pass

    @abstractmethod
    def query():
        pass

    @abstractmethod
    def run():
        pass

class RandomLearner(ActiveLearner):

    def __init__(self, segmenter: "Segmenter",  gt_break_points: List[int], tolerance: int):
        
        self.segmenter: Segmenter=segmenter
        self.gt_break_points = gt_break_points
        self.nsamples = segmenter.supervised_domain.nsamples
        self.tolerance = tolerance

    def get_candidate(self):

        if np.any(np.isnan(self.segmenter.get_score())):
            raise ValueError("Found NaN in scores!")

        unsupervised_indices = np.nonzero(self.segmenter.supervised_domain.get_unsupervised_indices())[0]
        
        try:
            return np.random.choice(range(self.nsamples))
        except:
            return np.random.choice(unsupervised_indices)
        
    
    def query(self, ccp: int):

        start = max(0, ccp-self.tolerance)
        end = min(self.nsamples, ccp+self.tolerance)

        supervised_gt = (self.gt_break_points > start) & (self.gt_break_points < end)
        if supervised_gt.any():
            return True, self.gt_break_points[np.where(supervised_gt)[0][0]]
        else:
            return False, 0
        
class ExpertLearner(ActiveLearner):

    def __init__(self, segmenter: "Segmenter",  gt_break_points: List[int], tolerance: int):
        
        self.segmenter: Segmenter=segmenter
        self.gt_break_points = gt_break_points
        self.nsamples = segmenter.supervised_domain.nsamples
        self.tolerance = tolerance

    def get_candidate(self):

        score = self.segmenter.get_unsupervised_score()

        if np.any(np.isnan(self.segmenter.get_score())):
            raise ValueError("Found NaN in scores!")
        
        threshold = self.segmenter.threshold

        pos_indices = np.nonzero(score>0)[0]
        if len(pos_indices) == 0:
            if self.segmenter.supervised_domain.__len__ == self.nsamples:
                raise RuntimeError("Trying to query on fully supervised data")
            unsupervised_indices = np.nonzero(self.segmenter.supervised_domain.get_unsupervised_indices())[0]
            return np.random.choice(range(self.nsamples))

        pos_score = score[pos_indices]

        ccp_idx = np.argmin(np.abs(pos_score - threshold))
        ccp = pos_indices[ccp_idx]

        return ccp
    
    def query(self, ccp: int):

        start = max(0, ccp-self.tolerance)
        end = min(self.nsamples, ccp+self.tolerance)

        supervised_gt = (self.gt_break_points >= start) & (self.gt_break_points <= end)
        if supervised_gt.any():
            return True, self.gt_break_points[np.where(supervised_gt)[0][0]]
        else:
            return False, 0
        
class DQNLearner(ActiveLearner):

    def __init__(self, 
                num_samples: int,
                gt_break_points: List[int],
                args: dict = None,
                options: dict = None
                ):
        
        self.gamma = args.gamma
        self.tau = args.tau
        self.epsilon = args.initial_exploration
        self.epsilon_decay = args.exploration_decay
        self.epsilon_min = args.minimum_exploration
        self.gt_break_points = gt_break_points
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_samples = num_samples

        self.options = options or {}
        self.mode = self.options.get('mode', 'train')

        self.model = TCNModel(args=args)
        self.target_model = TCNModel(args=args)

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        

    def get_candidate(self, state: torch.tensor):

        if self.mode == "eval":
            self.model.eval()
            with torch.no_grad():
                state = state.to(self.device)
                q_values, _ = self.model(state)  # q_values of shape [N, P]

            action = torch.argmax(q_values)
            return action

        state = torch.from_numpy(state) #This because AsyncVectEnv returns a numpy array

        self.model.eval()
        with torch.no_grad():
            q_values, _ = self.model(state.to(self.device, dtype=torch.float32))  # q_values of shape [N, P]

        self.model.train()

        q_values = q_values.cpu().detach().numpy()
        actions = []
        for i in range(q_values.shape[0]):
            if np.random.rand() < self.epsilon:
                action = np.random.choice(np.arange(q_values.shape[1]))
            else:
                action = np.argmax(q_values[i])
            
            actions.append(action)
        
        return actions

    def query():
        pass

    def learn(self, batch: dict):

        state_batch = batch["state"].to(self.device, dtype = torch.float32)
        action_batch = batch["action"].to(self.device, dtype = torch.int64)
        reward_batch = batch["reward"].to(self.device, dtype = torch.float32)
        next_state_batch = batch["next_state"].to(self.device, dtype = torch.float32)
        done_batch = batch["done"].to(self.device, dtype = torch.float32)

        with torch.no_grad():
            next_q_values, _ = self.target_model(next_state_batch)
            max_next_q_values = torch.max(next_q_values, dim=1).values
            target_q_values = reward_batch + self.gamma*max_next_q_values*(1-done_batch)

        current_q_values, _ = self.model(state_batch)
        action_q_values = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze()

        td_errors = target_q_values - action_q_values
        loss = nn.MSELoss()(action_q_values, target_q_values)

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        self.update_network_param()

        return loss.item(), td_errors
     
    def update_network_param(self, tau:float = None):
        if tau is None:
            tau=self.tau

        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.lerp_(param.data, tau)

