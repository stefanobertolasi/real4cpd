import gymnasium as gym
import numpy as np
import math, torch, random
from typing import List
from .segmenters import Segmenter, WaveletDecompositionModel, CMAESOptimizer, HingeMarginLoss, MangoOptimizer

class SegmentationEnv(gym.Env):

    def __init__(
            self, 
            samples: np.ndarray,
            gt_break_points: List[int],
            level_wavelet: int,
            window_size: int,
            tolerance: int,
            args: dict = None,
            options: dict = None
            ):

        score_model = WaveletDecompositionModel(samples, level_wavelet=level_wavelet, window_size=window_size)
        segmenter = Segmenter(score_model=score_model)
        optimizer = CMAESOptimizer(max_calls=args.opt_max_calls)
        loss = HingeMarginLoss(threshold=segmenter.threshold, peak_tolerance=tolerance)
        segmenter.compile(loss, optimizer)

        self.score_model = score_model
        self.segmenter = segmenter
        self.optimizer = optimizer
        self.loss = loss
        self.tolerance = tolerance

        self.options = options or {}
        self.mode = self.options.get('mode', 'train')

        self.num_samples = segmenter.num_samples
        self.samples = samples
        self.gt_break_points = gt_break_points
        self.gt_normal_points = [i for i in range(self.num_samples) if i not in self.gt_break_points]
        self.level_wavelet = level_wavelet
        self.window_size = window_size
        self.budget = args.budget 
        self.pool_size = args.pool_size

        self.unlabeled_set = list(range(0, self.num_samples, self.tolerance//2))

        if self.mode == 'train':
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10, self.pool_size, 2*self.tolerance),
                )

            self.action_space = gym.spaces.Discrete(
                self.pool_size
                )
        elif self.mode == "eval":
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10, len(self.unlabeled_set), 2*self.tolerance),
                )

            self.action_space = gym.spaces.Discrete(
                len(self.unlabeled_set)
                )
    
    def _get_state_pool(self):

        if self.mode == "train":
            self.pool = random.sample(self.unlabeled_set, self.pool_size)
        elif self.mode == "eval":
            self.pool = self.unlabeled_set

        self.score = self.segmenter.get_score()
        features = np.stack([feature.get_profile() for feature in self.segmenter.score_model.all_resample_profiles], axis=0)
        _score = (self.score-self.segmenter.threshold).reshape(1, self.num_samples)

        supervised_domain = np.full(self.segmenter.supervised_domain.get_supervised_indices().shape, -1)
        supervised_domain[self.segmenter.supervised_domain.get_supervised_indices()] = 0
        for gt in self.segmenter.gt_break_points:
            supervised_domain[gt] = 1

        self.break_points = self.segmenter.get_break_points()
        ccp = np.full(supervised_domain.shape, 0)
        for bkp in self.break_points:
            ccp[bkp] = 1

        supervision = np.stack([supervised_domain, ccp], axis=0)

        state = np.concatenate([features, _score, supervision], axis=0, dtype=np.float32)
        state = torch.from_numpy(state)

        windows = []
        for idx in self.pool:
            start = max(0, idx - self.tolerance)
            end = min(state.shape[1], idx + self.tolerance)
            window = state[:, start:end]
            if window.shape[1] < 2*self.tolerance:
                pad_width = 2*self.tolerance - window.shape[1]
                window = torch.nn.functional.pad(window, (0, pad_width))
            windows.append(window)

        windows = torch.stack(windows, dim=1)
        return windows

    def _get_info(self):

        precision, recall, f1 = self.segmenter.get_f1_score(self.break_points, self.gt_break_points)
        reward = self.segmenter.get_margin_score(self.score, self.gt_break_points, self.gt_normal_points)
        return {
            "reward": reward,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "steps": self.num_steps
        }

    def reset(self, *, seed = None, options = None, args=None):
        super().reset(seed=seed, options=options)

        self.score_model.reset()
        self.segmenter.reset()
        self.break_points = self.segmenter.get_break_points()
        self.num_steps = 0
        
        self.unlabeled_set = list(range(0, self.num_samples, self.tolerance//2))

        state_pool = self._get_state_pool()
        info = self._get_info()

        return state_pool, info 

    def step(self, action = None):

        info = self._get_info()
        idx = self.pool[action]

        try:
            start = max(0, idx-self.tolerance)
            end = min(self.num_samples, idx+self.tolerance)
        except TypeError:
            raise(TypeError, f"Action type {idx.type()}")

        supervised_gt = (self.gt_break_points >= start) & (self.gt_break_points <= end)
        if supervised_gt.any():
            gt = self.gt_break_points[np.where(supervised_gt)[0][0]]
            self.segmenter.add_break_point_to_gt(gt, [start, end])
        else:
            self.segmenter.add_supervised_interval([start, end])

        self.segmenter.update()
        self.num_steps += 1
        self.unlabeled_set.remove(idx)

        next_state = self._get_state_pool()
        next_info = self._get_info()

        done = next_info["f1"]==1
        terminated = self.num_steps==self.budget

        reward = (
            (1 + (1-self.num_steps/self.budget)) * (next_info["f1"]-info["f1"])
            # (1 + (1-self.num_steps/self.budget)) * (next_info["reward"]-info["reward"])
        )

        return next_state, reward, done, terminated, next_info
    
    def render(self):
        return super().render()