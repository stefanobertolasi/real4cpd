
"""
This module provides utility classes for handling circular arrays and computing moving moments and covariance ratios.

Classes:
    CircularArray: A class for managing circular arrays for multivariate data.
    CircularArrayUnivariate: A subclass of CircularArray tailored for univariate data.
    MovingMoments: A class for computing and updating the mean and covariance of a dataset.
    MovingCovarianceRatio: A class for computing the divergence of a dataset based on moving covariance.

CircularArray:
    Manages a circular array structure for multivariate data, supporting operations such as adding new samples,
    and retrieving the first, last, and middle samples.

CircularArrayUnivariate:
    Extends CircularArray for univariate data, providing methods to add new samples and check if the middle sample
    is a maximum with respect to the entire array.

MovingMoments:
    Computes and updates the mean and covariance of a dataset. Supports updating the dataset with new samples
    and removing old ones.

MovingCovarianceRatio:
    Utilizes MovingMoments to compute the divergence of a dataset based on the covariance of samples before and
    after a midpoint in a sliding window approach.

"""

from typing import Dict, Tuple, List, Protocol
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random



class CircularArray:
    """
    A class representing a circular array.

    Attributes:
        samples (np.array): The array of samples.
        start (int): The starting index of the circular array.

    Methods:
        dimension() -> int: Returns the dimension of the samples.
        n_samples() -> int: Returns the number of samples.
        middle_index() -> int: Returns the index of the middle sample.
        get_first() -> np.array: Returns the first sample.
        get_last() -> np.array: Returns the last sample.
        get_middle() -> np.array: Returns the middle sample.
        add(sample: np.array): Adds a new sample to the circular array.
        get_samples() -> np.array: Returns all the samples in the circular array.

    The CircularArray class is designed to manage a circular array of samples. It supports operations such as adding new samples, and retrieving the first, last, and middle samples. The class automatically handles the circular nature of the array, ensuring that indices wrap around as necessary.
    """

    def __new__(cls, samples: np.array):
        """
        Create a new instance of CircularArray.

        Args:
            samples (np.array): The array of samples.

        Returns:
            object: Returns an instance of CircularArray if the input samples are 
                multidimensional, otherwise returns an instance of CircularArrayUnivariate.
        """

        samples = np.squeeze(samples)
        if len(samples.shape) > 1:
            return object.__new__(cls)
        else:
            return object.__new__(CircularArrayUnivariate)


    def __init__(self, samples: np.array):
        """
        Initializes a CircularArray object.

        Args:
            samples (np.array): The array of samples.
        """
        self.samples = samples
        self.start = 0

    @property
    def dimension(self) -> int:
        """
        Returns the dimension of the samples.

        Returns:
            int: The dimension of the samples.
        """
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        """
        Returns the number of samples.

        Returns:
            int: The number of samples.
        """
        return self.samples.shape[1]
    
    @property
    def middle_index(self) -> int:
        """
        Returns the index of the middle sample.

        Returns:
            int: The index of the middle sample.
        """
        return (self.start + self.n_samples//2) % self.n_samples  

    def get_first(self) -> np.array:
        """
        Returns the first sample.

        Returns:
            np.array: The first sample.
        """
        return self.samples[:, self.start, np.newaxis]
    
    def get_last(self) -> np.array:
        """
        Returns the last sample.

        Returns:
            np.array: The last sample.
        """
        end = (self.start - 1) % self.n_samples
        return self.samples[:, end, np.newaxis]

    def get_middle(self) -> np.array:
        """
        Returns the middle sample.

        Returns:
            np.array: The middle sample.
        """
        return self.samples[:, self.middle_index, np.newaxis]

    def add(self, sample: np.array):
        """
        Adds a new sample to the circular array.

        Args:
            sample (np.array): The new sample to be added.
        """
        self.samples[:, self.start] = sample.squeeze()
        self.start = (self.start + 1) % self.n_samples
    
    def get_samples(self) -> np.array:
        """
        Returns all the samples in the circular array.

        Returns:
            np.array: The array of samples.
        """
        samples = np.hstack((self.samples[:, self.start:], self.samples[:, :self.start]))
        return samples
 

class CircularArrayUnivariate(CircularArray):
    """
    A class representing a univariate circular array, derived from CircularArray.

    This class is intended for use when the input samples are unidimensional. It inherits from CircularArray and overrides methods as necessary to accommodate the unidimensional nature of its samples.

    Attributes and methods are inherited from CircularArray, with modifications as needed for unidimensional data handling.

    Methods:
        is_middle_maximum() -> bool: Check if the middle element of the circular array is the maximum.
    """

    def __init__(self, samples: np.array):
        """
        Initialize the CircularArrayUnivariate object.

        Args:
            samples (np.array): The univariate data samples.

        """
        samples = np.reshape(samples, (1, len(samples.squeeze())))
        super().__init__(samples)
    
    def get_first(self) -> float:
        """
        Get the first element of the circular array.

        Returns:
            float: The first element of the circular array.

        """
        return super().get_first().squeeze()
    
    def get_last(self) -> float:
        """
        Get the last element of the circular array.

        Returns:
            float: The last element of the circular array.

        """
        return super().get_last().squeeze()
    
    def add(self, sample: float):
        """
        Add a new sample to the circular array.

        Args:
            sample (float): The new sample to be added.

        """
        super().add(np.array(sample).reshape((1, 1)))
    
    def is_middle_maximum(self) -> bool:
        """
        Check if the middle element of the circular array is the maximum.

        Returns:
            bool: True if the middle element is the maximum and unique, False otherwise.

        """
        middle = self.get_middle()
        is_maximum = np.all(middle >= self.samples)
        is_unique = np.sum(middle == self.samples) == 1
        return is_maximum and is_unique


class MovingMoments:
    """
    Class for computing moving moments of a set of samples.

    Attributes:
        _mean (np.array): The mean of the samples.
        _covariance (np.array): The covariance matrix of the samples.
        n_samples (int): The number of samples.

    Methods:
        __init__(samples: np.array): Initializes the MovingMoments object with the given samples.
        dimension() -> int: Returns the dimension of the samples.
        update(old_sample: np.array, new_sample: np.array): Updates the moving moments with a new sample.
        mean() -> np.array: Returns the mean of the samples.
        covariance() -> np.array: Returns the covariance matrix of the samples.
    """

    def __init__(self, samples: np.array):
        """
        Initializes the MovingMoments object with the given samples.

        Args:
            samples (np.array): The samples to compute the moving moments on.
        """
        self._mean = np.mean(samples, axis=1, keepdims=True)
        self._covariance = np.cov(samples)
        self.n_samples = samples.shape[1]
    
    @property
    def dimension(self) -> int:
        """
        Returns the dimension of the samples.

        Returns:
            int: The dimension of the samples.
        """
        return len(self._mean)

    def update(self, old_sample: np.array, new_sample: np.array):
        """
        Updates the moving moments with a new sample.

        Args:
            old_sample (np.array): The old sample to be replaced.
            new_sample (np.array): The new sample to be added.
        """
        old_sample = np.reshape(old_sample, (self.dimension, 1))
        new_sample = np.reshape(new_sample, (self.dimension, 1))

        mean_old = self._mean
        covariance_old = self._covariance

        self._mean = mean_old + (new_sample - old_sample) / self.n_samples
        self._covariance = covariance_old + 1 / (self.n_samples -1) * \
            (new_sample @ new_sample.T - old_sample @ old_sample.T - \
            mean_old @ (new_sample.T - old_sample.T) - \
            (new_sample - old_sample) @ self._mean.T)
    
    @property
    def mean(self) -> np.array:
        """
        Returns the mean of the samples.

        Returns:
            np.array: The mean of the samples.
        """
        return self._mean
    
    @property
    def covariance(self) -> np.array:
        """
        Returns the covariance matrix of the samples.

        Returns:
            np.array: The covariance matrix of the samples.
        """
        return self._covariance


class MovingCovarianceRatio:
    """
    Calculates the moving covariance ratio for a given set of samples.

    Attributes:
        samples (np.array): The array of samples used for calculating the moving covariance ratio.
        moments_total (MovingMoments): The moving moments object for the total set of samples.
        moments_before (MovingMoments): The moving moments object for the samples before the middle sample.
        moments_after (MovingMoments): The moving moments object for the samples after the middle sample.

    Methods:
        __init__(samples: np.array): Initializes the MovingCovarianceRatio object with the given samples.
        update(new_sample: np.array): Updates the moving covariance ratio with a new sample.
        get_divergence() -> float: Calculates the divergence of the moving covariance ratio.
    """

    def __init__(self, samples: np.array):
        self.samples = CircularArray(samples)

        if len(samples.shape) == 1:
            samples = samples[np.newaxis, :]
 
        self.moments_total = MovingMoments(samples)
        self.moments_before = MovingMoments(samples[:, :samples.shape[1]//2])
        self.moments_after = MovingMoments(samples[:, samples.shape[1]//2:])

    def update(self, new_sample: np.array):
        """
        Updates the moving covariance ratio with a new sample.

        Args:
            new_sample (np.array): The new sample to be added to the moving covariance ratio calculation.
        """
        middle_sample = self.samples.get_middle()
        old_sample = self.samples.get_first()

        self.moments_total.update(old_sample=old_sample, new_sample=new_sample)
        self.moments_before.update(old_sample=old_sample, new_sample=middle_sample)
        self.moments_after.update(old_sample=middle_sample, new_sample=new_sample)

        self.samples.add(new_sample)

    def get_divergence(self) -> float:
        """
        Calculates the divergence of the moving covariance ratio.

        Returns:
            float: The divergence of the moving covariance ratio.
        """
        I = 1e-6 * np.eye(self.samples.dimension)

        y_total = np.linalg.slogdet(self.moments_total.covariance + I)[1]
        y_before = np.linalg.slogdet(self.moments_before.covariance + I)[1]
        y_after = np.linalg.slogdet(self.moments_after.covariance + I)[1]

        mean_before = self.moments_before.mean
        mean_after = self.moments_after.mean

        #return y_total - 0.5 * (y_before + y_after) + 0.5 * np.log(np.linalg.norm(mean_after - mean_before))
        return y_total - 0.5 * (y_before + y_after)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


# class PrioritizedReplayBuffer:
#     def __init__(self, capacity, device, alpha=0.4, beta=0.4, beta_increment_per_sampling=1e-4, epsilon=1e-6):
#         self.capacity = capacity
#         self.device = device
#         self.tree = SumTree(capacity)

#         self.alpha = alpha
#         self.beta = beta
#         self.beta_increment_per_sampling = beta_increment_per_sampling
#         self.epsilon = epsilon

#     def store(self, state, action, reward, next_state, done):
#         num_envs = len(done)
#         for i in range(num_envs):
#             transition = (
#                 {key: state[key][i] for key in state.keys()},
#                 action[i],
#                 reward[i],
#                 {key: next_state[key][i] for key in next_state.keys()},
#                 done[i]
#             )
#             # inizialmente alta priorità
#             max_p = np.max(self.tree.tree[-self.capacity:]) if np.any(self.tree.tree[-self.capacity:]) else 10.0 
#             self.tree.add(max_p, transition)

#     def sample(self, batch_size):
#         batch = []
#         idxs = []
#         segment = self.tree.total / batch_size
#         priorities = []

#         for i in range(batch_size):
#             s = random.uniform(segment * i, segment * (i + 1))
#             idx, p, data = self.tree.get(s)
#             batch.append(data)
#             idxs.append(idx)
#             priorities.append(p)

#         sampling_probabilities = np.array(priorities) / self.tree.total
#         is_weight = np.power(self.capacity * sampling_probabilities, -self.beta)
#         is_weight /= is_weight.max()

#         state_batch = {}
#         next_state_batch = {}

#         for key in batch[0][0].keys():
#             state_batch[key] = torch.tensor(
#                 [transition[0][key] for transition in batch],
#                 dtype=torch.float32,
#                 device=self.device
#             )
#             next_state_batch[key] = torch.tensor(
#                 [transition[3][key] for transition in batch],
#                 dtype=torch.float32,
#                 device=self.device
#             )

#         action_batch = torch.tensor(
#             [transition[1] for transition in batch],
#             dtype=torch.int64,
#             device=self.device
#         )
#         reward_batch = torch.tensor(
#             [transition[2] for transition in batch],
#             dtype=torch.float32,
#             device=self.device
#         )
#         done_batch = torch.tensor(
#             [transition[4] for transition in batch],
#             dtype=torch.float32,
#             device=self.device
#         )
#         is_weight = torch.tensor(is_weight, dtype=torch.float32, device=self.device)

#         self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

#         return {
#             'state': state_batch,
#             'action': action_batch,
#             'reward': reward_batch,
#             'next_state': next_state_batch,
#             'done': done_batch,
#             'weights': is_weight,
#             'idxs': idxs
#         }

#     def update_priorities(self, idxs, td_errors):
#         td_errors = td_errors.detach().cpu().numpy()
#         new_priorities = np.abs(td_errors) + self.epsilon
#         for idx, p in zip(idxs, new_priorities):
#             self.tree.update(idx, p)

#     @property
#     def is_full(self):
#         return len([x for x in self.tree.data if x is not None]) >= self.capacity

#     def __len__(self):
#         return len([x for x in self.tree.data if x is not None])
    


# class ReplayBuffer:
#     def __init__(self, capacity, device):
#         self.capacity = capacity
#         self.device = device
#         self.buffer = []
#         self.position = 0
    
#     def store(self, state, action, reward, next_state, done):
#         """
#         Salva transizioni singole per ogni env.
#         Gli input devono essere batch con shape (N_ENVS, ...)
#         """
#         num_envs = len(done)

#         for i in range(num_envs):
#             transition = (
#                 {key: state[key][i] for key in state.keys()},
#                 action[i],
#                 reward[i],
#                 {key: next_state[key][i] for key in next_state.keys()},
#                 done[i]
#             )
            
#             if len(self.buffer) < self.capacity:
#                 self.buffer.append(None)
            
#             self.buffer[self.position] = transition
#             self.position = (self.position + 1) % self.capacity
    
#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
        
#         state_batch = {}
#         next_state_batch = {}
        
#         for key in batch[0][0].keys():
#             state_batch[key] = torch.tensor(
#                 [transition[0][key] for transition in batch],
#                 dtype=torch.float32,
#                 device=self.device
#             )
            
#             next_state_batch[key] = torch.tensor(
#                 [transition[3][key] for transition in batch],
#                 dtype=torch.float32,
#                 device=self.device
#             )
        
#         action_batch = torch.tensor(
#             [transition[1] for transition in batch],
#             dtype=torch.int64,
#             device=self.device
#         )
        
#         reward_batch = torch.tensor(
#             [transition[2] for transition in batch],
#             dtype=torch.float32,
#             device=self.device
#         )
        
#         done_batch = torch.tensor(
#             [transition[4] for transition in batch],
#             dtype=torch.float32,
#             device=self.device
#         )
        
#         return {
#             'state': state_batch,
#             'action': action_batch,
#             'reward': reward_batch,
#             'next_state': next_state_batch,
#             'done': done_batch
#         }
    
#     @property
#     def is_full(self):
#         return len(self.buffer) >= self.capacity
    
#     def __len__(self):
#         return len(self.buffer)


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0

    def store(self, state, action, reward, next_state, done):

        # numero di ambienti paralleli
        num_envs = state.shape[0]

        for i in range(num_envs):
            s = state[i]
            a = action[i]
            r = reward[i]
            s_next = next_state[i]
            d = done[i]

            transition = (s, a, r, s_next, d)

            if len(self.buffer) < self.capacity:
                self.buffer.append(None)  # espando fino alla capacità

            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            raise ValueError(f"Batch size {batch_size} larger than current buffer size {len(self.buffer)}")

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        def to_tensor(x, dtype):
            arr = np.stack(x, axis=0)
            return torch.as_tensor(arr, dtype=dtype, device=self.device)

        state_batch = to_tensor(states, dtype=torch.float32)
        next_state_batch = to_tensor(next_states, dtype=torch.float32)

        action_dtype = torch.int64 if np.issubdtype(np.array(actions).dtype, np.integer) else torch.float32
        action_batch = to_tensor(actions, dtype=action_dtype)

        reward_batch = to_tensor(rewards, dtype=torch.float32)
        done_batch = to_tensor(dones, dtype=torch.float32)

        return {
            'state': state_batch,           
            'action': action_batch,        
            'reward': reward_batch,         
            'next_state': next_state_batch, 
            'done': done_batch              
        }

    @property
    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity

    def __len__(self) -> int:
        return len(self.buffer)
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, device: torch.device, alpha=0.4, beta=0.4, beta_increment_per_sampling=1e-4, epsilon=1e-6):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon

        self.tree = SumTree(capacity)

    def store(self, state, action, reward, next_state, done):
        # Batch da ambienti paralleli
        num_envs = state.shape[0]
        for i in range(num_envs):
            s = state[i]
            a = action[i]
            r = reward[i]
            s_next = next_state[i]
            d = done[i]

            transition = (s, a, r, s_next, d)
            max_p = np.max(self.tree.tree[-self.capacity:]) if np.any(self.tree.tree[-self.capacity:]) else 10.0
            self.tree.add(max_p, transition)

    def sample(self, batch_size: int):
        batch = []
        idxs = []
        priorities = []

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total
        is_weight = np.power(self.capacity * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)

        def to_tensor(x, dtype):
            arr = np.stack(x, axis=0)
            return torch.as_tensor(arr, dtype=dtype, device=self.device)

        state_batch = to_tensor(states, torch.float32)
        next_state_batch = to_tensor(next_states, torch.float32)
        action_dtype = torch.int64 if np.issubdtype(np.array(actions).dtype, np.integer) else torch.float32
        action_batch = to_tensor(actions, action_dtype)
        reward_batch = to_tensor(rewards, torch.float32)
        done_batch = to_tensor(dones, torch.float32)
        is_weight = torch.tensor(is_weight, dtype=torch.float32, device=self.device)

        return {
            'state': state_batch,
            'action': action_batch,
            'reward': reward_batch,
            'next_state': next_state_batch,
            'done': done_batch,
            'weights': is_weight,
            'idxs': idxs
        }

    def update_priorities(self, idxs, td_errors):
        td_errors = td_errors.detach().cpu().numpy()
        new_priorities = np.abs(td_errors) + self.epsilon
        for idx, p in zip(idxs, new_priorities):
            self.tree.update(idx, p)

    @property
    def is_full(self) -> bool:
        return len([x for x in self.tree.data if x is not None]) >= self.capacity

    def __len__(self) -> int:
        return len([x for x in self.tree.data if x is not None])

def get_parameters(dataset: str):

    if dataset == 'uschad' or dataset == 'uschad_nosit':
        return 30, 30, 3, 30
    elif dataset == 'babyecg':
        return 30, 15, 3, 92
    elif dataset == 'honeybee':
        return 30, 15, 3, 30
    elif dataset == 'ucihar':
        return 12, 8, 7, 100


    
                                    

