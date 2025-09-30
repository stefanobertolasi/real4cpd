
"""    
This module contains classes for segmentation tasks. The module provides classes for defining score models, segmenters, and optimizers for segmentation tasks. 

Classes:
    Segmenter: A class representing a segmenter for segmentation tasks.
    Loss: Abstract base class for defining loss functions used in segmentation.
    F1Loss: A class that calculates the F1 loss for a given set of scores and ground truth break points.
    ScoreModel: Abstract base class for score models used in segmentation.
    WaveletDecompositionModel: A class representing a wavelet decomposition model for segmentation.
    Optimizer: Abstract base class for optimizers used in segmentation.
    BayesianOptimizer: A class that performs Bayesian optimization to find the optimal weights for a segmenter.
    MangoOptimizer: A class that optimizes the weights of a segmenter using the Mango optimization algorithm.
    SupervisedDomain: A class representing a supervised domain for segmentation.

Segmenter:
    A class representing a segmenter for segmentation tasks. The segmenter is used to perform segmentation on a datastream using a score model. The segmenter can be compiled with a loss function and an optimizer to find the optimal weights for the score model. 

Loss:
    Abstract base class for defining loss functions used in segmentation.

F1Loss:
    A class that calculates the F1 loss for a given set of scores and ground truth break points. The F1 loss is defined as 1-F1, where F1 is the maximum F1 score that can be achieved by varying the threshold value.

ScoreModel:
    Abtract base class for score models used in segmentation. The score model is used to calculate the score for a segmentation task. In particular, the method get_profile() return a score profile, that has the same length of the datastream to be segmented. A peak in the score profile indicates a change point in the datastream.

 WaveletDecompositionModel:
    A class representing a wavelet decomposition model for segmentation. It performs wavelet decomposition on the input samples using the specified level and wavelet type (db3). For each subband obtained from the decomposition, it calculates a profile using the NormalDivergenceProfile, resamples it, applies a power transform, and stores it. It then computes an average score profile across all subbands and sets this as both the average and the score profile of the model. 

Optimizer:
    Abstract base class for optimizers used in segmentation. The optimizer is devoted to find the optimal weights for a segmenter.

BayesianOptimizer:
    A class that performs Bayesian optimization to find the optimal weights for a segmenter.

MangoOptimizer:
    The MangoOptimizer class is used to optimize the weights of a segmenter using the Mango optimization algorithm.

SupervisedDomain:
    Represents a supervised domain for segmentation. This class is designed to manage a domain of supervised intervals for segmentation tasks. It encapsulates the concept of a domain that is partially labeled or supervised, where certain ranges (intervals) of the domain are marked as supervised regions. This class provides a structured way to handle these intervals, check for membership, add new intervals, resolve overlaps between them, and generate a boolean array indicating the supervised samples.

"""

from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pywt
import cma
from scipy.signal import peak_prominences, find_peaks
from segmentation.profiles import NormalDivergenceProfile, ProminenceProfile, AverageScoreProfile
from segmentation.profiles import PowerTransformProfile, ResampleProfile, LSEProfile, SoftmaxProfile, SigmoidProfile
from mango.tuner import Tuner
from mango import scheduler
from scipy.stats import uniform, randint, beta
from scipy.optimize import differential_evolution
import pickle
import json
import gymnasium as gym


class Loss(ABC):
    """
    Abstract base class for defining loss functions used in segmentation.
    """

    @abstractmethod
    def __call__(self, score: np.array, gt_break_points: List[int]) -> float:
        """
        Calculate the loss value given the predicted score and ground truth break points.

        Args:
            score (np.array): The predicted score.
            gt_break_points (List[int]): The ground truth break points.

        Returns:
            float: The calculated loss value.
        """
        pass

    @abstractmethod
    def get_optimal_threshold(self, score: np.array, gt_break_points: List[int]) -> float:
        """
        Calculate the optimal threshold value given the predicted score and ground truth break points.

        Args:
            score (np.array): The predicted score.
            gt_break_points (List[int]): The ground truth break points.

        Returns:
            float: The calculated optimal threshold value.
        """
        pass

    @abstractmethod
    def get_minimum_value(self) -> float:
        """
        Get the minimum possible value for the loss function.

        Returns:
            float: The minimum value for the loss function.
        """
        pass

class F1Loss(Loss):
    """
    F1Loss is a class that calculates the F1 loss for a given set of scores and ground truth break points. 
    The F1 loss is defined as 1-F1, where F1 is the maximum F1 score that can be achieved by varying the threshold value.
    """

    def __init__(self, threshold: float, peak_tolerance: int = 100):
        """
        Initializes the F1Loss object.

        Args:
            peak_tolerance (int): The tolerance value for matching peaks in the ground truth break points.
                                  Defaults to 100.
        """
        self.peak_tolerance = peak_tolerance
        self.threshold = threshold

    def _compute_best_f1(self, scores: np.array, gt_break_points: List[int]) -> Tuple[float, float]:
        """
        Computes the best F1 score and threshold value for a given set of scores and ground truth break points.

        Args:
            scores (np.array): The array of scores.
            gt_break_points (List[int]): The list of ground truth break points.

        Returns:
            Tuple[float, float]: A tuple containing the best F1 score and the corresponding threshold value.
        """

        threshold_values = scores[scores > 0]

        if len(threshold_values) == 0:
            return 0.0, 0.0
        
        threshold_values = np.unique(threshold_values)[::-1]

        best_f1 = 0
        max_fp = len(scores)
        best_index = 0

        for i_threshold, threshold in enumerate(threshold_values):

            # indices = np.where(scores >= threshold)[0]
            indices = np.where(scores > threshold)[0]
            false_positive = 0
            true_positive = 0 

            if len(indices) == 0:
                f1 = 0
            else:
                not_matched_positive = indices.copy()
                for p in gt_break_points:
                    if len(not_matched_positive) == 0:
                        break
                    if np.min(np.abs(not_matched_positive-p)) < self.peak_tolerance:
                        i = np.argmin(np.abs(not_matched_positive-p))
                        not_matched_positive = np.delete(not_matched_positive, i)

                false_positive = len(not_matched_positive)
                true_positive = len(indices) - false_positive

                if true_positive == 0:
                    f1 = 0
                else:
                    precision = true_positive / len(indices)
                    recall = true_positive / len(gt_break_points)

                    f1 = 2*precision * recall / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                max_fp = 2*len(gt_break_points) / best_f1 - 2*len(gt_break_points)
                best_index = i_threshold
            else:
                if false_positive > max_fp:
                    break

        
        if best_index+1 < len(threshold_values):
            best_threshold = (threshold_values[best_index] + threshold_values[best_index+1])/2
        else:
            best_threshold = (threshold_values[best_index] + threshold_values[best_index-1])/2
        # best_threshold = threshold_values[best_index]

        return 1 - best_f1, best_threshold
    
    def _compute_f1(self, scores: np.array, gt_break_points: List[int], threshold: float) -> Tuple[float, float]:
        """
        Computes the F1 score for a given set of scores, ground truth break points and a fixed threhsold.

        Args:
            scores (np.array): The array of scores.
            gt_break_points (List[int]): The list of ground truth break points.

        Returns:
            Tuple[float, float]: A tuple containing the F1 score and the corresponding threshold value.
        """

        indices = np.where(scores >= threshold)[0]
        false_positive = 0
        true_positive = 0

        if len(indices) == 0:
            f1 = 0
        else:
            not_matched_positive = indices.copy()
            for p in gt_break_points:
                if len(not_matched_positive) == 0:
                    break
                if np.min(np.abs(not_matched_positive-p)) < self.peak_tolerance:
                    i = np.argmin(np.abs(not_matched_positive-p))
                    not_matched_positive = np.delete(not_matched_positive, i)

            false_positive = len(not_matched_positive)
            true_positive = len(indices) - false_positive

            if true_positive == 0:
                f1 = 0
            else:
                precision = true_positive / len(indices)
                recall = true_positive / len(gt_break_points)

                f1 = 2*precision * recall / (precision + recall)
              
        return 1 - f1, threshold

    def __call__(self, score: np.array, gt_break_points: List[int], threshold: float) -> float:
        """
        Calculates the F1 loss for a given set of scores and ground truth break points.

        Args:
            score (np.array): The array of scores.
            gt_break_points (List[int]): The list of ground truth break points.

        Returns:
            float: The F1 loss value.
        """
        f1, threshold = self._compute_f1(score, gt_break_points, threshold)
        return f1, threshold

    def get_optimal_threshold(self, score: np.array, gt_break_points: List[int]) -> float:
        """
        Calculates the optimal threshold value for a given set of scores and ground truth break points.

        Args:
            score (np.array): The array of scores.
            gt_break_points (List[int]): The list of ground truth break points.

        Returns:
            float: The optimal threshold value.
        """
        _, threshold = self._compute_best_f1(score, gt_break_points)
        self.threshold = threshold
        return threshold
    
    def get_minimum_value(self) -> float:
        """
        Returns the minimum possible value for the F1 loss.

        Returns:
            float: The minimum value for the F1 loss.
        """
        return 0

class HingeMarginLoss(Loss):
    """
    AUCLoss is a class that calculates the rank-based surrogate loss for AUC using pairwise hing loss for a given set of scores and ground truth break points. 
    The AUC loss is defined as 1-AUC. This class has no get optimal threshold since no threshold is needed.
    """

    def __init__(self, threshold: float, peak_tolerance: int = 100):
        """
        Args:
            peak_tolerance (int): Tolerance window around each true break point.
        """
        self.peak_tolerance = peak_tolerance
        self.threshold = threshold
        self.margin = 0.1
        self.alpha = 0.8
    
    def __call__(self, scores: np.ndarray, gt_break_points: List[int], gt_normal_points: List[int]) -> float:

        gt_cp = np.array(gt_break_points, dtype=int)
        gt_non_cp = np.array(gt_normal_points, dtype=int)

        cp_scores = scores[gt_cp]
        cp_mis = np.maximum(0.0, self.threshold + self.margin - cp_scores)
        cp_loss = cp_mis.sum() if len(cp_mis)>0 else 0.0

        normal_scores = scores[gt_non_cp]
        normal_mis = np.maximum(0.0, normal_scores - (self.threshold - self.margin))
        normal_loss = normal_mis.sum() if len(normal_mis)>0 else 0.0

        loss = self.alpha*cp_loss + (1-self.alpha)*normal_loss
        return loss

    
    def get_optimal_threshold(self, score: np.array, gt_break_points: List[int]) -> float:
        """
        Calculate the optimal threshold value given the predicted score and ground truth break points.

        Args:
            score (np.array): The predicted score.
            gt_break_points (List[int]): The ground truth break points.

        Returns:
            float: The calculated optimal threshold value.
        """
        pass
    
    def get_minimum_value(self) -> float:
        """
        Returns the minimum possible value for the AUC loss.

        Returns:
            float: The minimum value for the AUC loss.
        """
        return 0.0
    
class ScoreModel(ABC):
    """
    Abstract base class for score models used in segmentation.
    """

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """
        Get the number of samples used in the score model.

        Returns:
            int: The number of samples.
        """
        pass

    @property
    @abstractmethod
    def weights(self) -> List[float]:
        """
        Get the weights used in the score model.

        Returns:
            List[float]: The weights.
        """
        pass

    @property
    @abstractmethod
    def windows(self) -> List[float]:
        """
        Get the weights used in the score model.

        Returns:
            List[float]: The weights.
        """
        pass

    @abstractmethod
    def get_score(self) -> np.array:
        """
        Calculate the score using the score model.

        Returns:
            np.array: The calculated score.
        """
        pass

    @abstractmethod
    def get_weights_constraints(self) -> List[Tuple[float, float]]:
        """
        Get the constraints for the weights used in the score model.

        Returns:
            List[Tuple[float, float]]: The constraints for the weights.
        """
        pass

    @abstractmethod
    def get_windows_constraints(self) -> List[Tuple[float, float]]:
        """
        Get the constraints for the weights used in the score model.

        Returns:
            List[Tuple[float, float]]: The constraints for the weights.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Get the constraints for the weights used in the score model.

        Returns:
            List[Tuple[float, float]]: The constraints for the weights.
        """
        pass

class WaveletDecompositionModel(ScoreModel):
    """
    A class representing a wavelet decomposition model for segmentation. 
    It performs wavelet decomposition on the input samples using the specified 
    level and wavelet type (db3). For each subband obtained from the decomposition, 
    it calculates a profile using the NormalDivergenceProfile, resamples it, 
    applies a power transform, and stores it. It then computes an average score 
    profile across all subbands and sets this as both the average and the score 
    profile of the model. Initial weights (that correspond to the exponents of
    the power transforms are set to 1 for each subband).

    Attributes:
        samples (np.array): The input samples for the model.
        level_wavelet (int): The level of wavelet decomposition.
        windows_size (int): The size of the windows used for profile calculations.
        average_profile (AverageScoreProfile): The average score profile of all subbands.
        score_profile (ProminenceProfile): The score profile of the model.
        _weights (List[float]): The weights assigned to each subband profile.
        _constraints (List[Tuple[float, float]]): The constraints for the weights.

    Methods:
        num_samples() -> int: Returns the number of samples.
        weights() -> List[float]: Returns the weights assigned to each subband profile.
        weights(w: List[float]): Sets the weights for each subband profile.
        get_score() -> np.array: Returns the score profile of the model.
        get_weights_constraints() -> List[Tuple[float, float]]: Returns the constraints for the weights.
    """

    def __init__(self, samples: List[float], level_wavelet: int = 7, window_size: int = 100):
        """
        Initializes a new instance of the WaveletDecompositionModel class.

        Args:
            samples (np.array): The input samples for the model.
            level_wavelet (int, optional): The level of wavelet decomposition. Defaults to 7.
            windows_size (int, optional): The size of the windows used for profile calculations. Defaults to 100.
        """
        self.samples = samples
        self.window_size = window_size
        self.level_wavelet = level_wavelet

        self.all_divergence_profiles = []
        all_subband_profile = []
        subbands = []
        weights = []
        windows = []
        windows_constraints = []
        weights_constraints = []

        subbands.append(samples)
        #level_max_wavelet = min(level_wavelet, int(np.floor(np.log2(samples.shape[0]/(2*window_size)))))
        xh = samples
        for i in range(level_wavelet):
            xh, xl = pywt.dwt(xh, wavelet='db3', axis=0)
            subbands.append(xh)
            subbands.append(xl)

        for subband in subbands:
            ws = window_size
            div = NormalDivergenceProfile(subband, window_size=ws) 
            self.all_divergence_profiles.append(div)
            windows.append(ws)
            windows_constraints.append((ws//2, 2*ws))
       
        self.all_resample_profiles = [ResampleProfile(profile, self.num_samples) for profile in self.all_divergence_profiles]
        self.all_power_profiles = []
        for profile in self.all_resample_profiles:
            w = 1 
            profile = PowerTransformProfile(profile, coeff=w)
            all_subband_profile.append(profile)
            weights.append(w)
            weights_constraints.append((0.1, 10))
        
        profile = AverageScoreProfile(all_subband_profile, np.ones_like(all_subband_profile))
        self.average_profile = profile
        profile = ProminenceProfile(profile)
        self.score_profile = profile
        self._weights = weights
        self._weights_constraints = weights_constraints
        self._windows = windows
        self._windows_constraints = windows_constraints
        self.all_subband_profile = all_subband_profile


    @property
    def num_samples(self) -> int:
        """
        Returns the number of samples.

        Returns:
            int: The number of samples.
        """
        return self.samples.shape[0]

    @property
    def windows(self) -> List[float]:
        """
        Returns the weights assigned to each subband profile.

        Returns:
            List[float]: The weights assigned to each subband profile.
        """
        return self._windows
    
    @property
    def weights(self) -> List[float]:
        """
        Returns the weights assigned to each subband profile.

        Returns:
            List[float]: The weights assigned to each subband profile.
        """
        return self._weights
    

    @weights.setter
    def weights(self, w: List[float]):
        """
        Sets the weights for each subband profile.

        Args:
            w (List[float]): The weights to be assigned to each subband profile.
        """
        self._weights = w
        for ww, profile in zip(self._weights, self.all_subband_profile):
            profile.coeff = ww

    @windows.setter
    def windows(self, ws: List[float]):
        """
        Sets the weights for each subband profile.

        Args:
            w (List[float]): The weights to be assigned to each subband profile.
        """
        # if len(ws) != len(self._windows):
        #     raise ValueError('Number of windows must remain the same')
        self._windows = ws
        # for profile in self.all_divergence_profiles:
        #     profile.window_size = ws
        for ww, profile in zip(self._windows, self.all_divergence_profiles):
            profile.window_size = ww


    def get_score(self) -> np.array:
        """
        Returns the score profile of the model.

        Returns:
            np.array: The score profile of the model.
        """
        return self.score_profile.get_profile()

    def get_allsubband_score(self) -> np.array:
        """
        Returns the set of scores of each subband
        
        Returns:
            np.array: The list of subband scores
        """
        all_subband_profile = []
        for subband in self.all_subband_profile:
            all_subband_profile.append(subband.get_profile())
        return all_subband_profile

    def get_weights_constraints(self) -> List[Tuple[float, float]]:
        """
        Returns the constraints for the weights.

        Returns:
            List[Tuple[float, float]]: The constraints for the weights.
        """
        return self._weights_constraints
    
    def get_windows_constraints(self) -> List[Tuple[float, float]]:
        """
        Returns the constraints for the weights.

        Returns:
            List[Tuple[float, float]]: The constraints for the weights.
        """
        return self._windows_constraints
    
    def reset(self):

        self._windows = [self.window_size for _ in self._windows]
        for ww, profile in zip(self._windows, self.all_divergence_profiles):
            profile.coeff = ww

        self._weights = [1 for _ in self._weights]
        for ww, profile in zip(self._weights, self.all_subband_profile):
            profile.coeff = ww
    
class Optimizer:
    """
    Abstract base class for optimizers. The optimizer is devoted to find the 
    optimal weights for a segmenter

    This class defines the interface for optimizers used in the segmentation process.
    Subclasses of `Optimizer` should implement the `update`, `set_segmenter`, and `set_loss` methods.

    Attributes:
        None

    Methods:
        update: Update the weights of the score model used by the segmenter and 
            return the updated weights.
        set_segmenter: Set the segmenter for the optimizer.
        set_loss: Set the loss function and minimum loss value for the optimizer.
    """

    @abstractmethod
    def update(self) -> List[float]:
        """
        Compute the new weights of the segmenter and return the computed weights

        Returns:
            List[float]: The computed weights.
        """
        pass

    @abstractmethod
    def set_segmenter(self, segmenter: "Segmenter"):
        """
        Sets the segmenter for the object.

        Args:
        - segmenter: An instance of the Segmenter class.

        """
        pass

    @abstractmethod
    def set_loss(self, loss: callable, minimum_loss_value: float = -np.inf):
        """
        Sets the loss function for the segmenter.

        Args:
            loss (callable): The loss function to be used for segmentation.
            minimum_loss_value (float): The minimum loss value allowed.

        """
        pass

class BayesianOptimizer(Optimizer):
    """
    BayesianOptimizer is a class that performs Bayesian optimization to find the optimal weights for a segmenter.

    Args:
        max_calls (int): The maximum number of function calls to be made during optimization. Default is 100.
        n_initial_points (int): The number of initial points to be used for optimization. Default is 10.
        verbose (bool): Whether to print verbose output during optimization. Default is True.
    """

    def __init__(self, max_calls: int = 100, n_initial_points: int = 10, verbose: bool = True):
        self.loss: Loss = None
        self.segmenter: 'Segmenter' = None
        self.max_calls = max_calls
        self.stored_points = []
        self.n_initial_points = n_initial_points
        self.verbose = verbose

    def set_segmenter(self, segmenter: 'Segmenter'):
        """
        Sets the segmenter to be used for optimization.

        Args:
            segmenter (Segmenter): The segmenter object to be used for optimization.
        """
        self.segmenter = segmenter
        self.stored_points = [self.segmenter.weights]

    def set_loss(self, loss: callable, minimum_loss_value: float = -np.inf):
        """
        Sets the loss function to be minimized during optimization.

        Args:
            loss (callable): The loss function to be minimized.
            minimum_loss_value (float): The minimum value of the loss function. Default is -inf.
        """
        self.loss = loss

    def update(self):
        """
        Performs Bayesian optimization to find the optimal weights for the segmenter.

        Returns:
            The optimal weights for the segmenter.
        """
        res = forest_minimize(self.loss,
                              self.segmenter.get_weights_constraints(),
                              n_calls=self.max_calls,
                              verbose=self.verbose,
                              x0=self.stored_points,
                              )
        all_x = res.x_iters
        all_y = res.func_vals
        _, stored_points = zip(*sorted(zip(all_y, all_x), key=lambda t: t[0]))
        self.stored_points = [p for p in stored_points[:self.n_initial_points]]

        return res.x

class OptunaOptimizer(Optimizer):
    def __init__(self, max_calls=20, verbose=False):
        self.loss_to_adapt = None
        self.segmenter = None
        self.params_names = []        
        self.params_space = {}
        self.max_calls = max_calls
        self.verbose = verbose
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    def set_segmenter(self, segmenter):
        self.segmenter = segmenter
        self.params_names = []
        self.params_space = {}

        # Windows
        for i, (low, high) in enumerate(segmenter.get_windows_constraints()):
            name = f"ws_{i}"
            self.params_names.append(name)
            self.params_space[name] = ("int", low, high)

        # Weights
        for i, (low, high) in enumerate(segmenter.get_weights_constraints()):
            name = f"w_{i}"
            self.params_names.append(name)
            self.params_space[name] = ("float", low, high)

        # Threshold
        self.params_names.append("threshold")
        self.params_space["threshold"] = ("float", 0.0, 5.0)

    def set_loss(self, loss, minimum_loss_value: float = -np.inf):
        self.loss_to_adapt = loss

    def objective(self, trial):
        params = {}
        for name, (ptype, low, high) in self.params_space.items():
            if ptype == "int":
                params[name] = trial.suggest_int(name, low, high)
            elif ptype == "float":
                params[name] = trial.suggest_float(name, low, high)

        ws = [params[k] for k in self.params_names if k.startswith("ws_")]
        weights = [params[k] for k in self.params_names if k.startswith("w_")]
        threshold = params["threshold"]

        loss_value, _ = self.loss_to_adapt(ws, weights, threshold)
        return loss_value

    def update(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.max_calls, show_progress_bar=self.verbose)

        best = study.best_params
        ws = [best[k] for k in self.params_names if k.startswith("ws_")]
        weights = [best[k] for k in self.params_names if k.startswith("w_")]
        threshold = best["threshold"]

        return ws, weights, threshold
    

class MangoOptimizer(Optimizer):
    """
    The MangoOptimizer class is used to optimize the weights of a segmenter using the Mango optimization algorithm.

    Attributes:
        loss (Loss): The loss function to be minimized.
        segmenter (Segmenter): The segmenter whose weights are being optimized.
        params_space (dict): A dictionary representing the parameter space for optimization.
        params_names (list): A list of parameter names.
        loss_to_adapt (callable): The loss function to be adapted for optimization.
        max_calls (int): The maximum number of function calls to be made during optimization.
        stored_points (list): A list of stored points for updating the optimizer.
        previuous_loss_value (float): The previous loss value.
        max_stored_points (int): The maximum number of stored points.

    Methods:
        update(): Updates the optimizer and returns the optimized weights.
        set_segmenter(segmenter: Segmenter): Sets the segmenter for optimization.
        set_loss(loss: callable, minimum_loss_value: float = -np.inf): Sets the loss function for optimization.
        objective(params_batch): Computes the objective values for a batch of parameter sets.
        early_stopping(results): Determines if early stopping criteria is met.

    """

    def __init__(self, max_calls=20):
        """
        Initializes a new instance of the MangoOptimizer class.

        Args:
            max_calls (int): The maximum number of function calls to be made during optimization. Default is 20.
        """
        self.loss: Loss = None
        self.segmenter: Segmenter = None
        self.params_space = dict()
        self.params_names = []
        self.loss_to_adapt = None
        self.max_calls = max_calls
        self.n_updates = 0
        self.stored_points = []
        self.previuous_loss_value = None
        self.max_stored_points = max_calls // 4
        self.verbose = False

    def update(self) -> np.array:
        """
        Updates the optimizer and returns the optimized weights.

        Returns:
            np.array: The optimized weights.

        """
        conf_dict = {
            'domain_size': 5000,
            'num_iteration': self.max_calls,
            'initial_custom': self.stored_points,
            'early_stopping': self.early_stopping,
        }
        
        tuner = Tuner(self.params_space, self.loss, conf_dict)
        res = tuner.minimize(self.verbose)

        self.previuous_loss_value = res['best_objective']

        _, stored_points = zip(*sorted(zip(res['objective_values'], res['params_tried']), key=lambda t: t[0]))
        stored_points = [p for p in stored_points]
        self.stored_points = stored_points[:min(self.max_stored_points, len(stored_points))]
        params = {}
        for k in self.params_names:
            params[k] = res['best_params'][k]
        return params

    def set_segmenter(self, segmenter: "Segmenter"):
        self.segmenter = segmenter

        self.params_names = []
        self.params_space = {}
        params = dict()

        windows = segmenter.windows
        windows_constraints = segmenter.get_windows_constraints()
        for i, c in enumerate(windows_constraints):
            name = f"win_{i}"
            self.params_names.append(name)
            self.params_space[name] = randint(c[0], c[1])
            params[name] = windows[i]
            
        weights = segmenter.weights
        weights_contraints = segmenter.get_weights_constraints()
        for i, c in enumerate(weights_contraints):
            name = f'wei_{i}'
            self.params_names.append(name)
            self.params_space[name] = uniform(c[0], c[1]-c[0])
            params[name] = weights[i]

        self.stored_points = [params]
        
    def set_loss(self, loss: callable, minimum_loss_value: float = -np.inf):
        """
        Sets the loss function for optimization.

        Args:
            loss (callable): The loss function to be optimized.
            minimum_loss_value (float, optional): The minimum loss value. Defaults to -np.inf.

        """
        self.loss_to_adapt = loss
        self.loss = self.objective
        self.previuous_loss_value = minimum_loss_value
    
    # @scheduler.parallel(n_jobs=1)
    def objective(self, params_batch):
        """
        Computes the objective values for a batch of parameter sets.

        Args:
            params_batch: A batch of parameter sets.

        Returns:
            list: The objective values for the parameter sets.

        """
        values = []
        for params in params_batch:
            windows = [w for k, w in params.items() if k.startswith('win_')]
            weights = [w for k, w in params.items() if k.startswith('wei_')]
            value = self.loss_to_adapt(windows, weights)
            values.append(value)
        return values

    def early_stopping(self, results):
        """
        Determines if early stopping criteria is met.

        Args:
            results: The optimization results.

        Returns:
            bool: True if early stopping criteria is met, False otherwise.

        """
        # results['best_objective'] < self.previuous_loss_value or
        return  results['best_objective'] == self.segmenter.loss.get_minimum_value()

class DifferentialOptimizer(Optimizer):

    def __init__(self, max_calls: int = 100, n_initial_points: int = 10):
        self.loss: Loss = None
        self.segmenter: 'Segmenter' = None
        self.max_calls = max_calls
        self.stored_points = []
        self.n_initial_points = n_initial_points

    def update(self):
        bounds = self.segmenter.get_windows_constraints() + self.segmenter.get_weights_constraints()
        self.num_windows = len(self.segmenter.get_windows_constraints())
        res = differential_evolution(func=self.loss,
                                     bounds=bounds,
                                     strategy="randtobest1bin",
                                     maxiter = self.max_calls,
                                     x0 = self.stored_points)
        
        x = res.x
        self.stored_points = x
        params = dict()

        for i in range(self.num_windows):
            name = f"win_{i}" 
            params[name] = x[i]

        for i in range(self.num_windows):
            name = f"wei_{i}" 
            params[name] = x[self.num_windows+i]
        return params

    
    def set_segmenter(self, segmenter: "Segmenter"):
        self.segmenter = segmenter

        self.params_names = []
        self.params_space = {}
        self.stored_points = []
        params = dict()

        windows = segmenter.windows
        
        for i, c in enumerate(windows):
            name = f"ws_{i}"
            self.params_names.append(name)
            params[name] = windows[i]
            self.stored_points.append(windows[i])
            

        weights = segmenter.weights
        for i, c in enumerate(weights):
            name = f'w_{i}'
            self.params_names.append(name)
            params[name] = weights[i]
            self.stored_points.append(weights[i])
        
        
        # threshold = segmenter.threshold
        # name = f'threshold'
        # self.params_names.append(name)
        # self.params_space[name] = beta(10, 10, 0, scale=5)
        # params[name] = threshold
        # self.stored_points.append(threshold)


    def set_loss(self, loss: callable, minimum_loss_value: float = -np.inf):
        """
        Sets the loss function for optimization.

        Args:
            loss (callable): The loss function to be optimized.
            minimum_loss_value (float, optional): The minimum loss value. Defaults to -np.inf.

        """
        self.loss_to_adapt = loss
        self.loss = self.objective
        self.previuous_loss_value = minimum_loss_value
    
    def objective(self, params):
        """
        Computes the objective values for a batch of parameter sets.

        Args:
            params_batch: A batch of parameter sets.

        Returns:
            list: The objective values for the parameter sets.

        """
        windows = params[:self.num_windows]
        weights = params[self.num_windows:]
        # threshold = params.get('threshold')
        # value, _ = self.loss_to_adapt(params[:-1], params[-1]) + (self.regularization**self.n_updates)*np.linalg.norm(params[:-1], ord=2)
        value = self.loss_to_adapt(windows, weights)
        return value
    
class CMAESOptimizer(Optimizer):

    def __init__(self, max_calls: int = 100, n_initial_points: int = 10):
        self.loss: Loss = None
        self.segmenter: 'Segmenter' = None
        self.max_calls = max_calls
        self.stored_points = []
        self.n_initial_points = n_initial_points

    def update(self):
        fun = self.loss
        self.num_windows = len(self.segmenter.get_windows_constraints())
        lower_bounds = []
        upper_bounds = []
        bounds = self.segmenter.get_windows_constraints() + self.segmenter.get_weights_constraints()
        for i in range(len(bounds)):
            lower_bounds.append(bounds[i][0])
            upper_bounds.append(bounds[i][1])
        bfun = cma.BoundDomainTransform(fun, [lower_bounds, upper_bounds])

        res = cma.fmin(objective_function=bfun,
                       x0=self.stored_points, 
                       sigma0=0.01,
                       options={
                           'maxiter':self.max_calls,
                           'verb_disp':0,
                           'verb_log':0,
                           'verbose':-9
                       })
        x = res[0]
        self.stored_points = x

        params = dict()

        for i in range(self.num_windows):
            name = f"win_{i}" 
            params[name] = x[i]

        for i in range(self.num_windows):
            name = f"wei_{i}" 
            params[name] = x[self.num_windows+i]
        return params

    
    def set_segmenter(self, segmenter: "Segmenter"):
        self.segmenter = segmenter

        self.params_names = []
        self.params_space = {}
        self.stored_points = []
        params = dict()

        windows = segmenter.windows
        
        for i, c in enumerate(windows):
            name = f"win_{i}"
            self.params_names.append(name)
            params[name] = windows[i]
            self.stored_points.append(windows[i])
            

        weights = segmenter.weights
        for i, c in enumerate(weights):
            name = f'wei_{i}'
            self.params_names.append(name)
            params[name] = weights[i]
            self.stored_points.append(weights[i])
        
        
        # threshold = segmenter.threshold
        # name = f'threshold'
        # self.params_names.append(name)
        # self.params_space[name] = beta(10, 10, 0, scale=5)
        # params[name] = threshold
        # self.stored_points.append(threshold)


    def set_loss(self, loss: callable, minimum_loss_value: float = -np.inf):
        """
        Sets the loss function for optimization.

        Args:
            loss (callable): The loss function to be optimized.
            minimum_loss_value (float, optional): The minimum loss value. Defaults to -np.inf.

        """
        self.loss_to_adapt = loss
        self.loss = self.objective
        self.previuous_loss_value = minimum_loss_value
    
    def objective(self, params):
        """
        Computes the objective values for a batch of parameter sets.

        Args:
            params_batch: A batch of parameter sets.

        Returns:
            list: The objective values for the parameter sets.

        """
        windows = params[:self.num_windows]
        weights = params[self.num_windows:]
        # threshold = params.get('threshold')
        # value, _ = self.loss_to_adapt(params[:-1], params[-1]) + (self.regularization**self.n_updates)*np.linalg.norm(params[:-1], ord=2)
        value = self.loss_to_adapt(windows, weights)
        return value
    
class SupervisedDomain:
    """
    Represents a supervised domain for segmentation tasks, where specific intervals 
    in a data sequence are marked as supervised.

    Attributes:
        nsamples (int): Total number of samples in the domain.
        intervals (List[List[int]]): List of intervals (as [start, end)) representing supervised regions.

    Methods:
        __init__(nsamples: int): Initializes the supervised domain with the total sample count.
        __contains__(x: int) -> bool: Checks if a given sample index belongs to a supervised interval.
        __len__() -> int: Returns the total number of supervised samples.
        add_interval(interval: List[int]): Adds a new interval and resolves overlaps.
        _resolve_overlaps(): Merges overlapping or contiguous intervals.
        get_supervised_indices() -> np.array: Returns a boolean array indicating supervised samples.
        get_unsupervised_indices() -> np.array: Returns a boolean array for unsupervised samples.
    """
    
    def __init__(self, nsamples: int):
        self.nsamples = nsamples
        self.intervals = []  # Each interval is represented as [start, end)

    def __contains__(self, x: int) -> bool:
        """
        Checks if a sample index x is within any supervised interval.
        
        Args:
            x (int): The sample index to check.
        
        Returns:
            bool: True if x is supervised, False otherwise.
        """
        return any(start <= x < end for start, end in self.intervals)
    
    def __len__(self) -> int:
        """
        Returns the total number of supervised samples by summing the length of each interval.
        
        Returns:
            int: Number of supervised samples.
        """
        return sum(end - start for start, end in self.intervals)

    def add_interval(self, interval: list[int]):
        """
        Adds a new interval to the supervised domain and merges overlapping intervals.
        The interval is clamped to the bounds [0, nsamples).
        
        Args:
            interval (List[int]): An interval [start, end) to add.
        """
        start = max(0, interval[0])
        end = min(self.nsamples, interval[1])
        if start < end:  # Only add valid intervals
            self.intervals.append([start, end])
            self._resolve_overlaps()

    def _resolve_overlaps(self):
        """
        Resolves any overlapping intervals in the domain and fix new ground truth break points according to added intervals
        """
        self.intervals.sort(key=lambda x: x[0])
        new_intervals = []
        for interval in self.intervals:
            if not new_intervals:
                new_intervals.append(interval)
            else:
                last_interval = new_intervals[-1]
                if interval[0] <= last_interval[1]:
                    last_interval[1] = max(last_interval[1], interval[1])
                else:
                    new_intervals.append(interval)
        self.intervals = new_intervals

    def get_supervised_indices(self) -> np.array:
        """
        Returns a boolean array indicating which samples are supervised.

        Returns:
            np.array: A boolean array indicating which samples are supervised.
        """
        indices = np.zeros(self.nsamples, dtype=bool)
        for interval in self.intervals:
            try:
                indices[interval[0]:interval[1]] = True
            except:
                print(interval[0], interval[1])
        return indices
    
    def get_unsupervised_indices(self) -> np.array:
        """
        Returns a boolean array indicating which samples are unsupervised.

        Returns:
            np.array: A boolean array indicating which samples are unsupervised.
        """
        indices = self.get_supervised_indices()
        return ~indices
    
    def reset(self):

        self.intervals=[]

class Segmenter:
    """
    A class representing a segmenter for segmentation tasks. The segmenter is used to perform segmentation on a datastream using a score model. The segmenter can be compiled with a loss function and an optimizer to find the optimal weights for the score model.

    Attributes:
        score_model (ScoreModel): The score model used by the segmenter.
        gt_break_points (List[int]): The ground truth break points.
        supervised_domain (SupervisedDomain): The supervised domain for the segmenter.
        weights (List[float]): The weights assigned to the score model.
        threshold (float): The threshold value for segmentation.
        extension_window (int): The extension window for supervised intervals.
        loss (Loss): The loss function used by the segmenter.
        optimizer (Optimizer): The optimizer used by the segmenter.

    """

    def __init__(self, score_model: ScoreModel, extension_window: int = 100):
        """
        Initializes a new instance of the Segmenter class.

        Args:
            score_model (ScoreModel): The score model used by the segmenter.
            extension_window (int, optional): The extension window for supervised intervals. Defaults to 100.
        """
        
        self.score_model: ScoreModel = score_model
        self.gt_break_points = []
        self.gt_normal_points = []
        self.supervised_domain = SupervisedDomain(score_model.num_samples)
        self.weights = score_model.weights
        self.windows = score_model.windows

        # self.threshold = np.sort(score_model.get_score())[-30]
        # self.threshold = np.quantile(score_model.get_score(), 0.99)
        self.threshold = self.set_initial_threshold()
        # self.threshold = 10
        self.extension_window = extension_window
        self.loss = None
        self.optimizer = None
    
    def reset(self):
        self.gt_break_points = []
        self.gt_normal_points = []
        
        self.weights = self.score_model.weights
        self.windows = self.score_model.windows
        
        self.supervised_domain.reset()
        
        self.optimizer.set_segmenter(self)

    def compile(self, loss: Loss, optimizer: Optimizer, environment: gym.Env=None):
        """
        Compiles the segmenter with a loss function and an optimizer. The loss function is used to calculate the loss value for the segmenter, while the optimizer is used to find the optimal weights for the score model.

        Args:
            loss (Loss): The loss function to be used for segmentation.
            optimizer (Optimizer): The optimizer to be used for optimization.
        """
        self.loss = loss
        optimizer.set_segmenter(self)
        optimizer.set_loss(self.loss_fun, loss.get_minimum_value())
        self.optimizer = optimizer

        if environment is not None:
            environment.set_segmenter(self)
            self.environment = environment

    def set_initial_threshold(self, q = 0.0):
        """
        Set the initial threshold using the curvature, completely vectorial.
        """

        scores = self.score_model.get_score()

        y = np.sort(scores[self.score_model.get_score()>0])[::-1]
        if len(y) < 5:
            return np.median(y) if len(y) > 0 else 0.0
        
        qval = np.percentile(y, 100*q)
        y = y[y >= qval]
        x = np.linspace(-1, 1, len(y))

        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())

        dy_dx = np.gradient(y_norm, x_norm)
        d2y_dx2 = np.gradient(dy_dx, x_norm)

        curvature = np.abs(d2y_dx2) / (1 + dy_dx**2)**1.5

        valid = curvature[2:-2]
        idx_offset = 2 + np.argmax(valid)

        threshold = y[idx_offset]
        return threshold
    
    @property
    def num_samples(self) -> int:
        """
        Returns the number of samples used in the segmenter.

        Returns:
            int: The number of samples.
        """
        return self.score_model.num_samples

    def loss_fun(self, windows: List[float], weights: List[float]):
        """
        Calculates the loss value for the segmenter using the given weights.

        Args:
            weights (List[float]): The weights to be used for segmentation.
        Returns:
            float: The calculated loss value.
        """
        score = self.get_supervised_score(windows, weights)
        gt_break_points = self.gt_break_points
        gt_normal_points = self.gt_normal_points 
        loss_value = self.loss(score, gt_break_points, gt_normal_points)
        return loss_value

    def get_optimal_threshold(self, score: np.array = None) -> float:
        """
        Calculates the optimal threshold value for the segmenter.

        Returns:
            float: The optimal threshold value.
        """
        if score is None:
            score = self.get_supervised_score()
        gt_break_points = self.gt_break_points
        return self.loss.get_optimal_threshold(score, gt_break_points)
        
    def get_score(self, windows: List[float] = None, weights: List[float] = None) -> np.array:
        """
        Calculates the score for the segmenter using the given weights.

        Args:
            weights (List[float], optional): The weights to be used for segmentation. Defaults to None. If None is provided, the weights stored in the segmenter are used.
        
        Returns:
            np.array: The calculated score.
        """
        if weights is None:
            weights = self.weights
        if windows is None:
            windows = self.windows
        self.score_model.weights = weights
        self.score_model.windows = windows
        return self.score_model.get_score()
    
    @property
    def samples(self) -> np.array:
        """
        Returns the samples used by the segmenter.

        Returns:
            np.array: The samples.
        """
        return self.score_model.samples
        
    def get_supervised_score(self, windows: List[float] = None, weights: List[float] = None) -> np.array:
        """
        Calculates the score for the segmenter using the given weights, considering only the supervised regions.

        Args:
            weights (List[float], optional): The weights to be used for segmentation. Defaults to None. If None is provided, the weights stored in the segmenter are used.

        Returns:
            np.array: The calculated score.
        """
        score = self.get_score(windows, weights) * self.supervised_domain.get_supervised_indices()
        return score
    
    def get_unsupervised_score(self,  windows: List[float] = None, weights: List[float] = None) -> np.array:
        """
        Calculates the score for the segmenter using the given weights, considering only the unsupervised regions.

        Args:
            weights (List[float], optional): The weights to be used for segmentation. Defaults to None. If None is provided, the weights stored in the segmenter are used.

        Returns:
            np.array: The calculated score.
        """
        score = self.get_score(windows, weights) * self.supervised_domain.get_unsupervised_indices()
        return score

    def update(self):
        """
        Updates the segmenter using the optimizer. The weights of the score model are updated using the optimizer, and the optimal threshold value is calculated. Weights are updated if and only if a new gt_break_point
        has been added to the ground_truth, if not, only the threshold optimization is runned.
        """

        params = self.optimizer.update()

        self.windows = [w for k, w in params.items() if k.startswith('win_')]
        self.weights = [w for k, w in params.items() if k.startswith('wei_')]
        # self.threshold = params.get('threshold')
        # self.threshold = self.get_optimal_threshold()

    def get_weights_constraints(self) -> List[Tuple[float, float]]:
        """
        Returns the constraints for the weights used in the segmenter.

        Returns:
            List[Tuple[float, float]]: The constraints for the weights.
        """
        return self.score_model.get_weights_constraints()

    def get_windows_constraints(self) -> List[Tuple[float, float]]:
        """
        Returns the constraints for the weights used in the segmenter.

        Returns:
            List[Tuple[float, float]]: The constraints for the weights.
        """
        return self.score_model.get_windows_constraints()

    def get_break_points(self) -> List[int]:
        """
        Returns the break points detected by the segmenter.

        Returns:
            List[int]: The detected break points.
        """
        bkps = np.where(self.get_unsupervised_score() >= self.threshold)[0]
        bkps = np.concatenate((bkps, self.gt_break_points))
        bkps = np.sort(np.unique(bkps)).astype(int)
        return bkps.tolist()

    def add_break_point_to_gt(self, break_point: int, supervised_interval: List[int]):
        """
        Adds a break point to the ground truth break points and the supervised domain.

        Args:
            break_point (int): The break point to add.
            supervised_interval (List[int]): The supervised interval for the break point.
        """
        if not (supervised_interval[0] <= break_point <= supervised_interval[1]):
            raise ValueError("The break point must be inside the supervised interval")

        # Aggiorna break points
        break_points = self.gt_break_points
        break_points.append(break_point)
        self.gt_break_points = sorted(set(break_points))  # Rimuove eventuali duplicati

        # Aggiorna normal points
        start = max(0, supervised_interval[0])
        end = min(self.num_samples, supervised_interval[1])
        normal_points = self.gt_normal_points
        normal_points.extend(
            [t for t in range(start, end) if t != break_point]
        )
        self.gt_normal_points = sorted(set(normal_points))

        self.supervised_domain.add_interval(supervised_interval)
    
    def add_supervised_interval(self, supervised_interval: List[int]): 
        """
        Adds a supervised interval to the segmenter.

        Args:
            supervised_interval (List[int]): The supervised interval to add.
        """
        start = max(0, supervised_interval[0])
        end = min(self.num_samples, supervised_interval[1])
        normal_points = self.gt_normal_points
        normal_points.extend(
            [t for t in range(start, end)]
        )
        self.gt_normal_points = sorted(set(normal_points))

        self.supervised_domain.add_interval(supervised_interval)

    def get_f1_score(self, break_points: List[int], gt_break_points: List[int]) -> Tuple[float, float, float]:
        """
        Calculates the precision, recall, and F1 score for the segmenter.        
        Args:
            scores (np.array): The scores of the segmenter.
            gt_break_points (List[int]): The ground truth break points.
        Returns:
            Tuple[float, float, float]: A tuple containing the precision, recall, and F1 score.
        """
        indices = break_points
        true_positive = 0
        false_positive = 0

        if len(indices) == 0:
            precision = 0
            recall = 0
        else:
            not_matched_positive = indices.copy()
            for p in gt_break_points:
                if len(not_matched_positive) == 0:
                    break
                if np.min(np.abs(not_matched_positive-p)) < self.loss.peak_tolerance:
                    i = np.argmin(np.abs(not_matched_positive-p))
                    not_matched_positive = np.delete(not_matched_positive, i)
                
            false_positive = len(not_matched_positive)
            true_positive = len(indices) - false_positive

            if true_positive == 0:
                precision = 0
                recall = 0
            else:
                precision = true_positive / len(indices)
                recall = true_positive / len(gt_break_points)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2*precision * recall / (precision + recall)
        
        return precision, recall, f1
    
    def get_margin_score(self, scores: np.ndarray, gt_break_points: List[int], gt_normal_points: List[int]) -> float:
        gt_cp = np.array(gt_break_points, dtype=int)
        gt_non_cp = np.array(gt_normal_points, dtype=int)

        cp_scores = scores[gt_cp]
        cp_score_pos = np.maximum(0.0, cp_scores - (self.threshold + self.loss.margin))
        cp_score_sum = cp_score_pos.sum() if len(cp_score_pos) > 0 else 0.0

        normal_scores = scores[gt_non_cp]
        normal_score_pos = np.maximum(0.0, (self.threshold - self.loss.margin) - normal_scores)
        normal_score_sum = normal_score_pos.sum() if len(normal_score_pos) > 0 else 0.0

        score = self.loss.alpha * cp_score_sum + (1 - self.loss.alpha) * normal_score_sum
        return score

    def saver_config(self, path_prefix):

        with open(f'{path_prefix}_segmenter.pkl', 'wb') as f:
            pickle.dump(self, f)

        config = {
            'level_wavelet': self.score_model.level_wavelet,
            'window_size': self.score_model.window_size,
            'threshold': self.threshold,
            'weights': self.weights
        }
        with open(f'{path_prefix}_config.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    @classmethod
    def load_from_config(cls, path, samples):
        with open(path, 'r') as f:
            config = json.load(f)
        
        score_model = WaveletDecompositionModel(samples, windows_size=config['window_size'], level_wavelet=config['level_wavelet'])
        score_model.weights = config['weights']
        segmenter = Segmenter(score_model=score_model)
        segmenter.threshold = config['threshold']

        return score_model, segmenter