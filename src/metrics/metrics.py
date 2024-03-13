import math
import os
from typing import Callable, Union
import warnings
import torch
import sklearn.metrics

from .medpy_metrics import hd95 as hausdorff_dist

import numpy as np
from matplotlib import pyplot as plt

from .metric_wrapper import Metric

import utils
from utils.framework import plotters

"""
Metrics generally should implement the following methods:
    calculate_batch: to be called at each batch fragment, should expect input
    evaluate_batch: to be called at the end of a batch
    evaluate_epoch: to be called at the end of an epoch

Each of them should return a dictionary containing the name of the metric (or several metrics) and its value.

NOTE: calculate_batch and evaluate_batch are needed for gradient accumulation.
"""

class ConfusionMatrix(Metric): # TODO: implement these for multiclass tasks
    """Counts the number of true positive, true negative, false positive and false negative pixels at a certain threshold."""
    
    PARAMS = dict(multilabel = False, ignore_nans = True)
    
    def __init__(self, _config_dict, threshold = 0.5, accumulate = True, *args, **kwargs):
        self.threshold = threshold
        
        self.multilabel = _config_dict['metrics/calculation/multilabel']
        self.idx_start = int(self.multilabel)
        
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.accumulate = accumulate
        if accumulate:
            self.acc_TP = 0
            self.acc_TN = 0
            self.acc_FP = 0
            self.acc_FN = 0
        
        # if `ignore_nans` is True, then the NaN values will not be counted in the CM entries
        # if it's False, then NaNs are considered negatives
        # and any prediction predicting an entry where the ground truth is NaN is considered correct
        self.nan_multiplicity = int(_config_dict['metrics/calculation/ignore_nans'])
        self.class_counts = 0

    def __str__(self):
        return str([[self.TP, self.FP], [self.FN, self.TN]])

    def calculate_batch(self, prediction, mask = None, label = None, cumulate = True, *args, **kwargs):
        y : torch.Tensor = mask if mask is not None else label
        y_hat = prediction.unsqueeze(-1)
        y = y.view(y_hat.shape).moveaxis(0, 1)
        
        y_positives = y == 1
        
        self.class_counts += y_positives.flatten(start_dim = self.idx_start).sum(axis = -1)
        num_nans = y.isnan().flatten(start_dim = self.idx_start).sum(axis = -1) * self.nan_multiplicity
        
        y_negatives = ~y_positives
        y_hat_positives = y_hat.detach().moveaxis(0, 1) >= self.threshold
        y_hat_negatives = ~y_hat_positives
        
        TP = torch.flatten(y_positives & y_hat_positives, start_dim = self.idx_start).sum(axis = -1)
        TN = torch.flatten(y_negatives & y_hat_negatives, start_dim = self.idx_start).sum(axis = -1) - num_nans
        FP = torch.flatten(y_negatives & y_hat_positives, start_dim = self.idx_start).sum(axis = -1)
        FN = torch.flatten(y_positives & y_hat_negatives, start_dim = self.idx_start).sum(axis = -1)

        if cumulate:
            self.TP += TP
            self.TN += TN
            self.FP += FP
            self.FN += FN

        if self.accumulate:
            self.acc_TP += TP
            self.acc_TN += TN
            self.acc_FP += FP
            self.acc_FN += FN

        return {f'true_positives_threshold_{self.threshold}': TP,
                f'false_positives_threshold_{self.threshold}': FP,
                f'true_negatives_threshold_{self.threshold}': TN,
                f'false_negatives_threshold_{self.threshold}': FN}
    
    def evaluate_batch(self, flush = True, *args, **kwargs):
        TP, TN, FP, FN = self.acc_TP, self.acc_TN, self.acc_FP, self.acc_FN
        if flush:
            self.acc_TP = 0
            self.acc_TN = 0
            self.acc_FP = 0
            self.acc_FN = 0

        return {f'true_positives_threshold_{self.threshold}': TP,
                f'false_positives_threshold_{self.threshold}': FP,
                f'true_negatives_threshold_{self.threshold}': TN,
                f'false_negatives_threshold_{self.threshold}': FN}

    def evaluate_epoch(self, flush = True, *args, **kwargs):
        TP, TN, FP, FN = self.TP, self.TN, self.FP, self.FN
        class_counts = self.class_counts
        if flush:
            self.TP = 0
            self.TN = 0
            self.FP = 0
            self.FN = 0
            self.class_counts = 0

        return {f'true_positives_threshold_{self.threshold}': TP,
                f'false_positives_threshold_{self.threshold}': FP,
                f'true_negatives_threshold_{self.threshold}': TN,
                f'false_negatives_threshold_{self.threshold}': FN,
                f'class_counts_threshold_{self.threshold}': class_counts}

class DerivedConfusionMatrixMetric(Metric):
    """Abstract metric for calculating other metrics based on the confusion matrix."""

    PARENT_METRIC = ConfusionMatrix

    def __init__(self, name, calculator = None, neutral_value = 0, accumulate = True,
                 threshold = 0.5, _config_dict = {}, *args, **kwargs):
        
        if _config_dict.get('multilabel', False):
            warnings.warn(f'{type(self)} is not meant for calculating multilabel {name}.')
        
        self.name = name
        if 'threshold' not in name:
            self.name = '_'.join((self.name, 'threshold', str(threshold)))

        self.calculator = calculator
        self.neutral = neutral_value
        self.num_batches = 0
        self.accumulate = accumulate

    def calculate_batch(self, parent_value, calculate = False, *args, **kwargs):
        # parent_value: confusion matrix entries
        if self.accumulate and not calculate:
            return {}
        else:
            self.num_batches += 1
            value = self.calculator(**{k: v.item() for k, v in parent_value.items()})

            if value == 'invalid':
                return {self.name: self.neutral}
            return {self.name: value}
    
    def evaluate_batch(self, parent_value, *args, **kwargs):
        return self.calculate_batch(parent_value, calculate = True)

    def evaluate_epoch(self, parent_value, flush = True, *args, **kwargs):
        if self.num_batches == 0:
            return {self.name: self.neutral}
        if flush:
            self.num_batches = 0
        value = self.calculator(**{k: v.item() for k, v in parent_value.items()})
        if value == 'invalid':
            value = self.neutral
        return {self.name: value}


class Accuracy(DerivedConfusionMatrixMetric):
    """Metric calculating accuracy for binary classification."""
    def __init__(self, accumulate = True, *args, **kwargs):
        
        def accuracy(true_positives, false_positives, true_negatives, false_negatives, **kwargs):
            total = true_positives + false_positives + true_negatives + false_negatives
            return (true_positives + true_negatives) / total

        super().__init__(name = 'accuracy', calculator = accuracy,
                         accumulate = accumulate, *args, **kwargs)

class BalancedAccuracy(DerivedConfusionMatrixMetric):
    """Metric calculating balanced accuracy ((TPR + TNR) / 2) for binary classification."""
    def __init__(self, accumulate = True, *args, **kwargs):
        
        def balanced_accuracy(true_positives, true_negatives, false_positives, false_negatives, **kwargs):
            P = true_positives + false_negatives
            N = false_positives + true_negatives
            try:
                if P == 0:
                    return true_negatives / N
                if N == 0:
                    return true_positives / P
            except ZeroDivisionError:
                return 'invalid'
            return (true_positives / P + true_negatives / N) / 2
        
        super().__init__(name = 'balanced_accuracy', calculator = balanced_accuracy,
                         accumulate = accumulate, *args, **kwargs)

class Sensitivity(DerivedConfusionMatrixMetric):
    """Metric calculating sensitivity (true positive rate)."""
    def __init__(self, *args, **kwargs):

        def TPR(true_positives, false_negatives, **kwargs):
            P = true_positives + false_negatives
            if P == 0:
                return 'invalid'
            return true_positives / P
        
        super().__init__(name = 'sensitivity', calculator = TPR, *args, **kwargs)

class Specificity(DerivedConfusionMatrixMetric):
    """Metric calculating specificity (true negative rate)."""

    def __init__(self, *args, **kwargs):

        def TNR(false_positives, true_negatives, **kwargs):
            N = true_negatives + false_positives
            if N == 0:
                return 'invalid'
            return true_negatives / N
        
        super().__init__(name = 'specificity', calculator = TNR, *args, **kwargs)

class Precision(DerivedConfusionMatrixMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name = 'precision', calculator = self.calculator, *args, **kwargs)
        
    def calculator(self, true_positives, false_positives, *args, **kwargs):
        try:
            return true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            return 'invalid'

class TverskyIndex(DerivedConfusionMatrixMetric): # TODO: not always add epsilon for fully accurate comparison
    """
    General object for metrics based on the Tversky index.

    Parameters:
        name: name of metric
        weight_of_tps: coefficient of true positives (both in the numerator and denominator) when calculating the index
        weight_of_fps, weight_of_fns: coefficients of false positives and true negatives when calculating the index
        eps: smoothing value to avoid divison by zero
        accumulate: whether to expect gradient accumulation 
    """
    def __init__(self, name, weight_of_tps = 1, weight_of_fps = 1, weight_of_fns = 1,
                 eps = 1, accumulate = True, *args, **kwargs):
        
        def tversky_index(true_positives, false_positives, false_negatives, **kwargs):
            if true_positives + false_positives + false_negatives == 0:
                return 'invalid'
            num = weight_of_tps * true_positives
            denom = weight_of_tps * true_positives + weight_of_fps * false_positives + weight_of_fns * false_negatives
            return (num + eps) / (denom + eps)

        super().__init__(name = name, calculator = tversky_index, neutral_value = 1,
                         accumulate = accumulate, *args, **kwargs)
        
class DiceIndex(TverskyIndex):
    def __init__(self, eps = 1, accumulate = True, *args, **kwargs):
        super().__init__(
            name = 'dice_index',
            weight_of_tps = 2,
            weight_of_fps = 1,
            weight_of_fns = 1,
            eps = eps,
            accumulate = accumulate,
            *args, **kwargs
        )

class JaccardIndex(TverskyIndex):
    def __init__(self, eps = 1, accumulate = True, *args, **kwargs):
        super().__init__(
            name = 'jaccard_index',
            weight_of_tps = 1,
            weight_of_fps = 1,
            weight_of_fns = 1,
            eps = eps,
            accumulate = accumulate,
            *args, **kwargs
        )

class MCC(DerivedConfusionMatrixMetric):
    """Implements the Matthews correlation coefficient."""
    def __init__(self, *args, **kwargs):
        super().__init__('mcc', self.calculator, *args, **kwargs)
    
    def calculator(self, true_positives, false_positives, false_negatives, true_negatives, **kwargs):
        num = true_positives * true_negatives - false_positives * false_negatives
        
        p_real = true_positives + false_negatives
        p_pred = true_positives + false_positives
        n_real = true_negatives + false_positives
        n_pred = true_negatives + false_negatives
        denom_sq = p_real * p_pred * n_real * n_pred

        if denom_sq == 0:
            return 'invalid'
        
        return num / math.sqrt(denom_sq)

class ModifiedHausdorffDistance(Metric):
    """
    Metric calculating the average modified Hausdorff distance between the mask and the prediction. (The modified Hausdorff distance of point x from set S is the 95th percentile of d(x, y) where y goes through all points in S.)
    """
    def __init__(self, threshold = 0.5, accumulate = True, *args, **kwargs):
        self.threshold = threshold
        self.accumulate = accumulate

        self.y_hat, self.y = np.array([]), np.array([])

        self.num_batches = 0
        self.value = 0
    
    def concatenate(self, cum_y, y):
        y_ = y if isinstance(y, np.ndarray) else y.cpu().detach().numpy()
        if len(cum_y) == 0:
            return y_
        return np.concatenate((cum_y, y_), axis = 0)

    def calculate_batch(self, prediction, mask, cumulate = True, *args, **kwargs):
        y_hat = (prediction >= self.threshold).int()
        
        if not self.accumulate:
            self.y = mask.cpu().detach().numpy()
            self.y_hat = y_hat.cpu().detach().numpy()
            return self.evaluate_batch(cumulate, *args, **kwargs)
        
        self.y = self.concatenate(self.y, mask)
        self.y_hat = self.concatenate(self.y_hat, y_hat)
        return {}
    
    def calculate_distances(self):
        value = 0
        num_samples = 0
        self.y = self.y.reshape(self.y_hat.shape)
        for y, y_hat in zip(self.y, self.y_hat):
            if np.all(y == 0) or np.all(y_hat == 0):
                continue
            num_samples += 1
            value += hausdorff_dist(y, y_hat)
        if num_samples != 0:
            return value / num_samples
        return 0

    def evaluate_batch(self, cumulate = True, flush = True, *args, **kwargs):
        value = self.calculate_distances()
        
        if value != 0 and cumulate:
            self.num_batches += 1
            self.value += value
        
        if flush:
            self.y_hat, self.y = np.array([]), np.array([])
        
        return {f'modified_hausdorff_distance_threshold_{self.threshold}': value}
    
    def evaluate_epoch(self, flush = True, average = True, *args, **kwargs):
        if self.num_batches == 0:
            value = 0
        else:
            value = self.value
            if average:
                value = value / self.num_batches
            if flush:
                self.value, self.batches = 0, 0
        return {f'modified_hausdorff_distance_threshold_{self.threshold}': value}


class AUC(Metric):
    """
    General metric for calculating areas under different curves.

    Parameters:
        calculator: a function calculating the area based on the predictions and true values
        name: name of the metric
        accumulate: whether to expect gradient accumulation
    """
    def __init__(self,
                 calculator : Callable[[np.ndarray, np.ndarray], float],
                 name : str,
                 accumulate : bool = True,
                 *args, **kwargs):
        self.value = 0
        self.num_batches = 0
        self.calculator = calculator
        self.name = name

        self.accumulate = accumulate
        if accumulate:
            self.y, self.y_hat = np.array([]), np.array([])

    def concatenate(self, cum_y : np.ndarray, y : Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        y_ = y.cpu().detach().numpy().flatten() if isinstance(y, torch.Tensor) else y
        return np.concatenate((cum_y, y_))

    def calculate_batch(self, prediction, mask = None, label = None, cumulate = True, *args, **kwargs):
        y = mask if mask is not None else label
        if not self.accumulate:
            self.y_hat = prediction
            self.y = y
            return self.evaluate_batch(cumulate, *args, **kwargs)
        
        self.y = self.concatenate(self.y, y)
        self.y_hat = self.concatenate(self.y_hat, prediction)
    
    def evaluate_batch(self, cumulate = True, flush = True, *args, **kwargs):
        y, y_hat = self.y, self.y_hat
        if flush:
            self.y_hat, self.y = np.array([]), np.array([])
        if len(y) == 0 or np.all(y == 0) or np.all(y == 1):
            value = 0
        else:
            value = self.calculator(y, y_hat)
        
            if cumulate:
                self.num_batches += 1
                self.value += value
        
        return {self.name: value}
    
    def evaluate_epoch(self, flush = True, average = True, *args, **kwargs):
        if self.num_batches == 0:
            return {self.name: 0}
        value = self.value
        if average:
            value = value / self.num_batches
        if flush:
            self.value, self.num_batches = 0, 0
        return {self.name: value}

class AUROC(AUC):
    """Area under receiving operating characteristics (ROC) curve."""
    def __init__(self, *args, **kwargs):
        super().__init__(
            calculator = sklearn.metrics.roc_auc_score,
            name = 'area_under_roc',
            *args, **kwargs
        )

class AveragePrecision(AUC):
    """Area under the precision-recall curve."""
    def __init__(self, *args, **kwargs):
        super().__init__(
            calculator = sklearn.metrics.average_precision_score,
            name = 'average_precision_score',
            *args, **kwargs
        )

class Curve(Metric):
    """
    Object to calculate and save curves at certain epochs during a run.

    Parameters:
        neptune_run: a neptune run object
        neptune_save_path: str; where to log the curve in the run
        validate: whether a validation set will be used
        train_colour: colour of the training curve
        val_colour: colour of the validation curve
        accumulate: whether to expect gradient accumulation
        dir_name: str; name of the directory where the curves will be saved at if log_to_device is True
        exp_name: str; name of the experiment
        _config_dict: config dict specifying the hyperparameters
    """

    PARAMS = {
        'calculate curves at': {
            'argument name': 'active_epochs',
            'default': 'last'
            },
        'number of batches to sample curves from': {
            'argument name': 'num_batches',
            'default': 5
            }
    }

    def __init__(self, neptune_run = None, neptune_save_path = '', train_colour = 'blue',
                 val_colour = 'orange', accumulate = True, validate = True,
                 dir_name = '', exp_name = '', _config_dict = None, *args, **kwargs):
        
        metric_params = _config_dict['metrics/calculation']

        self.run = neptune_run
        self.number_of_batches = metric_params['number of batches to sample curves from']
        self.train_colour = train_colour
        self.val_colour = val_colour
        self.to_validate = validate
        self.log_to_device = _config_dict['meta/technical/log to device']
        self.log_to_neptune = _config_dict['meta/technical/log to neptune']
        self.accumulate = accumulate

        self.active_epochs = metric_params['calculate curves at']
        # convert the list of active epochs the a list containing ints
        if isinstance(self.active_epochs, (str, int)):
            self.active_epochs = [self.active_epochs]
        self.active_epochs = list(self.active_epochs)
        self.do_last = 'last' in self.active_epochs
        if self.do_last:
            num_epochs = _config_dict['experiment/number of epochs']
            self.active_epochs = list(map(lambda x: num_epochs if x == 'last' else x, self.active_epochs))
        
        self.extensions = _config_dict.get_str_tuple('meta/technical/export_plots_as')
        self.extensions = utils.get_extensions(self.extensions)

        # initialsie counters and other inner states
        self.y, self.y_hat = np.array([]), np.array([])
        self.epoch_idx = 1
        self.batches_calculated = 0
        self.train = True

        if accumulate:
            self.curr_y, self.curr_y_hat = np.array([]), np.array([])

        if self.run:
            self.neptune_save_path = neptune_save_path
        
        if self.log_to_device:
            save_dest = _config_dict['meta/technical/absolute path']
            self.save_path = f'{save_dest}{exp_name}/{dir_name}'
            os.mkdir(self.save_path)

    def concatenate(self, cum_y, y):
        y_ = y if isinstance(y, np.ndarray) else y.cpu().detach().numpy().flatten()
        return np.concatenate((cum_y, y_))
    
    def calculate_batch(self, prediction, mask = None, label = None, last = False, *args, **kwargs):
        if not self.epoch_idx in self.active_epochs and not (last and self.do_last):
            return {}
        if self.batches_calculated == self.number_of_batches:
            return {}
        
        y = mask if mask is not None else label
        if not self.accumulate:
            self.y_hat = self.concatenate(self.y_hat, prediction)
            self.y = self.concatenate(self.y, y)
            return self.evaluate_batch(*args, **kwargs)
        self.curr_y_hat = self.concatenate(self.curr_y_hat, prediction)
        self.curr_y = self.concatenate(self.curr_y, y)
        return {}
    
    def evaluate_batch(self, train, last = False, *args, **kwargs):
        # note whether the epoch is a train or validation loop
        self.train = train
        if not self.epoch_idx in self.active_epochs and not (last and self.do_last):
            return {}
        if self.batches_calculated == self.number_of_batches:
            return {}
        if np.any(self.curr_y == 1): 
            self.y = self.concatenate(self.y, self.curr_y)
            self.y_hat = self.concatenate(self.y_hat, self.curr_y_hat)
            self.batches_calculated += 1
        self.curr_y, self.curr_y_hat = np.array([]), np.array([])
        return {}
    
    def evaluate_epoch(self, last = False, *args, **kwargs):
        if last and self.train:
            self.epoch_idx -= 1
        if len(self.y) > 0:
            self.save()
            self.y, self.y_hat = np.array([]), np.array([])
        if not self.train or not self.to_validate:
            # the epoch index only changes after a validation loop
            # or if there is no validation dataset    
            self.epoch_idx += 1
        self.batches_calculated = 0
        return {}

    def save(self):
        # NOTE: this method should be implemented for each curve metric separately
        pass

class ROCCurve(Curve): # TODO: fancy plotting
    """
    Object to calculate and save ROC curves at certain epochs during a run. The plots may be saved to a 'ROC curves' subdirectory to the main directory and to Neptune.

    Parameters:
        neptune_run: a neptune run object
        neptune_save_path: str; where to log the curve in the run
        validate: whether a validation set will be used
        train_colour: colour of the training curve
        val_colour: colour of the validation curve
        accumulate: whether to expect gradient accumulation
        dir_name: str; name of the directory where the curves will be saved at if log_to_device is True
        exp_name: str; name of the experiment
        _config_dict: config dict specifying the hyperparameters
    """
    def __init__(self, *args, **kwargs):
        super().__init__(dir_name = 'ROC_curves/', *args, **kwargs)
        self.name = 'roc_curve'
        
    def save(self):
        fp_rates, tp_rates, _ = sklearn.metrics.roc_curve(self.y, self.y_hat)
        colour = self.train_colour if self.train else self.val_colour
        
        epoch_type_prefix = '' if self.train else 'val_'
        epoch_type = 'train' if self.train else 'validation'
        figtitle = f'{epoch_type} ROC curve at epoch {self.epoch_idx}'
        fig_name = epoch_type_prefix + 'roc_curve_epoch_{}'.format(self.epoch_idx)
        plotter = plotters.GeneralPlotter(dict(Ys = [list(tp_rates), [0, 1]],
                                               x = list(fp_rates),
                                               xlabel = 'false positive rate',
                                               ylabel = 'true positive rate',
                                               title = figtitle,
                                               colors = [colour, 'lightgrey'],
                                               dashes = ['solid', 'dashed'],
                                               fname = fig_name,
                                               dirname = self.save_path),
                                          self.run[self.neptune_save_path + 'roc_curves/'])
        utils.export_plot(plotter, self.extensions)

        

class PrecisionRecallCurve(Curve): # TODO: fancy plotting
    """
    Object to calculate and save precision-recall curves at certain epochs during a run. The plots may be saved to a 'Precision-recall curves' subdirectory to the main directory and to Neptune.

    Parameters:
        neptune_run: a neptune run object
        neptune_save_path: str; where to log the curve in the run
        validate: whether a validation set will be used
        train_colour: colour of the training curve
        val_colour: colour of the validation curve
        accumulate: whether to expect gradient accumulation
        dir_name: str; name of the directory where the curves will be saved at if log_to_device is True
        exp_name: str; name of the experiment
        _config_dict: config dict specifying the hyperparameters
    """

    def __init__(self, *args, **kwargs):
        super().__init__(dir_name = 'Precision-recall_curves/', *args, **kwargs)
    
    def save(self):
        precision, recall, _ = sklearn.metrics.precision_recall_curve(self.y, self.y_hat)
        colour = self.train_colour if self.train else self.val_colour
        
        epoch_type_prefix = '' if self.train else 'val_'
        epoch_type = 'train' if self.train else 'validation'
        figtitle = f'{epoch_type} precision-recall curve at epoch {self.epoch_idx}'
        fig_name = epoch_type_prefix + 'precision_recall_curve_epoch_{}'.format(self.epoch_idx)
        plotter = plotters.GeneralPlotter(dict(Ys = [list(precision)],
                                               x = list(recall),
                                               xlabel = 'recall',
                                               ylabel = 'precision',
                                               title = figtitle,
                                               colors = [colour],
                                               fname = fig_name,
                                               dirname = self.save_path),
                                          self.run[self.neptune_save_path + 'precision_recall_curves/'])
        utils.export_plot(plotter, self.extensions)