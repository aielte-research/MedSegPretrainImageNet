from typing import Dict
from .metrics import *

from exception_handling import handle_exception

from matplotlib import pyplot as plt

import pandas as pd
import utils
import os

class ElementwiseMetric(metrics.Metric):
    
    PARAMS = {'dimension_names': [f'({freq}_Hz)' for freq in [63, 125, 250, 500, 1000, 2000, 4000, 8000]]}
    
    def __init__(self, base_metric, _config_dict, *args, **kwargs):
        dim_names = _config_dict.get('metrics/calculation/dimension_names')
        self.base_metrics : Dict[str, metrics.Metric] = {}
        for dim_name in dim_names:
            if metrics.Metric not in getattr(base_metric, '__mro__', []):
                metric = metrics.Metric(base_metric, *args, **kwargs)
            else:
                metric = base_metric(_config_dict, *args, **kwargs)
            metric.name = metric.name + '_' + dim_name
            self.base_metrics[dim_name] = metric
    
    def eval_iter(self, func_name, *args, **kwargs):
        output = {}
        for metric in self.base_metrics.values():
            output.update(getattr(metric, func_name)(*args, **kwargs) or {})
        return output
    
    def calculate_batch(self, **batch):
        output = {}
        for i, metric in enumerate(self.base_metrics.values()):
            curr_batch = {
                name: getattr(value, '__getitem__', lambda *args, **kwargs: value)((slice(None), slice(i, i+1)))
                for name, value in batch.items() if name != 'predictions' # TODO: this is a hack
                }
            output.update(metric.calculate_batch(**curr_batch) or {})
        return output
    
    def evaluate_batch(self, *args, **kwargs):
        return self.eval_iter('evaluate_batch', *args, **kwargs)

    def evaluate_epoch(self, *args, **kwargs):
        return self.eval_iter('evaluate_epoch', *args, **kwargs)
    
class ElementwiseAccuracies(ElementwiseMetric):
    
    PARAMS = {**ElementwiseMetric.PARAMS, **AbsoluteOrRelativeAccuracy.PARAMS}
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(AbsoluteOrRelativeAccuracy, _config_dict, *args, **kwargs)
        self.save_path = _config_dict.get_str('meta/technical/absolute_path') + kwargs['exp_name'] + '/'
        self.neptune_run = kwargs['neptune_run']
        self.neptune_save_path = kwargs['neptune_save_path'].split('/')[0] + '/plots'
        self.to_validate = kwargs['validate']
        self.extensions = _config_dict.get_str_tuple('meta/technical/export_plots_as')
        
        if not os.path.isdir(self.save_path + 'plots'):
            os.mkdir(self.save_path + 'plots/')
    
    def evaluate_at_end(self, *args, **kwargs):
        if not self.to_validate: # TODO
            return
        prefix = 'val_metrics/' if self.to_validate else 'metrics/'
        names = [metric.name for metric in self.base_metrics.values()]
        logs = pd.read_csv(self.save_path + 'epoch_logs.csv')
        metric_logs = [logs[prefix + name].to_list() for name in names]
        plotter = utils.framework.plotters.GeneralPlotter(dict(Ys = metric_logs,
                                                               xlabel = 'epoch', ylabel = 'accuracy',
                                                               title = 'Validation accuracies',
                                                               legend = {'labels': names},
                                                               dirname = self.save_path + 'plots/',
                                                               fname = 'accuracies_plot'),
                                                          self.neptune_run[self.neptune_save_path])
        utils.export_plot(plotter, self.extensions)
        
class StrictAccuracy(AbsoluteOrRelativeAccuracy):
    
    def __init__(self, _config_dict, threshold = 0.05, *args, **kwargs):
        super().__init__(_config_dict, threshold = threshold, *args, **kwargs)
        self.name = f'strict_accuracy_(relative_threshold_{threshold})'
    
    def calculate_batch(self, prediction, label, *args, **kwargs):
        rel_bounds = (1 + self.rel_del) * label, (1 - self.rel_del) * label
        lower_bounds = torch.minimum(label - self.abs_del, torch.minimum(*rel_bounds))
        upper_bounds = torch.maximum(label + self.abs_del, torch.maximum(*rel_bounds))

        correct_preds = torch.all((lower_bounds <= prediction) & (prediction <= upper_bounds), dim = 1)
        self.curr_batch_total += torch.sum(correct_preds).item()
        self.curr_batch_size += correct_preds.numel()

class LenientStrictAccuracy(StrictAccuracy):
    
    PARAMS = {'feature_indices_to_ignore': (0, -1)}
    
    def __init__(self, _config_dict, threshold = 0.05, *args, **kwargs):
        super().__init__(_config_dict, threshold = threshold, *args, **kwargs)
        self.name = f'lenient_strict_accuracy_(relative_threshold_{threshold})'
        self.indices_to_ignore = _config_dict.get_tuple('metrics/calculation/feature_indices_to_ignore')
        self.indices_to_keep = None
    
    def calculate_batch(self, prediction, label, *args, **kwargs):
        self.indices_to_keep = self.indices_to_keep if self.indices_to_keep is not None else torch.tensor([i for i in range(prediction.shape[1]) if i not in self.indices_to_ignore])
        self.indices_to_keep = self.indices_to_keep.to(prediction.device)
        return super().calculate_batch(torch.index_select(prediction, 1, self.indices_to_keep),
                                       torch.index_select(label, 1, self.indices_to_keep),
                                       *args, **kwargs)

class WeightedAccuracy(AbsoluteOrRelativeAccuracy):
    
    PARAMS = {**AbsoluteOrRelativeAccuracy.PARAMS, 'feature_weights': (0.5, 0.75, 1., 1., 1., 1., 1., 0.5)}
    
    def __init__(self, _config_dict, threshold = 0.05, *args, **kwargs):
        super().__init__(_config_dict, threshold = threshold, *args, **kwargs)
        self.name = f'weighted_accuracy_(relative_threshold_{threshold})'
        self.weights = torch.tensor(_config_dict.get_tuple('metrics/calculation/feature_weights'))
        self.weights = (self.weights / self.weights.sum()).view(1, -1, 1)
        self.n_channels = self.weights.numel()
    
    def calculate_batch(self, prediction, label, *args, **kwargs):
        rel_bounds = (1 + self.rel_del) * label, (1 - self.rel_del) * label
        lower_bounds = torch.minimum(label - self.abs_del, torch.minimum(*rel_bounds))
        upper_bounds = torch.maximum(label + self.abs_del, torch.maximum(*rel_bounds))

        correct_preds = (lower_bounds <= prediction) & (prediction <= upper_bounds)
        self.curr_batch_total += torch.sum(correct_preds.unsqueeze(-1).flatten(2) * self.weights.to(correct_preds.device)).item()
        self.curr_batch_size += correct_preds.numel() / self.n_channels


class ConfidenceThresholdedStrictAccuracies(ConfidenceThresholdedPlot):
    
    PARAMS = ConfidenceThresholdedAccuracies.PARAMS
    
    def __init__(self, _config_dict,  *args, **kwargs):
        rel_del = _config_dict['metrics/calculation/relative_threshold']
        abs_del = _config_dict['metrics/calculation/absolute_threshold']
        def strict_accuracy(y, y_hat):
            rel_bounds = (1 + rel_del) * y, (1 - rel_del) * y
            lower_bound = np.minimum(y - abs_del, np.minimum(*rel_bounds))
            upper_bound = np.maximum(y + abs_del, np.maximum(*rel_bounds))
            return np.all((lower_bound <= y_hat) & (y_hat <= upper_bound), axis = 1)
        super().__init__(_config_dict,
                         title = 'Strict accuracy over more confident datapoints',
                         metric_calc = strict_accuracy,
                         metric_name = 'strict accuracy',
                         save_name = 'confidence_thresholded_strict_accuracies_plot',
                         *args, **kwargs)

class RecordwiseAccuracyHistogram(Plot):
    
    PARAMS = {'draw_plot_at': 'last', 'absolute_threshold': 0.01}
    
    def __init__(self, _config_dict, threshold = 0.05, *args, **kwargs):
        
        super().__init__(_config_dict, dirname = 'accuracy_histograms', *args, **kwargs)
        
        self.rel_threshold = threshold
        self.abs_threshold = _config_dict['metrics/calculation/absolute_threshold']
        self.base_fname = 'accuracy_histogram'
    
    def plot(self, *args, **kwargs):
        n_dims = self.train_y.shape[-1]
        self.train = True
        self.save(self.train_y.reshape(-1, n_dims), self.train_y_hat.reshape(-1, n_dims))
        if self.to_validate:
            self.train = False
            self.save(self.val_y.reshape(-1, n_dims), self.val_y_hat.reshape(-1, n_dims))
    
    def save(self, y, y_hat, last = False, *args, **kwargs):
        
        rel_accurate_preds = ((1 - self.rel_threshold) * y <= y_hat) & (y_hat <= (1 + self.rel_threshold) * y)
        abs_accurate_preds = np.abs(y - y_hat) <= self.abs_threshold
        n = y.shape[-1]
        accuracies = (rel_accurate_preds | abs_accurate_preds).mean(axis = -1)
        
        train_or_val = 'train' if self.train else 'validation'

        fig = plt.figure(figsize = (7, 7))
        plt.title(f'Accurately predicted target variables on {train_or_val} data')
        _, _, patches = plt.hist(accuracies, bins = np.arange(1, n + 2) / n,
                                 weights = np.ones_like(accuracies) / accuracies.size,
                                 range = (1/n, (n+1)/n),
                                 cumulative = -1, zorder = 2,
                                 label = f'accuracy relative threshold: {self.rel_threshold}')

        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.viridis_r(i / max(n - 1, 1)))

        labels = [r'$\frac{%d}{%d}$' % (i, n) if n <= 32 or i in (1, n) else None for i in range(1, n+1)]
        plt.xticks(np.arange(3, 2*n+3, 2) / (2*n), labels = labels)
        plt.grid(axis = 'y')

        plt.xlabel('minimal ratio of accurately predicted target variables')
        plt.ylabel('ratio of records')

        plt.legend()
        
        fig_name = f'{train_or_val}_{self.base_fname}_epoch_{self.epoch_idx-last}_rel_threshold_{self.rel_threshold}'
        if self.log_to_device:
            fig_path = self.save_path + fig_name + '.png'
            plt.savefig(fig_path)
        if self.log_to_neptune:
            self.neptune_run[self.neptune_save_path + '/' + fig_name + '/'].upload(fig)
        plt.close()

class ThresholdStrictAccuracyCurve(ThresholdAccuracyCurve):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = 'Threshold dependent strict accuracies at epoch {}'
        self.ylabel = 'strict accuracy'
        self.fname = 'strict_accuracy_curve_epoch_{}'
    
    def get_curve(self, abs_diffs, rel_diffs):
        return super().get_curve(abs_diffs.max(axis = -1), rel_diffs.max(axis = -1))

class ThresholdLenientStrictAccuracyCurve(ThresholdStrictAccuracyCurve):
    
    PARAMS = {**LenientStrictAccuracy.PARAMS, **ThresholdAccuracyCurve.PARAMS}
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(_config_dict, *args, **kwargs)
        self.indices = _config_dict.get_tuple('metrics/calculation/feature_indices_to_ignore')
        feats_left_out = ', '.join(map(str, self.indices[:-1])) + f' and {self.indices[-1]}'
        self.title = f'Threshold dependent strict accuracies (leaving out features at indices {feats_left_out}) at ' + 'epoch {}'
        self.fname = 'lenient_strict_accuracy_curve_epoch_{}'
    
    def get_curve(self, abs_diffs, rel_diffs):
        for idx in self.indices:
            abs_diffs[:, idx] = 0
            rel_diffs[:, idx] = 0
        return super().get_curve(abs_diffs, rel_diffs)

class ThresholdWeightedAccuracyCurve(ThresholdAccuracyCurve):
    
    PARAMS = {**ThresholdAccuracyCurve.PARAMS, 'feature_weights': (0.5, 0.75, 1., 1., 1., 1., 1., 0.5)}
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(_config_dict, *args, **kwargs)
        self.title = 'Threshold dependent weighted accuracies at epoch {}'
        self.ylabel = 'weighted accuracy'
        self.fname = 'weighted_accuracy_curve_epoch_{}'
        self.weights = np.array(_config_dict.get_tuple('metrics/calculation/feature_weights'))
        self.weights = np.reshape(self.weights / self.weights.sum(), (1, -1, 1))
    
    def get_curve(self, abs_diffs : np.ndarray, rel_diffs : np.ndarray):
        shape = abs_diffs.shape
        abs_diffs, rel_diffs = abs_diffs.reshape((*shape[:2], -1)), rel_diffs.reshape((*shape[:2], -1))
        shape = (shape[0], 1, np.product(shape[2:], dtype = int))
        weights = np.tile(self.weights, shape) / np.product(shape)
        
        abs_diffs, rel_diffs = abs_diffs.flatten(), rel_diffs.flatten()
        weights = weights.flatten()
        abs_accurates = abs_diffs <= self.abs_threshold
        n_abs_accurates = (abs_accurates * weights).sum()
        
        weights = weights[~abs_accurates]
        rel_diffs = rel_diffs[~abs_accurates]
        sort_args = np.argsort(rel_diffs, axis = None)
        sorted_diffs, sorted_weights = rel_diffs[sort_args], weights[sort_args]
        
        return lambda xs: [n_abs_accurates + np.sum(sorted_weights * (sorted_diffs <= x)) for x in xs]