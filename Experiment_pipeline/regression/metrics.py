import importlib
import os
import warnings
import numpy as np
import scipy.stats as stats
import torch

import metrics
import utils
from utils.framework.plotters import GeneralPlotter
from . import losses

class R2Score(object):

    def __init__(self, eps = 1e-10, *args, **kwargs):
        self.name = 'r2_score'
        self.eps = eps

    def __call__(self, prediction, target, *args, **kwargs):
        squared_error = torch.sum((prediction - target) ** 2).item()
        variance = torch.sum((target - target.mean()) ** 2).item()
        return 1 - (squared_error + self.eps) / (variance + self.eps)

class RelativeL1Distance(object):

    def __init__(self, *args, **kwargs):
        self.name = 'relative_l1_distance'
    
    def __call__(self, prediction, target, *args, **kwargs):
        return torch.mean(torch.nan_to_num(torch.abs(1 - prediction/target))).item()

class AbsoluteOrRelativeError(metrics.Metric):
    
    PARAMS = {'absolute_threshold': 0.01, 'relative_threshold': 0.05}

    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(losses.AbsoluteOrRelativeLoss,
                         relative_threshold = _config_dict['metrics/calculation/relative_threshold'],
                         absolute_threshold = _config_dict['metrics/calculation/absolute_threshold'])
        self.name = 'absolute_or_relative_error'

class AbsoluteOrRelativeAccuracy(metrics.Metric):

    PARAMS = {'absolute_threshold': 0.01}

    def __init__(self, _config_dict, threshold = None, *args, **kwargs):

        self.rel_del = threshold or _config_dict['metrics/calculation/relative_threshold']
        self.abs_del = _config_dict['metrics/calculation/absolute_threshold']

        self.name = 'accuracy' if threshold is None else f'accuracy_(relative_threshold_{threshold})'

        self.total = 0
        self.num_datapoints = 0

        self.curr_batch_total = 0
        self.curr_batch_size = 0
    
    def calculate_batch(self, prediction, label, *args, **kwargs):
        rel_bounds = (1 + self.rel_del) * label, (1 - self.rel_del) * label
        lower_bounds = torch.minimum(label - self.abs_del, torch.minimum(*rel_bounds))
        upper_bounds = torch.maximum(label + self.abs_del, torch.maximum(*rel_bounds))

        correct_preds = (lower_bounds <= prediction) & (prediction <= upper_bounds)
        self.curr_batch_total += torch.sum(correct_preds).item()
        self.curr_batch_size += correct_preds.numel()
    
    def evaluate_batch(self, *args, **kwargs):
        curr_value = self.curr_batch_total
        curr_bs = self.curr_batch_size

        self.curr_batch_total = 0
        self.curr_batch_size = 0

        self.total += curr_value
        self.num_datapoints += curr_bs

        return {self.name: curr_value / curr_bs}
    
    def evaluate_epoch(self, *args, **kwargs):
        value = self.total / self.num_datapoints if self.num_datapoints > 0 else 0
        self.total = 0
        self.num_datapoints = 0

        return {self.name: value}

class AggregateInput(metrics.Metric):
    
    def __init__(self, _config_dict, validate = True, inputs = {}, *args, **kwargs):
        
        self.attr_names = [*(f'train_{kw}' for kw in inputs.keys()), *(f'val_{kw}' for kw in inputs.keys())]
        for attr in self.attr_names:
            setattr(type(self), attr, np.array([]))
        
        self.inputs = inputs    
        self.epoch_idx = 1
        self.to_validate = validate
        
        self.always = False
        self.train = True
        self.first = True
        active_kws = set()
        for metric_name in _config_dict.get_str_tuple('metrics/metrics'):
            split_metric_path = metric_name.split('.')
            metric_constr = getattr(importlib.import_module('.'.join(split_metric_path[:-1])), split_metric_path[-1])
            while hasattr(metric_constr, 'PARENT_METRIC'):
                if metric_constr.PARENT_METRIC == type(self):
                    self.always = self.always or getattr(metric_constr, 'ALWAYS_ACTIVE', True)
                    active_kws.add(getattr(metric_constr, 'ACTIVE_EPOCHS_KW', None))
                    break
                else:
                    metric_constr = metric_constr.PARENT_METRIC
        
        if not self.always:
            self.active_epochs = []
            for kw in active_kws:
                if kw is None:
                    continue
                self.active_epochs += list(_config_dict.get_tuple(f'metrics/calculation/{kw}'))
            self.do_last = 'last' in self.active_epochs
            if self.do_last:
                self.active_epochs = [n for n in set(self.active_epochs) if n != 'last']
    
    @staticmethod
    def concatenate(cum_y, y):
        y = y.cpu().detach().numpy()
        if cum_y.size == 0:
            return y
        else:
            return np.concatenate((cum_y, y), axis = 0)
    
    def calculate_batch(self, last = False, train = True, *args, **kwargs):
        if self.first and (self.always or self.epoch_idx - 1 in self.active_epochs):
            for attr in self.attr_names:
                setattr(type(self), attr, np.array([]))
            self.first = False
        if self.always or (last and self.do_last) or self.epoch_idx in self.active_epochs:
            for attr_kw, input_kw in self.inputs.items():
                train_or_val = 'train' if train else 'val'
                kw = f'{train_or_val}_{attr_kw}'
                setattr(type(self), kw, self.concatenate(getattr(self, kw), kwargs[input_kw]))
    
    def evaluate_batch(self, train = True, *args, **kwargs):
        self.train = train
    
    def evaluate_epoch(self, *args, **kwargs):
        if self.to_validate and self.train:
            return
        
        self.epoch_idx += 1
        self.first = True

class AggregatePredictionAndLabel(AggregateInput):
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(_config_dict, inputs = {'y': 'label', 'y_hat': 'prediction'}, *args, **kwargs)
        
class Plot(metrics.Metric):
    
    REQUIRES_LAST_PASS = True
    ALWAYS_ACTIVE = False
    ACTIVE_EPOCHS_KW = 'draw_plot_at'
    
    PARENT_METRIC = AggregatePredictionAndLabel
    
    PARAMS = {'draw_plot_at': 'last'}

    def __init__(self, _config_dict, neptune_run = None, neptune_save_path = '',
                 validate = True, exp_name = '', dirname = '', *args, **kwargs):
        
        self.neptune_run = neptune_run
        self.neptune_save_path = neptune_save_path.split('/')[0] + f'/plots/{dirname}/'
        self.to_validate = validate
        self.active_epochs = _config_dict.get('metrics/calculation/draw_plot_at', ['last'])

        self.log_to_device = _config_dict['meta/technical/log_to_device']
        self.log_to_neptune = _config_dict['meta/technical/log_to_neptune']
        
        self.extensions = _config_dict.get_str_tuple('meta/technical/export_plots_as')
        self.extensions = utils.get_extensions(self.extensions)
        
        # convert the list of active epochs the a list containing ints
        if isinstance(self.active_epochs, (str, int)):
            self.active_epochs = [self.active_epochs]
        self.active_epochs = list(self.active_epochs)
        self.do_last = 'last' in self.active_epochs
        if self.do_last:
            self.active_epochs = [n for n in self.active_epochs if n != 'last']
        
        if self.log_to_device:
            save_dest = _config_dict['meta/technical/absolute path']
            self.save_path = f'{save_dest}{exp_name}/{dirname}/'
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)
        
        self.epoch_idx = 1
        self.train = True
        self.ignore_nans = True
    
    def calculate_batch(self, *args, **kwargs):
        return
    
    def evaluate_batch(self, train = True, *args, **kwargs):
        self.train = train
    
    def evaluate_epoch(self, last = False, *args, **kwargs):
        if self.to_validate and self.train:
            return

        if self.epoch_idx not in self.active_epochs and not (last and self.do_last):
            self.epoch_idx += 1
            return
        
        if last and self.do_last:
            self.epoch_idx -= 1
        
        for attr_type in ('train', 'val'):
            y = getattr(self.PARENT_METRIC, f'{attr_type}_y')
            y_hat = getattr(self.PARENT_METRIC, f'{attr_type}_y_hat')
            dim_range = tuple(range(1, y.ndim))
            y_nonnans = ~np.isnan(y).any(axis = dim_range)
            y_hat_nonnans = ~np.isnan(y_hat).any(axis = dim_range)
            nonnans = y_nonnans & y_hat_nonnans
            setattr(self, f'{attr_type}_y', y[nonnans])
            setattr(self, f'{attr_type}_y_hat', y_hat[nonnans])
        
        self.plot()
        
        for attr in ('train_y', 'train_y_hat', 'val_y', 'val_y_hat'):
            setattr(self, attr, np.array([]))
                
        self.epoch_idx += 1

class ScatterPlot(Plot):

    PARAMS = {**Plot.PARAMS, 'show_heatmap': True, 'draw_boundary': True}
    BOUNDARY_PARAMS = {'absolute_threshold': 0.01, 'relative_threshold': 0.05}

    @staticmethod
    def fill_kwargs(config_dict):
        if config_dict['draw_boundary']:
            for k, v in ScatterPlot.BOUNDARY_PARAMS.items():
                config_dict.get_or_update(k, v)

    def __init__(self, _config_dict, *args, **kwargs):
        
        super().__init__(_config_dict, dirname = 'scatter_plots', *args, **kwargs)

        if _config_dict['metrics/calculation/draw_boundary']:
            rel_del = _config_dict['metrics/calculation/relative_threshold']
            abs_del = _config_dict['metrics/calculation/absolute_threshold']

            upper_boundary = f'max(x*(1+{rel_del}),x*(1-{rel_del}),x+{abs_del})'
            lower_boundary = f'min(x*(1+{rel_del}),x*(1-{rel_del}),x-{abs_del})'
            self.boundary_funcs = (upper_boundary, lower_boundary)
        else:
            self.boundary_funcs = []
        
        self.heatmap = _config_dict['metrics/calculation/show_heatmap']
        self.base_fname = 'scatter_plot'
    
    
    def plot(self, *args, **kwargs):
        
        for attr in ('train_y', 'val_y', 'train_y_hat', 'val_y_hat'):
            setattr(self, attr, getattr(self, attr).flatten())
        
        y = [self.train_y, self.val_y] if self.to_validate else [self.train_y]
        y_hat = [self.train_y_hat, self.val_y_hat] if self.to_validate else [self.train_y_hat]

        exp = None if self.neptune_run is None else self.neptune_run[self.neptune_save_path]
        labels = ['train', 'validation'] if self.to_validate else []
        plotter = utils.framework.plotters.ScatterPlotter(
                                                dict(Xs = y, Ys = y_hat,
                                                     xlabel = 'target', ylabel = 'prediction',
                                                     title = f'Predictions at epoch {self.epoch_idx}',
                                                     fname = f'{self.base_fname}_epoch_{self.epoch_idx}',
                                                     dirname = self.save_path,
                                                     legend = {'labels': labels},
                                                     opacity = 0.3,
                                                     circle_size = 4,
                                                     boundary = {'functions': self.boundary_funcs,
                                                                 'colors': ['black']},
                                                     heatmap = self.heatmap),
                                                neptune_experiment = exp)
        utils.export_plot(plotter, self.extensions)
        if self.heatmap:
            plotter.Xs = [list(self.val_y)] if self.to_validate else [list(self.train_y)]
            plotter.Ys = [list(self.val_y_hat)] if self.to_validate else [list(self.train_y_hat)]
            plotter.legend = {'location': None, 'labels': [None]}
            plotter.opacity = 0.5
            plotter.circle_size = 2
            plotter.colors = plotter.colors[:1]
            loop_type = 'validation' if self.to_validate else 'train'
            plotter.title = f'Heatmap of {loop_type} predictions at epoch {self.epoch_idx}'
            plotter.fname = f'{self.base_fname}_heatmap_epoch_{self.epoch_idx}'
            extensions = [extension for extension in self.extensions if extension != 'json']
            utils.export_plot(plotter, extensions, heatmap = True)

class TargetDistribution(Plot):
    """Creates a plot of the distribution of the target variable after the first epoch. Only works for one-dimensional target."""
    
    REQUIRES_LAST_PASS = False
    PARENT_METRIC = None

    def __init__(self, _config_dict, neptune_run = None, neptune_save_path = '',
                 validate = True, exp_name = '', *args, **kwargs):
        
        self.neptune_run = neptune_run
        self.neptune_save_path = neptune_save_path.split('/')[0] + '/plots'
        self.to_validate = validate
        
        self.log_to_device = _config_dict['meta/technical/log_to_device']
        self.log_to_neptune = _config_dict['meta/technical/log_to_neptune']
        
        self.extensions = _config_dict.get_str_tuple('meta/technical/export_plots_as')
        
        if self.log_to_device:
            save_dest = _config_dict['meta/technical/absolute path']
            self.save_path = f'{save_dest}{exp_name}/'
        
        for attr in ('train_y', 'val_y'):
            setattr(self, attr, np.array([]))
        
        self.train = True
        self.first_epoch = True
    
    def concatenate(self, cum_y, y): # TODO: this only works now for one-dimensional regression
        y_ = y if isinstance(y, np.ndarray) else y.cpu().detach().numpy().flatten()
        return np.concatenate((cum_y, y_))
    
    def calculate_batch(self, label, train = True, *args, **kwargs):
        if not self.first_epoch:
            return

        if train:
            self.train_y = self.concatenate(self.train_y, label)
        else:
            self.val_y = self.concatenate(self.val_y, label)
    
    def evaluate_batch(self, train = True, *args, **kwargs):
        if not self.first_epoch:
            return
        
        self.train = train
    
    def evaluate_epoch(self, *args, **kwargs):
        if not self.first_epoch:
            return

        if self.to_validate and self.train:
            return
        
        self.first_epoch = False
        
        all_y = np.concatenate((self.train_y, self.val_y))
        x = np.linspace(all_y.min(), all_y.max(), num = len(all_y) // 4)
        train_dens = stats.gaussian_kde(self.train_y)(x)

        if self.to_validate:
            val_dens = stats.gaussian_kde(self.val_y)(x)
            y = [list(train_dens), list(val_dens)]
        else:
            y = [list(train_dens)]

        for attr in ('train_y', 'train_y_hat', 'val_y', 'val_y_hat'):
            setattr(self, attr, np.array([]))

        exp = None if self.neptune_run is None else self.neptune_run[self.neptune_save_path]
        labels = ['train', 'val'] if self.to_validate else ['train']
        plotter = GeneralPlotter(dict(Ys = y, x = list(x), title = 'Distribution of the target variable',
                                      legend = {'labels': labels},
                                      dirname = self.save_path if self.log_to_device else '',
                                      fname = 'target_distribution'),
                                 neptune_experiment = exp)
        utils.export_plot(plotter, self.extensions)
        
        del self.train_y, self.val_y


class DeviationCurve(Plot):

    PARAMS = {'draw_plot_at': 'last', 'deviation_plot_step_number': 2500, 'deviation_plot_interval_size': 0.5}

    def __init__(self, _config_dict, *args, **kwargs):
        
        super().__init__(_config_dict, dirname = 'deviation_curves', *args, **kwargs)

        self.eps = _config_dict['metrics/calculation/deviation_plot_interval_size'] / 2
        self.steps = _config_dict['metrics/calculation/deviation_plot_step_number']
    
    def calc_dev_bias(self, y, y_hat, x_range):
        deviations=[]
        biases=[]

        permut = np.argsort(y)
        y = np.array(y)[permut]
        y_hat = np.array(y_hat)[permut]
            
        zero_centered=[i-t for t,i in zip(y,y_hat)]

        for xval in x_range:
            values=zero_centered[np.searchsorted(y, xval-self.eps):np.searchsorted(y, xval+self.eps,side='right')]
            
            std = np.std(values)
            deviations.append(np.where(np.isnan(std), 0, std))

            if len(values)==0:
                avg = 0
            else:
                avg = sum(values)/len(values)
            biases.append(avg)

        return deviations, biases

    def plot(self, *args, **kwargs):
        
        for attr in ('train_y', 'val_y', 'train_y_hat', 'val_y_hat'):
            setattr(self, attr, getattr(self, attr).flatten())
        
        y = [self.train_y, self.val_y] if self.to_validate else [self.train_y]
        y_hat = [self.train_y_hat, self.val_y_hat] if self.to_validate else [self.train_y_hat]

        exp = None if self.neptune_run is None else self.neptune_run[self.neptune_save_path]
        min_x = min([y_.min() for y_ in y])
        max_x = max([y_.max() for y_ in y])
        x_range = np.linspace(min_x, max_x, self.steps)
        deviations_lst, biases_lst = [], []
        for y_, y_hat_ in zip(y, y_hat):
            dev, bias = self.calc_dev_bias(y_, y_hat_, x_range)
            deviations_lst.append(dev)
            biases_lst.append(bias)

        labels = ['train', 'val'] if self.to_validate else ['train']
        dev_plotter = GeneralPlotter(dict(Ys = deviations_lst, x = x_range,
                                          xlabel = 'target', ylabel = 'deviations',
                                          title="Deviations (steps: {}, interval: +-{})".format(self.steps,self.eps),
                                          legend = {'labels': labels, 'location': 'top_right'},
                                          dirname = self.save_path if self.log_to_device else None,
                                          fname = f'deviation_plot_epoch_{self.epoch_idx}'),
                                     neptune_experiment = exp)
        bias_plotter = GeneralPlotter(dict(Ys = biases_lst, x = x_range,
                                           xlabel = 'target', ylabel = 'biases',
                                           title="Biases (steps: {}, interval: +-{})".format(self.steps,self.eps),
                                           legend = {'labels': labels, 'location': 'top_right'},
                                           dirname = self.save_path if self.log_to_device else None,
                                           fname = f'bias_plot_epoch_{self.epoch_idx}'),
                                      neptune_experiment = exp)
        for plotter in (dev_plotter, bias_plotter):
            utils.export_plot(plotter, self.extensions)
        

class ConfidencePlot(Plot):

    PARAMS = {'draw_plot_at': 'last'}

    def __init__(self, _config_dict, neptune_run = None, neptune_save_path = '',
                 validate = True, exp_name = '', dir_name = 'confidence_thresholded_plots',
                 *args, **kwargs):
        
        self.neptune_run = neptune_run
        self.neptune_save_path = neptune_save_path.split('/')[0] + '/plots'
        self.to_validate = validate
        self.active_epochs = _config_dict['metrics/calculation/draw_plot_at']

        self.log_to_device = _config_dict['meta/technical/log_to_device']
        self.log_to_neptune = _config_dict['meta/technical/log_to_neptune']
        
        # convert the list of active epochs the a list containing ints
        if isinstance(self.active_epochs, (str, int)):
            self.active_epochs = [self.active_epochs]
        self.active_epochs = list(self.active_epochs)
        if 'last' in self.active_epochs:
            num_epochs = _config_dict['experiment/number_of_epochs']
            self.active_epochs = list(map(lambda x: num_epochs if x == 'last' else x, self.active_epochs))
        
        if self.log_to_device:
            save_dest = _config_dict['meta/technical/absolute path']
            self.save_path = f'{save_dest}{exp_name}/{dir_name}/'
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)
        
        for attr in ('train_y', 'train_y_hat', 'val_y', 'val_y_hat', 'train_c', 'val_c'):
            setattr(self, attr, np.array([]))
        
        self.epoch_idx = 0
        self.train = True
        
        self.extensions = _config_dict.get_str_tuple('meta/technical/export_plots_as')
    
    def concatenate(self, cum_y, y): # TODO: this only works now for one-dimensional regression
        y_ = y if isinstance(y, np.ndarray) else y.cpu().detach().numpy()
        if len(cum_y) == 0:
            return y_
        return np.concatenate((cum_y, y_))
    
    def calculate_batch(self, label, predictions, *args, **kwargs):
        if self.epoch_idx + 1 in self.active_epochs:
            y_hat, c = predictions
            if self.train:
                self.train_y = self.concatenate(self.train_y, label)
                self.train_y_hat = self.concatenate(self.train_y_hat, y_hat)
                self.train_c = self.concatenate(self.train_c, c)
            else:
                self.val_y = self.concatenate(self.val_y, label)
                self.val_y_hat = self.concatenate(self.val_y_hat, y_hat)
                self.val_c = self.concatenate(self.val_c, c)
    
    def evaluate_batch(self, train = True, *args, **kwargs):
        self.train = train
    
    def evaluate_epoch(self, *args, **kwargs):
        if self.to_validate and self.train:
            return
        
        self.epoch_idx += 1
        
        if self.epoch_idx not in self.active_epochs:
            return
        
        y = [self.train_y, self.val_y] if self.to_validate else [self.train_y]
        y_hat = [self.train_y_hat, self.val_y_hat] if self.to_validate else [self.train_y_hat]
        c = [self.train_c, self.val_c] if self.to_validate else [self.train_c]
        
        labels = ['train', 'val'] if self.to_validate else []

        for attr in ('train_y', 'train_y_hat', 'val_y', 'val_y_hat', 'train_c', 'val_c'):
            setattr(self, attr, np.array([]))

        exp = None if self.neptune_run is None else self.neptune_run[self.neptune_save_path]
        fname = f'{self.name}_epoch_{self.epoch_idx}'
        
        self.plot(c, y, y_hat, labels, exp, fname)

class ConfidenceCCDF(ConfidencePlot):
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(_config_dict, dir_name = 'confidence_ccdf_plots', *args, **kwargs)
        self.name = 'confidence_ccdf'
        
    @staticmethod
    def ccdf(c):
        return lambda xs: [np.mean(c > x) for x in xs]
    
    def plot(self, cs, ys, y_hats, labels, neptune_experiment, fname, *args, **kwargs):
        x = np.linspace(0, 1, 2001, endpoint = True)
        ccdfs = [self.ccdf(c)(x) for c in cs]
        plotter = GeneralPlotter(dict(Ys = ccdfs, x = list(x),
                                      xlabel = 'confidence threshold',
                                      ylabel = 'ratio of datapoints above threshold',
                                      title = 'Tail distribution of confidence',
                                      legend = {'labels': labels},
                                      dirname = self.save_path,
                                      fname = fname),
                                 neptune_experiment = neptune_experiment)
        utils.export_plot(plotter, self.extensions)
            

class ConfidenceThresholdedPlot(ConfidencePlot):
    
    def __init__(self, _config_dict, neptune_run = None, neptune_save_path = '',
                 validate = True, exp_name = '', title = '',
                 save_name = 'confidence_thresholded_plot', metric_name = 'relative_error',
                 metric_calc =  lambda y, y_hat: np.abs((y_hat - y) / y),
                 *args, **kwargs):
        
        super().__init__(_config_dict, neptune_run, neptune_save_path, validate, exp_name,
                         dir_name = 'confidence_thresholded_plots', *args, **kwargs)
        
        self.metric_calc =  metric_calc
        self.title = title
        self.name = save_name
        self.metric_name = metric_name
    
    
    def plot(self, cs, ys, y_hats, labels, neptune_experiment, fname, *args, **kwargs):
        
        x = np.linspace(0, 1, 2001, endpoint = True)
        mean_diffs = []
        
        for c, y, y_hat in zip(cs, ys, y_hats):
            metric = np.nan_to_num(self.metric_calc(y, y_hat))
            if metric.ndim > 1:
                c = np.repeat(c, metric.shape[1], axis = 1)
            else:
                c = c.squeeze()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mean_diff = [np.nan_to_num(np.nanmean(metric[c >= th])) for th in x]
            mean_diffs.append(mean_diff)
        
        plotter = GeneralPlotter(dict(Ys = mean_diffs, x = list(x),
                                      title = f'{self.title} at epoch {self.epoch_idx}',
                                      xlabel = 'confidence threshold',
                                      ylabel = self.metric_name,
                                      dirname = self.save_path,
                                      fname = fname,
                                      legend = {'labels': labels}),
                                 neptune_experiment = neptune_experiment)
        
        utils.export_plot(plotter, self.extensions)
    
class ConfidenceThresholdedRelError(ConfidenceThresholdedPlot):
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            title = 'Average relative error on more confident datapoints',
            metric_name = 'relative error',
            metric_calc = lambda y, y_hat: np.abs((y - y_hat) / y),
            save_name = 'confidence_thresholded_rel_errors_plot',
            *args, **kwargs)

class ConfidenceThresholdedAccuracies(ConfidenceThresholdedPlot):
    
    PARAMS = {**ConfidenceThresholdedPlot.PARAMS, **AbsoluteOrRelativeAccuracy.PARAMS}
    
    def __init__(self, _config_dict, *args, **kwargs):
        rel_del = _config_dict['metrics/calculation/relative_threshold']
        abs_del = _config_dict['metrics/calculation/absolute_threshold']
        def accuracy(y, y_hat):
            rel_bounds = (1 + rel_del) * y, (1 - rel_del) * y
            lower_bound = np.minimum(y - abs_del, np.minimum(*rel_bounds))
            upper_bound = np.maximum(y + abs_del, np.maximum(*rel_bounds))
            return (lower_bound <= y_hat) & (y_hat <= upper_bound)
        super().__init__(_config_dict,
                         title = 'Accuracy over more confident datapoints',
                         metric_calc = accuracy,
                         metric_name = 'accuracy',
                         save_name = 'confidence_thresholded_accuracies_plot',
                         *args, **kwargs)

class ThresholdAccuracyCurve(Plot):
    
    PARAMS = {**Plot.PARAMS, 'absolute_threshold': 0.01}
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(_config_dict, dirname = 'accuracy_curves', *args, **kwargs)
        self.abs_threshold = _config_dict['metrics/calculation/absolute_threshold']
        self.eps = 0.01
        self.title = 'Threshold dependent accuracies at epoch {}'
        self.ylabel = 'accuracy'
        self.fname = 'accuracy_curve_epoch_{}'
    
    def get_curve(self, abs_diffs, rel_diffs):
        abs_diffs, rel_diffs = abs_diffs.flatten(), rel_diffs.flatten()
        n_records = abs_diffs.size
        abs_accurates = abs_diffs <= self.abs_threshold
        n_abs_accurates = abs_accurates.sum()
        sorted_diffs = np.sort(rel_diffs[~abs_accurates], axis = None)
        
        return lambda xs: [(n_abs_accurates + (sorted_diffs <= x).sum()) / n_records for x in xs]
    
    def plot(self, *args, **kwargs):
        train_abs_diffs = np.abs(self.train_y - self.train_y_hat)
        train_rel_diffs = np.abs(train_abs_diffs / self.train_y)
        
        val_abs_diffs = np.abs(self.val_y - self.val_y_hat)
        val_rel_diffs = np.abs(val_abs_diffs / self.val_y)
        
        max_x = max(train_rel_diffs.max(), val_rel_diffs.max())
        
        train_curve_maker = self.get_curve(train_abs_diffs, train_rel_diffs)
        val_curve_maker = self.get_curve(val_abs_diffs, val_rel_diffs)
        
        xs = np.arange(0, max_x + 1.01 * self.eps, self.eps)
        train_curve = train_curve_maker(xs)
        val_curve = val_curve_maker(xs)
        
        plotter = GeneralPlotter(dict(Ys = [train_curve, val_curve], x = list(xs),
                                      title = self.title.format(self.epoch_idx),
                                      xlabel = 'relative threshold',
                                      ylabel = self.ylabel,
                                      dirname = self.save_path,
                                      fname = self.fname.format(self.epoch_idx),
                                      legend = {'labels': ['train', 'validation']}),
                                 neptune_experiment = self.neptune_run[self.neptune_save_path])
        
        utils.export_plot(plotter, self.extensions)
