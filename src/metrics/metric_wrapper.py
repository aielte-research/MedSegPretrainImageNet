import inspect
import re
import sys
import types
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import torch

import utils
from exception_handling import handle_exception
from utils.config_dict import ConfigDict


class Metric(object):
    """Wrapper object for metrics."""

    PARAMS = {'label_type': 'mask'}
    
    @staticmethod
    def convert_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def __init__(self, metric_constr_or_func : Callable,
                 threshold : Union[float, None] = None,
                 accumulate : bool = True, *args, **kwargs):
        """
        Arguments:
            `metric_constr_or_func`: constructor of a metric object, or a function that calculates the metric
            `threshold`: optional float; if given, will be passed onto the metric constructor
            `accumulate`: bool; whether to use gradient accumulation; default: True
            `args`, `kwargs`: any number of positional and keyword arguments that will be passed onto the metric constructor
        """

        if isinstance(metric_constr_or_func, types.FunctionType):
            if threshold is not None:
                def calculator(y, y_hat):
                    y = y.cpu().numpy().astype(int)
                    y_hat = (y_hat.cpu().numpy() >= threshold).astype(int)

                    return metric_constr_or_func(y, y_hat)
            else:
                def calculator(y, y_hat):
                    y = y.cpu().numpy()
                    y_hat = y_hat.cpu().numpy()
                    return metric_constr_or_func(y, y_hat)
            
            self.calculator = calculator
        
        else:
            if threshold is not None:
                self.calculator = metric_constr_or_func(*args, **kwargs, threshold = threshold)
            else:
                self.calculator = metric_constr_or_func(*args, **kwargs)
        
        self.name = getattr(self.calculator, 'name', self.convert_to_snake(metric_constr_or_func.__name__))

        self.value = 0
        self.num_batches = 0

        self.accumulate = accumulate

        if accumulate:
            self.num_batch_fragments = 0
            self.acc_value = 0
    
    @torch.no_grad()
    def calculate_batch(self, cumulate = True, **batch):
        """
        Calculates loss value at a batch fragment.

        Parameters:
            `batch`: the dictionary containing the batched data
            `cumulate`: whether to add the current loss to a moving average (over either batch fragments or epoch)
            `kwargs`: any number of keyword argument that will be passed onto the metric calculator
        """
        value = self.calculator(batch['prediction'], batch[self.PARAMS.get('label_type', Metric.PARAMS['label_type'])])
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        if getattr(value, 'size', 2) == 1:
            value = value.item()
        if self.accumulate:
            self.num_batch_fragments += 1
            self.acc_value += value
        else:
            if cumulate:
                self.value += value
                self.num_batches += 1

            return {self.name: value}
    
    def evaluate_batch(self, cumulate = True, flush = True, average = True, *args, **kwargs):
        """Function to be called at the last batch fragment of the batch."""
        if self.accumulate:
            if self.num_batch_fragments == 0:
                return {self.name: 0}
            value = self.acc_value
            if average:
                value = value / self.num_batch_fragments
            if flush:
                self.acc_value = 0
                self.num_batch_fragments = 0
            if cumulate:
                self.value += value
        else:
            value = self.value
        if cumulate:
            self.num_batches += 1
        return {self.name: value}
    
    def evaluate_epoch(self, flush = True, average = True, *args, **kwargs):
        """Function to be called at the and of the epoch."""
        if self.num_batches == 0:
            return {self.name: 0}
        value = self.value
        if average:
            value = value / self.num_batches
        if flush:
            self.value, self.num_batches = 0, 0
        return {self.name: value}

class MetricsCalculator(object):
    """Object for calculating multiple metrics."""

    METRIC_CALC_PATH = 'metrics/calculation'
    METRICS_PATH = 'metrics/metrics'

    PARAMS = {
        'thresholds': 0.5
        }
    
    PATTERN = '(.*)_threshold_.*' # pattern for getting the base name of the metric

    @staticmethod
    def fill_metric_kwargs(metric_constr : Callable, config_dict : ConfigDict):
        """Fill a ConfigDict with the default hyperparameters of a metric."""

        # fill dict with the default parameters of the metric.
        config_dict.fill_with_defaults(getattr(metric_constr, 'PARAMS', {}))

        # if the metric has a parent metric, fill the dict with its hyperparameters as well.
        if hasattr(metric_constr, 'PARENT_METRIC'):
            MetricsCalculator.fill_metric_kwargs(metric_constr.PARENT_METRIC, config_dict)
        
        # add thresholds to the config dict if absent
        if 'thresholds' not in config_dict and 'threshold' in inspect.signature(metric_constr).parameters:
            config_dict['thresholds'] = MetricsCalculator.PARAMS['thresholds']
        
        # fill the config dict with additional kwargs if it is specified in the metric class
        if hasattr(metric_constr, 'fill_kwargs'):
            metric_constr.fill_kwargs(config_dict)

    @staticmethod
    def fill_kwargs(config_dict : ConfigDict):
        """Fill a ConfigDict with the default hyperparameters of all metrics."""

        metric_calcs_dict = config_dict.get_or_update(MetricsCalculator.METRIC_CALC_PATH,
                                                      {'calculation': {'default': {}}})
        for metric_name in config_dict[MetricsCalculator.METRICS_PATH]:
            metric_constr = utils.get_class_constr(metric_name)
            MetricsCalculator.fill_metric_kwargs(metric_constr, metric_calcs_dict)

    @staticmethod
    def requires_threshold(metric_constr) -> bool:
        """Decide whether a metric requires a threshold."""
        has_threshold_arg = 'threshold' in inspect.signature(metric_constr).parameters
        has_parent = hasattr(metric_constr, 'PARENT_METRIC') and metric_constr.PARENT_METRIC is not None
        parent_has_threshold = has_parent and MetricsCalculator.requires_threshold(metric_constr.PARENT_METRIC)
        return has_threshold_arg or parent_has_threshold
    
    @staticmethod
    def create_metric(metric_constr : Callable, config_dict : ConfigDict,
                      threshold : Union[float, None] = None, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Creates a metric from a constructor and a ConfigDict."""

        metric_kwargs = {}
        for arg_name, arg_dict_or_default in getattr(metric_constr, 'PARAMS', {}).items():
            key = arg_dict_or_default.get('argument name', arg_name) if isinstance(arg_dict_or_default, dict) else arg_name
            value = config_dict[key]
            metric_kwargs[key] = value
        if MetricsCalculator.requires_threshold(metric_constr):
            metric_kwargs['threshold'] = threshold

        # create metric instance
        if Metric not in getattr(metric_constr, '__mro__', []):
            # if the constructor does not belong to a subclass of Metric, it will be wrapped in a Metric object
            metric = Metric(metric_constr, **metric_kwargs)
        else:
            metric = metric_constr(**metric_kwargs, **kwargs)

        # get the name of the metric
        name = getattr(
                    metric, 'name',
                    Metric.convert_to_snake(metric_constr.__name__)
                    )
        if 'threshold' not in name and MetricsCalculator.requires_threshold(metric_constr):
            name = '_'.join((name, 'threshold', str(threshold)))
    
        curr_metric_dict = {'calculator': metric}
        metric_dict = {}
        
        # create parent metric
        if hasattr(metric, 'PARENT_METRIC') and metric.PARENT_METRIC is not None:
            parent_metric_dict = MetricsCalculator.create_metric(metric.PARENT_METRIC, config_dict,
                                                                 threshold = threshold, **kwargs)
            curr_metric_dict['parent'] = list(parent_metric_dict.keys())[0]
            metric_dict.update(parent_metric_dict)
        metric_dict[name] = curr_metric_dict
        return metric_dict
    
    def __init__(self, config_dict : ConfigDict, loss = None, *args, **kwargs):
        """Creates an instance from a config dict specifying the metrics and the hyperparameters for their calculations."""

        metric_calcs_dict : ConfigDict = config_dict[self.METRIC_CALC_PATH]
        metrics_list : Tuple[str] = config_dict.get_tuple(self.METRICS_PATH)

        thresholds : Iterable[float] = metric_calcs_dict.get('thresholds', [])
        if isinstance(thresholds, (float, int)):
            thresholds = [thresholds]

        self.metrics : Dict[str, Dict[str, Metric]] = {}
        for metric_name in metrics_list:
            metric_constr = utils.get_class_constr(metric_name)
            if self.requires_threshold(metric_constr):
                for threshold in thresholds:
                    self.metrics.update(self.create_metric(metric_constr, metric_calcs_dict, threshold,
                                                           _config_dict = config_dict, *args, **kwargs))
            else:
                self.metrics.update(self.create_metric(metric_constr, metric_calcs_dict,
                                                       _config_dict = config_dict, *args, **kwargs))
        
        if loss is None:
            loss = lambda *args, **kwargs: {}
        self.loss = loss
        self.loss_name = getattr(loss, 'name', 'loss')
        
        self.requires_last_pass = any(getattr(metric_dict['calculator'], 'REQUIRES_LAST_PASS', False) for metric_dict in self.metrics.values())
        
        self.num_epochs = 0
        self.num_batches = 0
        self.num_batch_fragments = 0
        self.train = True
        self.acc_scale = 1

        self.to_validate = kwargs.get('validate', True)

    def calc_or_eval(self, batch, func_to_call = 'calculate_batch', msg = lambda _: '', *args, **kwargs):
        """Calls the `func_to_call` method of all metrics and aggregates the results."""
        value_dicts, values = {}, {}
        
        def calculate(metric_name):
            if metric_name in value_dicts:
                return value_dicts[metric_name]
            parent = self.metrics[metric_name].get('parent')
            if parent:
                parent_value = calculate(parent)
                if 'threshold' in metric_name and parent_value is not None:
                    parent_value = {re.match(self.PATTERN, k).group(1): v for k, v in parent_value.items()}
                value =  getattr(self.metrics[metric_name]['calculator'], func_to_call)(
                                                                            parent_value = parent_value,
                                                                            *args, **kwargs, **batch
                                                                            )
            else:
                value = getattr(self.metrics[metric_name]['calculator'], func_to_call)(*args, **kwargs, **batch)
            value_dicts[metric_name] = value
            return value
    
        for metric_name in self.metrics.keys():
            try:
                calculate(metric_name)
            except Exception as e:
                handle_exception(e, msg(metric_name))
        
        for metric_name, value in value_dicts.items():
            if value is not None:
                try:
                    values.update(value)
                except TypeError:
                    print(f'Output of {func_to_call} must be dict or None, but {metric_name} returned {type(value)} ({value}). Value was not logged.', file = sys.stderr)
        
        values = {'metrics/' + k: v for k, v in values.items() if isinstance(v, (int, float))}
        try:
            values.update(getattr(self.loss, func_to_call, self.loss)(batch, *args, **kwargs))
        except Exception as e:
            handle_exception(e, msg(self.loss_name))
        
        return values
    
    def batch_error_msg(self, metric_name):
        loop_name = 'train' if self.train else 'validation'
        if self.acc_scale == 1:
            return f'An error occured trying to calculate {metric_name} in batch {self.num_batches} in the {loop_name} loop of epoch {self.num_epochs}.'
        else:
            return f'An error occured trying to calculate {metric_name} in batch fragment {self.num_batch_fragments} of batch {self.num_batches} in the {loop_name} loop of epoch {self.num_epochs}.'
        
    def calculate_batch(self, batch : Dict[str, Any], *args, **kwargs) -> Dict[str, Union[int, float]]:
        """Calls the `calculate_batch` method of all metrics then aggregates the results."""
        self.train = kwargs.get('train', True)
        self.acc_scale = kwargs.get('accumulation_scale', 1)
        self.num_batch_fragments += 1
        return self.calc_or_eval(batch, 'calculate_batch', msg = self.batch_error_msg, *args, **kwargs)
    
    def batch_evaluation_error(self, metric_name):
        loop_name = 'train' if self.train else 'validation'
        return f'An error occured trying to evaluate {metric_name} in batch {self.num_batches} in the {loop_name} loop of epoch {self.num_epochs}.'
    
    def evaluate_batch(self, batch : Dict[str, Any], *args, **kwargs) -> Dict[str, Union[int, float]]:
        """Calls the `evaluate_batch` method of all metrics then aggregates the results."""
        self.num_batch_fragments = 0
        self.num_batches += 1
        return self.calc_or_eval(batch, 'evaluate_batch', msg = self.batch_evaluation_error, *args, **kwargs)
    
    def epoch_error_msg(self, metric_name):
        loop_name = 'train' if self.train else 'validation'
        return f'An error occured trying to evaluate {metric_name} at the end of the {loop_name} loop of epoch {self.num_epochs}.'
    
    def evaluate_epoch(self, *args, **kwargs) -> Dict[str, Union[int, float]]:
        """Calls the `evaluate_epoch` method of all metrics then aggregates the results."""
        self.num_batches = 0
        if not self.train or not self.to_validate:
            self.num_epochs += 1
        return self.calc_or_eval({}, 'evaluate_epoch', msg = self.epoch_error_msg, *args, **kwargs)
    
    def evaluate_at_end(self, *args, **kwargs):
        for metric_name, metric_dict in self.metrics.items():
            try:
                calc = metric_dict['calculator']
                if hasattr(calc, 'evaluate_at_end'):
                    calc.evaluate_at_end(*args, **kwargs)
            except Exception as e:
                msg = f'An exception occured while trying to evaluate {metric_name} at the end of training.'
                handle_exception(e, msg)
        
    
