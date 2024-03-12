import builtins
from typing import Callable, Dict, List, Literal, Optional
import utils


class EarlyStoppingWrapper(object):
    def __init__(self, early_stopping_const, config_dict, *args, **kwargs):
        early_stopping_kwargs = utils.get_kwargs(early_stopping_const, config_dict.trim())
        self.early_stopping = early_stopping_const(**early_stopping_kwargs)
    
    def stop(self, metric_values : Dict[str, float], *args, **kwargs) -> bool:
        return self.early_stopping.stop(metric_values)

class GeneralEarlyStopping(object):
    """
    Watches a metric or loss function. Its `stop` method should be called after every epoch. It will return `True` if some `stopping_criterium` is satisfied.
    Arguments:
        `watched_metric`: str; name of metric that should be watched
        `memory_length`: int; stopping criterium will be calculated based on only the last `memory_length` epochs
        `stopping_criterium`: callable (list[float] -> bool); based on the last few epoch, determines whether training should stop
        `wait_period`: int or None; number of epochs before the stopping criterium is examined; if set to None, it defaults to `memory_length`
        `neutral_value`: float; if `wait_period < memory_length`, the missing memory places will be filled with this float; otherwise, it has no effect
    """

    def __init__(self,
                 watched_metric : str,
                 memory_length : int,
                 stopping_criterium : Callable[[List[float]], bool],
                 wait_period : Optional[int] = None,
                 neutral_value : float = 0,
                 message : Optional[str] = None,
                 *args, **kwargs):
        
        watched_metric = watched_metric.replace('validation', 'val')
        watched_metric = watched_metric.replace(' ', '_')
        self.watched_metric = watched_metric
        self.memory_length = memory_length
        self.stopping_criterium = stopping_criterium
        self.message = message

        if wait_period is None:
            wait_period = memory_length

        self.wait = wait_period

        self.memory = [neutral_value for _ in range(memory_length)]
    
    def stop(self, metrics : Dict[str, float]):
        
        self.memory = [metrics[self.watched_metric], *self.memory[:self.memory_length - 1]]
        
        if self.wait:
            self.wait = self.wait - 1
            return False
        
        to_stop = self.stopping_criterium(self.memory)

        if self.message and to_stop:
            print(self.message)

        return to_stop

class RollingAverageEarlyStopping(GeneralEarlyStopping):
    """
    Watches whether a metric has improved in `patience` epochs, compared to the last `patience` epochs before the current ones.
    Arguments:
        `watched metric`: str; name of metric to watch
        `patience`: int; how many epochs can pass between improvements
        `statistic`: str, one of 'average' or 'best'; what statistic to compare. If set to 'average', the average metric should improve; if set to 'best', it is enough if the best value in the last `patience` epochs beats the previous average; default: average
        `mode`: str, one of 'min' or 'max'; decides whether increase or decrease counts as improvement; default: max
        `min_delta`: float; minimum amount that the metric has to improve by; alternatively, if `eps < 0`, if the metric only deteriorates by `-eps`, training is not stopped; default: 0.0001
    """

    PARAMS = {
        'watched_metric': 'metrics/sensitivity_threshold_0.5',
        'patience': 10,
        'statistic': 'best',
        'mode': 'max',
        'min_delta': 0.0
    }

    def __init__(self,
                 watched_metric : str,
                 patience : int,
                 statistic : Literal['average', 'best'] = 'average',
                 mode : Literal['max', 'min'] = 'max',
                 min_delta : float = 1e-4):

        patience = patience + 1
        memory_length = 2 * patience

        if statistic not in ('average', 'best'):
            raise ValueError(
                f'Argument `statistic` must be one of \'average\' or \'best\', not \'{statistic}\'.'
            )
        
        if mode not in ('min', 'max'):
            raise ValueError(
                f'Argument `mode` must be one of \'min\' or \'max\', not \'{mode}\'.'
            )

        def stop_crit(memory : List[float]):
            old_value = sum(memory[patience:]) / patience
            if statistic == 'average':
                new_value = sum(memory[:patience]) / patience
            elif statistic == 'best':
                mix = getattr(builtins, mode)
                new_value = mix(memory[:patience])
            
            if mode == 'max':
                return new_value < old_value + min_delta
            elif mode == 'min':
                return new_value > old_value - min_delta
            
        message = 'Metric \'{}\' has stopped improving, halting training.'.format(watched_metric)
        super().__init__(watched_metric, memory_length, stop_crit, message = message)