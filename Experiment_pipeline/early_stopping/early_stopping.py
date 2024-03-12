class EarlyStopping(object):
    """
    Watches a metric or loss function. Its `stop` method should be called after every epoch. It will return `True` if some `stopping_criterium` is satisfied.
    Arguments:
        `watched_metric`: str; name of metric that should be watched
        `memory_length`: int; stopping criterium will be calculated based on only the last `memory_length` epochs
        `stopping_criterium`: callable (list[float] -> bool); based on the last few epoch, determines whether training should stop
        `wait_period`: int or None; number of epochs before the stopping criterium is examined; if set to None, it defaults to `memory_length`
        `neutral_value`: float; if `wait_period < memory_length`, the missing memory places will be filled with this float; otherwise, it has no effect
    """

    def __init__(self, watched_metric, memory_length, stopping_criterium,
                 wait_period = None, neutral_value = 0, message = None, *args, **kwargs):
        
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
    
    def stop(self, **metrics):
        
        self.memory = [metrics[self.watched_metric], *self.memory[:self.memory_length - 1]]
        
        if self.wait:
            self.wait = self.wait - 1
            return False
        
        to_stop = self.stopping_criterium(self.memory)

        if self.message and to_stop:
            print(self.message)

        return to_stop

class PatientImprovement(EarlyStopping):
    """
    Watches whether a metric has improved in `patience` epochs, compared to the last `patience` epochs before the current ones.
    Arguments:
        `watched metric`: str; name of metric to watch
        `patience`: int; how many epochs can pass between improvements
        `statistic`: str, one of 'average' or 'best'; what statistic to compare. If set to 'average', the average metric should improve; if set to 'best', it is enough if the best value in the last `patience` epochs beats the previous average; default: average
        `mode`: str, one of 'min' or 'max'; decides whether increase or decrease counts as improvement; default: max
        `eps`: float; minimum amount that the metric has to improve by; alternatively, if `eps < 0`, if the metric only deteriorates by `-eps`, training is not stopped; default: 0.0001
    """

    def __init__(self, watched_metric, patience, statistic = 'average', mode = 'max', eps = 1e-4):
        memory_length = 2 * patience

        def stop_crit(memory):
            old_value = sum(memory[patience:]) / patience
            if statistic == 'average':
                new_value = sum(memory[:patience]) / patience
            elif statistic == 'best':
                mix = max if mode == 'max' else min
                new_value = mix(memory[:patience])
            
            if mode == 'max':
                return new_value < old_value + eps
            elif mode == 'min':
                return new_value > old_value - eps
            
        message = 'Metric \'{}\' has stopped improving, halting training.'.format(watched_metric)
        super().__init__(watched_metric, memory_length, stop_crit, message = message)