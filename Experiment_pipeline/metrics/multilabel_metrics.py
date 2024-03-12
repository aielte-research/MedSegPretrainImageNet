import numpy as np
import pandas as pd

import utils
from . import metrics

class MultiLabelCMMetric(metrics.Metric):
    
    PARENT_METRIC = metrics.ConfusionMatrix
    
    PARAMS = dict(class_average_type = ('micro', 'macro'))
    
    keywords = ('true_positives', 'false_positives', 'true_negatives', 'false_negatives')
    
    def __init__(self, binary_metric_constr, _config_dict, threshold = 0.5,
                 name = None, class_names = [], return_classwise_kw = None,
                 micro_averaged_name = None, macro_averaged_name = None, weighted_averaged_name = None,
                 plural_name = None, *args, **kwargs):
        binary_metric = binary_metric_constr(_config_dict = _config_dict, threshold = threshold, *args, **kwargs)
        self.calculator = binary_metric.calculator
        if name:
            base_name = name
            self.name = f'{name}_threshold_{threshold}'
        else:
            self.name = binary_metric.name
            base_name = self.name[:-len(f'_threshold_{threshold}')] if f'_threshold_{threshold}' in self.name else self.name
        avg_types = _config_dict.get_str_tuple('metrics/calculation/class_average_type')
        all_types = ('micro', 'macro', 'weighted')
        if 'all' in avg_types:
            avg_types = all_types
        for avg_type in ('micro', 'macro', 'weighted'):
            setattr(self, avg_type, avg_type in avg_types)
        
        self.return_classwise = return_classwise_kw and _config_dict.get(f'metrics/calculation/{return_classwise_kw}', False)
        if self.return_classwise:
            class_names = [class_name.replace(' ', '_') for class_name in class_names]
            self.labels = [f'{base_name}_{class_name}_threshold_{threshold}' for class_name in class_names]
            
            self.to_validate = kwargs['validate']
            self.save_path = _config_dict.get_str('meta/technical/absolute_path') + kwargs['exp_name'] + '/'
            self.extensions = _config_dict.get_str_tuple('meta/technical/export_plots_as')
            self.neptune_run = kwargs['neptune_run']
            self.neptune_save_path = kwargs['neptune_save_path'].split('/')[0] + '/plots'
            self.REQUIRES_LAST_PASS = True
        
        self.micro_averaged_name = f'{micro_averaged_name}_threshold_{threshold}' if micro_averaged_name else f'micro_averaged_{self.name}'
        self.macro_averaged_name = f'{macro_averaged_name}_threshold_{threshold}' if macro_averaged_name else f'macro_averaged_{self.name}'
        self.weight_averaged_name = f'{weighted_averaged_name}_threshold_{threshold}' if weighted_averaged_name else f'weighted_averaged_{self.name}'
        
        self.plural_name = plural_name or base_name + 's'
    
    def calculate_batch(self, *args, **kwargs):
        return
    
    def evaluate_batch(self, *args, **kwargs):
        return
    
    def evaluate_epoch(self, parent_value, *args, **kwargs):
        values_dict = {}
        if self.micro:
            try:
                values_dict[self.micro_averaged_name] = self.calculator(**{k: v.sum().item() for k, v in parent_value.items()})
            except Exception as e:
                msg = f'Exception occurred while trying to calculate {self.micro_averaged_name}.'
                utils.handle_exception(e, msg)
        if self.macro or self.return_classwise:
            try:
                values_list = []
                for values in zip(*(parent_value[kw] for kw in self.keywords)):
                    values_list.append(self.calculator(**dict(zip(self.keywords, values))).item())
                if self.macro:
                    values_dict[self.macro_averaged_name] = np.mean(values_list)
                if self.weighted:
                    values_dict[self.weight_averaged_name] = np.average(values_list, weights = parent_value['class_counts'])
                if self.return_classwise:
                    values_dict.update(dict(zip(self.labels, values_list)))
            except Exception as e:
                msg = f'Exception occurred while trying to calculate classwise {self.plural_name}. No classwise, macro- or weighted averaged {self.plural_name} will be logged.'
                utils.handle_exception(e, msg)
        return values_dict
    
    
    def evaluate_at_end(self, *args, **kwargs):
        if not self.return_classwise or not self.to_validate: # TODO
            return
        prefix = 'val_metrics/' if self.to_validate else 'metrics/'
        logs = pd.read_csv(self.save_path + 'epoch_logs.csv')
        metric_logs = [logs[prefix + metric_name].to_list() for metric_name in self.labels]
        plotter = utils.framework.plotters.GeneralPlotter(dict(Ys = metric_logs,
                                                               xlabel = 'epoch', ylabel = self.name,
                                                               title = f'Validation {self.plural_name}'.replace('_', ' '),
                                                               legend = {'labels': self.labels},
                                                               dirname = self.save_path + 'plots/',
                                                               fname = f'{self.plural_name}_plot'),
                                                          self.neptune_run[self.neptune_save_path])
        utils.export_plot(plotter, self.extensions)

class F1Score(MultiLabelCMMetric):
    
    def __init__(self, *args, **kwargs):
        super().__init__(metrics.DiceIndex, name = 'f1_score',
                         return_classwise_kw = 'log_classwise_f1_scores', *args, **kwargs)

class JaccardIndex(MultiLabelCMMetric):
    
    def __init__(self, *args, **kwargs):
        super().__init__(metrics.JaccardIndex, *args, **kwargs)

class Accuracy(MultiLabelCMMetric):
    
    def __init__(self, *args, **kwargs):
        super().__init__(metrics.Accuracy, name = 'accuracy',
                         micro_averaged_name = 'accuracy',
                         macro_averaged_name = 'balanced_accuracy',
                         weighted_averaged_name = 'weighted_accuracy',
                         plural_name = 'accuracies',
                         return_classwise_kw = 'log_classwise_accuracies',
                         *args, **kwargs)