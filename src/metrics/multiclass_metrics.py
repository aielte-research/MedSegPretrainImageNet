import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
import sklearn.metrics

import utils
from . import metrics

class MultiClassConfusionMatrix(metrics.Metric):

    PARAMS = {
        'number_of_classes': 1000,
        'log_confusion_matrix': False
    }

    LOG_PARAMS = {
        'log_confusion_matrix_at': 'last'
    }

    MAX_CLASSES = 10

    @staticmethod
    def fill_kwargs(config_dict):
        if config_dict['log_confusion_matrix']:
            config_dict.fill_with_defaults(MultiClassConfusionMatrix.LOG_PARAMS)
    
    def __init__(self, accumulate = True, neptune_run = None, neptune_save_path = '',
                 validate = True, exp_name = '', _config_dict = None, class_names = [],
                 *args, **kwargs):
        
        number_of_classes = _config_dict.get('metrics/calculation/number_of_classes')
        def init_cm():
            return np.zeros((number_of_classes,) * 2)
        
        self.cm = init_cm()
        self.range = list(range(number_of_classes))

        self.accumulate = accumulate
        if accumulate:
            self.acc_cm = init_cm()

        self.init_cm = init_cm

        self.log_confusion_matrix = _config_dict.get('metrics/calculation/log_confusion_matrix', False)
        self.train = True
        if self.log_confusion_matrix:
            self.num_epochs = 0
            metric_params = _config_dict['metrics/calculation']

            self.labels = class_names
            self.labels = [str(label).replace('_', ' ') for label in self.labels]
            if len(self.labels) < len(self.range):
                self.labels = ['background', *self.labels]
            if len(self.labels) > self.MAX_CLASSES:
                self.labels = [None for _ in self.labels]

            self.run = neptune_run
            self.to_validate = validate
            self.log_to_device = _config_dict['meta/technical/log to device']
            self.log_to_neptune = _config_dict['meta/technical/log to neptune']

            self.active_epochs = metric_params['log_confusion_matrix_at']
            # convert the list of active epochs the a list containing ints
            if isinstance(self.active_epochs, (str, int)):
                self.active_epochs = [self.active_epochs]
            self.active_epochs = list(self.active_epochs)
            self.do_last = 'last' in self.active_epochs
            self.REQUIRES_LAST_PASS = self.do_last
            
            if self.run:
                self.neptune_save_path = neptune_save_path
        
            if self.log_to_device:
                save_dest = _config_dict['meta/technical/absolute path']
                self.save_path = f'{save_dest}{exp_name}/Confusion_matrices/'
                if not os.path.isdir(self.save_path):
                    os.mkdir(self.save_path)

        self.class_counts = [0] * number_of_classes
    
    def update_class_counts(self, ground_truth : np.ndarray):
        for i, count in enumerate(self.class_counts):
            self.class_counts[i] = count + np.sum(ground_truth == i)
    
    def flush_class_counts(self):
        self.class_counts = [0 for _ in self.class_counts]

    def calculate_batch(self, prediction, mask = None, label = None, cumulate = True, *args, **kwargs):
        
        y = mask if mask is not None else label
        if y.shape == prediction.shape:
            y = y.argmax(dim = 1)
        
        y = y.cpu().flatten().detach().numpy()
        self.update_class_counts(y)
        y_hat = prediction.argmax(dim = 1).cpu().flatten().detach().numpy()
        cm = sklearn.metrics.confusion_matrix(y, y_hat, labels = self.range)

        if cumulate:
            self.cm += cm

        if self.accumulate:
            self.acc_cm += cm

        return {'confusion_matrix': cm}
    
    def evaluate_batch(self, flush = True, train = True, *args, **kwargs):
        self.train = train
        cm = self.acc_cm
        if flush:
            self.acc_cm = self.init_cm()

        return {'confusion_matrix': cm}

    def evaluate_epoch(self, flush = True, last = False, *args, **kwargs):
        cm = self.cm
        class_counts = self.class_counts
        if flush:
            self.cm = self.init_cm()
            self.flush_class_counts()
        
        if self.log_confusion_matrix:
            self.save(cm, last = last)

        return {'confusion_matrix': cm, 'class_counts': class_counts}
    
    def save(self, cm, last = False):
        if self.train and not last:
            self.num_epochs += 1
        
        if self.num_epochs not in self.active_epochs and not (last and self.do_last):
            return
        
        epoch_type = 'Train' if self.train else 'Validation'
        epoch_prefix = 'train_' if self.train else 'val_'
        normed_matrix = (cm.T / cm.sum(axis = 1)).T
        disp = sklearn.metrics.ConfusionMatrixDisplay(normed_matrix, display_labels = self.labels)
        disp = disp.plot(cmap = 'Blues', include_values = len(self.labels) <= self.MAX_CLASSES, values_format = '.2f')
        disp.im_.set_clim(0, 1)
        plt.xticks(rotation = 45, ha = 'right')
        if len(self.labels) > self.MAX_CLASSES:
            # remove ticks if there are too many classes
            plt.xticks(ticks = [])
            plt.yticks(ticks = [])
        plt.title(f'{epoch_type} confusion matrix at epoch {self.num_epochs}')
        if self.log_to_device:
            fig_name = epoch_prefix + f'confusion_matrix_epoch_{self.num_epochs}.png'
            plt.savefig(self.save_path + fig_name, bbox_inches = 'tight')
        plt.close()
        if self.log_to_neptune and self.log_to_device: # TODO: this should work even when not logging to device
            self.run[self.neptune_save_path + 'confusion_matrices/' + fig_name[:-4] + '/'].upload(self.save_path + fig_name)


class AverageBinaryCMMetric(metrics.Metric):

    PARENT_METRIC = MultiClassConfusionMatrix

    PARAMS = {'include_background_in_averages': False}

    def __init__(self, name, binary_metric, key = None, _config_dict = None,
                 return_classwise_kw = None, base_name = '', base_name_plural = None,
                 class_names = [], *args, **kwargs):
        ignore_background = not _config_dict['metrics/calculation/include_background_in_averages']
        self.start = int(ignore_background)
        self.num_classes = _config_dict['metrics/calculation/number_of_classes']
        self.binary_calcs = [binary_metric(*args, **kwargs) for _ in range(self.start, self.num_classes)]
        self.idcs = np.arange(self.num_classes).repeat(self.num_classes).reshape((self.num_classes,) * 2)
        self.name = name
        self.key = key or self.binary_calcs[0].name
        self.neutral = getattr(self.binary_calcs[0], 'neutral', 0)
        
        self.return_classwise = return_classwise_kw and _config_dict.get(f'metrics/calculation/{return_classwise_kw}', False)
        if self.return_classwise:
            self.labels = class_names
            if len(self.labels) > self.num_classes - self.start:
                self.labels = self.labels[1:]
            self.metric_names = ['_'.join((base_name, label)).replace(' ', '_') for label in self.labels]
            self.base_name = base_name
        
            self.plural_name = base_name_plural or base_name + 's'
            
            self.to_validate = kwargs['validate']
            self.save_path = _config_dict.get_str('meta/technical/absolute_path') + kwargs['exp_name'] + '/'
            self.extensions = _config_dict.get_str_tuple('meta/technical/export_plots_as')
            self.neptune_run = kwargs['neptune_run']
            self.neptune_save_path = kwargs['neptune_save_path'].split('/')[0] + '/plots'
            self.REQUIRES_LAST_PASS = True
    
    def get_binary_matrix(self, multiclass_cm, idx):
        tp = np.array([multiclass_cm[idx][idx]])

        real_not_idx = self.idcs != idx
        pred_not_idx = self.idcs.T != idx
        tn = multiclass_cm[real_not_idx & pred_not_idx].sum(keepdims = True)
        fn = multiclass_cm[~real_not_idx & pred_not_idx].sum(keepdims = True)
        fp = multiclass_cm[real_not_idx & ~pred_not_idx].sum(keepdims = True)

        return {'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn}

    def calculate_batch(self, parent_value, *args, **kwargs):
        values = []
        cm = parent_value['confusion_matrix']
        for i, binary_metric in enumerate(self.binary_calcs):
            idx = i + self.start
            if cm[idx, :].sum() + cm[:, idx].sum() > 0:
                k_value = binary_metric.calculate_batch(self.get_binary_matrix(cm, idx), *args, **kwargs)
                if k_value is not None:
                    values.append(k_value.get(self.key, self.neutral))
        values_dict = {self.name: self.neutral if values == [] else np.mean(values)}
        if self.return_classwise:
            values_dict.update(dict(zip(self.metric_names, values)))
        return values_dict
    
    def evaluate_batch(self, parent_value, *args, **kwargs):
        values = []
        cm = parent_value['confusion_matrix']
        for i, binary_metric in enumerate(self.binary_calcs):
            idx = i + self.start
            if cm[idx, :].sum() + cm[:, idx].sum() > 0:
                values.append(
                    binary_metric.evaluate_batch(self.get_binary_matrix(cm, idx), *args, **kwargs)[self.key]
                )
        values_dict = {self.name: self.neutral if values == [] else np.mean(values)}
        if self.return_classwise:
            values_dict.update(dict(zip(self.metric_names, values)))
        return values_dict
    
    def evaluate_epoch(self, parent_value, *args, **kwargs):
        values = []
        cm = parent_value['confusion_matrix']
        for i, binary_metric in enumerate(self.binary_calcs):
            idx = i + self.start
            if cm[idx, :].sum() + cm[:, idx].sum() > 0:
                values.append(
                    binary_metric.evaluate_epoch(self.get_binary_matrix(cm, idx), *args, **kwargs)[self.key]
                )
        values_dict = {self.name: self.neutral if values == [] else np.mean(values)}
        if self.return_classwise:
            values_dict.update(dict(zip(self.metric_names, values)))
        return values_dict

    def evaluate_at_end(self, *args, **kwargs):
        if not self.return_classwise or not self.to_validate: # TODO
            return
        prefix = 'val_metrics/' if self.to_validate else 'metrics/'
        logs = pd.read_csv(self.save_path + 'epoch_logs.csv')
        metric_logs = [logs[prefix + metric_name].to_list() for metric_name in self.metric_names]
        plotter = utils.framework.plotters.GeneralPlotter(dict(Ys = metric_logs,
                                                               xlabel = 'epoch', ylabel = self.base_name,
                                                               title = f'Validation {self.plural_name}'.replace('_', ' '),
                                                               legend = {'labels': self.labels},
                                                               dirname = self.save_path + 'plots/',
                                                               fname = f'{self.plural_name}_plot'),
                                                          self.neptune_run[self.neptune_save_path])
        utils.export_plot(plotter, self.extensions)

class DiceIndex(AverageBinaryCMMetric):
    
    RETURN_CLASSWISE_KW = 'log_classwise_dice_idcs'
    RETURN_CLASSWISE_DEFAULT = True
    
    @staticmethod
    def fill_kwargs(config_dict):
        config_dict.get_or_update(DiceIndex.RETURN_CLASSWISE_KW, DiceIndex.RETURN_CLASSWISE_DEFAULT)
    
    def __init__(self, *args, **kwargs):
        super().__init__(name = 'mean_dice_index', binary_metric = metrics.DiceIndex,
                         base_name = 'dice_index', base_name_plural = 'dice_indices',
                         return_classwise_kw = self.RETURN_CLASSWISE_KW,
                         *args, **kwargs)

class JaccardIndex(AverageBinaryCMMetric):
    
    RETURN_CLASSWISE_KW = 'log_classwise_jaccard_idcs'
    RETURN_CLASSWISE_DEFAULT = False
    
    @staticmethod
    def fill_kwargs(config_dict):
        config_dict.get_or_update(JaccardIndex.RETURN_CLASSWISE_KW, JaccardIndex.RETURN_CLASSWISE_DEFAULT)
    
    def __init__(self, *args, **kwargs):
        super().__init__(name = 'mean_jaccard_index', binary_metric = metrics.JaccardIndex,
                         base_name = 'jaccard_index', base_name_plural = 'jaccard_indices',
                         return_classwise_kw = self.RETURN_CLASSWISE_KW,
                         *args, **kwargs)

class Accuracy(metrics.Metric):
    
    PARENT_METRIC = MultiClassConfusionMatrix

    def __init__(self, accumulate = True, *args, **kwargs):
        self.name = 'accuracy'
        self.accumulate = accumulate
        self.num_batches = 0
        self.value = 0
    
    def calculate_batch(self, *args, **kwargs): # TODO
        return
    
    def evaluate_batch(self, parent_value, *args, **kwargs):
        cm = parent_value['confusion_matrix']
        value = np.diagonal(cm).sum() / np.sum(cm)
        self.value += value
        self.num_batches += 1
        return {self.name: value}
    
    def evaluate_epoch(self, flush = True, *args, **kwargs):
        value = self.value / self.num_batches
        if flush:
            self.value, self.num_batches = 0, 0
        return {self.name: value}

class ClasswiseBinaryCMMetric(metrics.Metric):

    PARENT_METRIC = MultiClassConfusionMatrix

    PARAMS = {'include_background_in_averages': False}

    def __init__(self, metric_constr, metric_name, metric_name_plural = None,
                 _config_dict = None, class_names = None, *args, **kwargs):
        num_classes = _config_dict.get('metrics/calculation/number_of_classes')
        
        self.class_names = class_names or _config_dict.get(
                                            'metrics/calculation/class_names',
                                            getattr(utils.get_class_constr(_config_dict['data/data']), 'CLASSES',
                                            [f'class {i}' for i in self.range])
                                            )
        
        if len(self.class_names) < num_classes:
            self.class_names = ('background', *self.class_names)
        self.class_names = [name.replace(' ', '_') for name in self.class_names]
        ignore_background = not _config_dict['metrics/calculation/include_background_in_averages']
        self.start = int(ignore_background)
        if ignore_background:
            self.class_names = self.class_names[1:]
        self.calcs = [metric_constr(*args, **kwargs) for _ in range(self.start, num_classes)]
        self.key = self.calcs[0].name
        self.idcs = np.arange(num_classes).repeat(num_classes).reshape((num_classes,) * 2)
        self.base_name = metric_name
        self.plural_name = metric_name_plural or metric_name + 's'
        
        self.to_validate = kwargs['validate']
        self.save_path = _config_dict.get_str('meta/technical/absolute_path') + kwargs['exp_name'] + '/'
        self.extensions = _config_dict.get_str_tuple('meta/technical/export_plots_as')
        self.neptune_run = kwargs['neptune_run']
        self.neptune_save_path = kwargs['neptune_save_path'].split('/')[0] + '/plots'
    
    def get_binary_matrix(self, parent_value, idx):
        multiclass_cm = parent_value['confusion_matrix']
        tp = multiclass_cm[idx][idx]

        real_not_idx = self.idcs != idx
        pred_not_idx = self.idcs.T != idx
        tn = multiclass_cm[real_not_idx & pred_not_idx].sum()
        fn = multiclass_cm[~real_not_idx & pred_not_idx].sum()
        fp = multiclass_cm[real_not_idx & ~pred_not_idx].sum()

        return {'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn}
    

    def calculate_batch(self, parent_value, *args, **kwargs):
        for i, acc in enumerate(self.calcs):
            acc.calculate_batch(self.get_binary_matrix(parent_value, i + self.start), *args, **kwargs)
    
    def evaluate_batch(self, parent_value, *args, **kwargs):
        return {self.base_name + '_' + self.class_names[i]: acc.evaluate_batch(self.get_binary_matrix(parent_value, i + self.start), *args, **kwargs)[self.key] for i, acc in enumerate(self.calcs)}
    
    def evaluate_epoch(self, *args, **kwargs):
        return {self.base_name + '_' + self.class_names[i]: acc.evaluate_epoch(*args, **kwargs)[self.key] for i, acc in enumerate(self.calcs)}
    
    def evaluate_at_end(self, *args, **kwargs):
        if not self.to_validate: # TODO
            return
        prefix = 'val_metrics/' if self.to_validate else 'metrics/'
        names = [f'{self.base_name}_{class_name}' for class_name in self.class_names]
        logs = pd.read_csv(self.save_path + 'epoch_logs.csv')
        metric_logs = [logs[prefix + name].to_list() for name in names]
        plotter = utils.framework.plotters.GeneralPlotter(dict(Ys = metric_logs,
                                                               xlabel = 'epoch', ylabel = self.base_name,
                                                               title = f'Validation {self.plural_name}'.replace('_', ' '),
                                                               legend = {'labels': names},
                                                               dirname = self.save_path + 'plots/',
                                                               fname = f'{self.plural_name}_plot'),
                                                          self.neptune_run[self.neptune_save_path])
        utils.export_plot(plotter, self.extensions)

class ClasswiseAccuracies(ClasswiseBinaryCMMetric):
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(metrics.Accuracy, 'accuracy', 'accuracies', _config_dict, *args, **kwargs)

class ClasswiseBalancedAccuracies(ClasswiseBinaryCMMetric):
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(metrics.BalancedAccuracy, 'balanced_accuracy', 'balanced_accuracies', _config_dict, *args, **kwargs)

class ClasswiseDiceIndices(ClasswiseBinaryCMMetric):
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(metrics.DiceIndex, 'dice_index', 'dice_indicies', _config_dict, *args, **kwargs)

class Top5Accuracy(metrics.Metric):
    
    def __init__(self, accumulate = True, *args, **kwargs):
        self.name = 'top_5_accuracy'
        self.n = 5
        
        self.accumulate = accumulate
        self.num_records = 0
        self.num_correct_preds = 0
        
        if self.accumulate:
            self.num_records_in_batch = 0
            self.num_correct_preds_in_batch = 0
    
    def calculate_batch(self, prediction, mask = None, label = None, cumulate = True, *args, **kwargs):    
        y = mask if mask is not None else label
        if y.shape == prediction.shape:
            y = y.argmax(dim = 1, keepdim = True)
        else:
            dimference = prediction.dim() - y.dim()
            y = y.view((y.shape[0],) + (1,) * dimference + y.shape[1:])
        
        tiled_y = torch.tile(y, (1, self.n) + (1,) * len(y.shape[2:]))
        top_n_preds = prediction.topk(self.n, dim = 1)[1]
        correct_preds = torch.any(top_n_preds == tiled_y, dim = 1)
        
        num_preds, num_correct_preds = correct_preds.numel(), correct_preds.sum().item()
        
        if cumulate:
            self.num_correct_preds += num_correct_preds
            self.num_records += num_preds
        
        if self.accumulate:
            self.num_correct_preds_in_batch += num_correct_preds
            self.num_records_in_batch += num_preds
        
        return {self.name: num_correct_preds / num_preds}

    def evaluate_batch(self, flush = True, *args, **kwargs):
        num_preds, num_correct_preds = self.num_records_in_batch, self.num_correct_preds_in_batch
        if flush:
            self.num_correct_preds_in_batch, self.num_records_in_batch = 0, 0
        return {self.name: num_correct_preds / num_preds}
    
    def evaluate_epoch(self, flush = True, *args, **kwargs):
        num_preds, num_correct_preds = self.num_records, self.num_correct_preds
        if flush:
            self.num_records, self.num_correct_preds = 0, 0
        return {self.name: num_correct_preds / num_preds}
    

class AverageBinaryContinuousMetric(metrics.Metric):

    PARAMS = {'include_background_in_averages': False, 'apply_softmax': False}

    def __init__(self, name, binary_metric, key = None, _config_dict = None,
                 return_classwise_kw = None, base_name = None, base_name_plural = None,
                 class_names = [], *args, **kwargs):
        
        ignore_background = not _config_dict['metrics/calculation/include_background_in_averages']
        self.start = int(ignore_background)
        self.num_classes = _config_dict['metrics/calculation/number_of_classes']
        self.prob = torch.nn.Softmax(dim = 1) if _config_dict['metrics/calculation/apply_softmax'] else (lambda x: x)
        self.binary_calcs = [binary_metric(*args, **kwargs) for _ in range(self.start, self.num_classes)]
        self.name = name
        self.key = key or self.binary_calcs[0].name
        self.neutral = getattr(self.binary_calcs[0], 'neutral', 0)
        
        self.return_classwise = return_classwise_kw and _config_dict.get(f'metrics/calculation/{return_classwise_kw}', False)
        
        if self.return_classwise:
            self.labels = class_names
            if len(self.labels) > self.num_classes - self.start:
                self.labels = self.labels[1:]
            self.base_name = base_name or self.name
            self.metric_names = ['_'.join((self.base_name, label)).replace(' ', '_') for label in self.labels]
        
            self.plural_name = base_name_plural or self.base_name + 's'
            
            self.to_validate = kwargs['validate']
            self.save_path = _config_dict.get_str('meta/technical/absolute_path') + kwargs['exp_name'] + '/'
            self.extensions = _config_dict.get_str_tuple('meta/technical/export_plots_as')
            self.neptune_run = kwargs['neptune_run']
            self.neptune_save_path = kwargs['neptune_save_path'].split('/')[0] + '/plots'
            self.REQUIRES_LAST_PASS = True

    def calculate_batch(self, prediction, mask = None, label = None, *args, **kwargs):

        values = []
        prediction = self.prob(prediction)
        for i, binary_metric in enumerate(self.binary_calcs):
            idx = i + self.start
            bin_y = prediction[:, idx]
            bin_mask = None if mask is None else (mask == idx).int()
            bin_label = None if label is None else (label == idx).int()
            k_value = binary_metric.calculate_batch(prediction = bin_y, mask = bin_mask, label = bin_label, *args, **kwargs)
            if k_value:
                values.append(k_value[self.key])

        values_dict = {self.name: self.neutral if values == [] else np.mean(values)}
        if self.return_classwise:
            values_dict.update(dict(zip(self.metric_names, values)))
        return values_dict

    def evaluate_batch(self, *args, **kwargs):
        values = [bin_metric.evaluate_batch(*args, **kwargs)[self.key] for bin_metric in self.binary_calcs]
        values_dict = {self.name: np.mean(values)}
        if self.return_classwise:
            values_dict.update(dict(zip(self.metric_names, values)))
        return values_dict
    
    def evaluate_epoch(self, *args, **kwargs):
        values = [bin_metric.evaluate_epoch(*args, **kwargs)[self.key] for bin_metric in self.binary_calcs]
        values_dict = {self.name: np.mean(values)}
        if self.return_classwise:
            values_dict.update(dict(zip(self.metric_names, values)))
        return values_dict
    
    def evaluate_at_end(self, *args, **kwargs):
        if not self.to_validate: # TODO
            return
        prefix = 'val_metrics/' if self.to_validate else 'metrics/'
        logs = pd.read_csv(self.save_path + 'epoch_logs.csv')
        metric_logs = [logs[prefix + name].to_list() for name in self.metric_names]
        plotter = utils.framework.plotters.GeneralPlotter(dict(Ys = metric_logs,
                                                               xlabel = 'epoch', ylabel = self.base_name,
                                                               title = f'Validation {self.plural_name}'.replace('_', ' '),
                                                               legend = {'labels': self.metric_names},
                                                               dirname = self.save_path + 'plots/',
                                                               fname = f'{self.plural_name}_plot'),
                                                          self.neptune_run[self.neptune_save_path])
        utils.export_plot(plotter, self.extensions)

class AUROC(AverageBinaryContinuousMetric):
    
    RETURN_CLASSWISE_KW = 'log_classwise_auroc'
    RETURN_CLASSWISE_DEFAULT = False
    
    @staticmethod
    def fill_kwargs(config_dict):
        config_dict.get_or_update(AUROC.RETURN_CLASSWISE_KW, AUROC.RETURN_CLASSWISE_DEFAULT)
    
    def __init__(self, *args, **kwargs):
        super().__init__(name = 'area_under_roc', binary_metric = metrics.AUROC,
                         return_classwise_kw = self.RETURN_CLASSWISE_KW,
                         base_name_plural = 'areas_under_roc',
                         *args, **kwargs)
        
class AveragePrecision(AverageBinaryContinuousMetric):
    
    RETURN_CLASSWISE_KW = 'log_classwise_average_precision'
    RETURN_CLASSWISE_DEFAULT = False
    
    @staticmethod
    def fill_kwargs(config_dict):
        config_dict.get_or_update(AveragePrecision.RETURN_CLASSWISE_KW, AveragePrecision.RETURN_CLASSWISE_DEFAULT)
    
    def __init__(self, *args, **kwargs):
        super().__init__(name = 'average_precision_score', binary_metric = metrics.AveragePrecision,
                         return_classwise_kw = self.RETURN_CLASSWISE_KW, *args, **kwargs)

class AverageBinaryDiscreteMetric(metrics.Metric):

    PARAMS = {'include_background_in_averages': False}
    
    def __init__(self, name, binary_metric, key = None, _config_dict = None, *args, **kwargs):
        
        ignore_background = not _config_dict['metrics/calculation/include_background_in_averages']
        self.start = int(ignore_background)
        self.num_classes = _config_dict['metrics/calculation/number_of_classes']
        self.binary_calcs = [binary_metric(*args, **kwargs) for _ in range(self.start, self.num_classes)]
        self.add_to_average = [False for _ in range(self.start, self.num_classes)]
        self.name = name
        self.key = key or self.binary_calcs[0].name
        self.neutral = getattr(self.binary_calcs[0], 'neutral', 0)

    def calculate_batch(self, prediction, mask = None, label = None, *args, **kwargs):
        y = prediction.argmax(1)
        values = []
        for i, binary_metric in enumerate(self.binary_calcs):
            idx = i + self.start
            bin_y = (y == idx).int().unsqueeze(1)
            bin_mask = None if mask is None else (mask == idx).int()
            bin_label = None if label is None else (label == idx).int()
            bin_y_hat = bin_mask if bin_mask is not None else bin_label
            if bin_y.sum() + bin_y_hat.sum() > 0:
                k_value = binary_metric.calculate_batch(prediction = bin_y, mask = bin_mask, label = bin_label, *args, **kwargs)
                if k_value:
                    values.append(k_value[self.key])
                self.add_to_average[i] = True

        return {self.name: self.neutral if values == [] else np.mean(values)}

    def evaluate_batch(self, *args, **kwargs):
        values = [bin_metric.evaluate_batch(*args, **kwargs)[self.key] for bin_metric in self.binary_calcs]
        values = np.array(values)[self.add_to_average]
        self.add_to_average = [False for _ in range(self.start, self.num_classes)]
        return {self.name: self.neutral if len(values) == 0 else np.mean(values)}
    
    def evaluate_epoch(self, *args, **kwargs):
        return {self.name: np.mean([bin_metric.evaluate_epoch(*args, **kwargs)[self.key] for bin_metric in self.binary_calcs])}

class ModifiedHausdorffDistance(AverageBinaryDiscreteMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name = 'modified_hausdorff_distance',
                         binary_metric = metrics.ModifiedHausdorffDistance,
                         key = 'modified_hausdorff_distance_threshold_0.5', # TODO
                         *args, **kwargs)