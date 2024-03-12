import re
import numpy as np
import pandas as pd
import torch

from metrics import Metric
import regression.metrics as metrics
import regression.multidim_metrics as multidim_metrics

class AggregatePredictionLabelAndIndex(metrics.AggregateInput):
    
    y_pattern = re.compile('y_\d+')
    y_hat_pattern = re.compile('y_hat_\d+')
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(_config_dict, inputs = dict(y = 'label', y_hat = 'prediction'),
                         *args, **kwargs)
        self.train_df, self.val_df = pd.DataFrame(), pd.DataFrame()
    
    def update_df(self, df, label, prediction, index):
        data_dict = {**{f'y_{i}': label_slice for i, label_slice in enumerate(label)},
                     **{f'y_hat_{i}': pred_slice for i, pred_slice in enumerate(prediction)},
                     'count': 1}
        if df.empty:
            df = pd.DataFrame(data_dict, index = [index])
        elif index not in df.index:
            df.loc[index] = pd.Series(data_dict)
        else:
            df.loc[index] += pd.Series(data_dict)
        return df
    
    def calculate_batch(self, label, prediction, index, last = False, train = True, *args, **kwargs):
        if self.first and (self.always or self.epoch_idx - 1 in self.active_epochs):
            for attr in self.attr_names:
                setattr(type(self), attr, np.array([]))
            self.first = False
        if self.always or (last and self.do_last) or self.epoch_idx in self.active_epochs:
            train_or_val = 'train' if train else 'val'
            df = getattr(self, f'{train_or_val}_df')
            for y, y_hat, idx in zip(label.cpu().numpy(),
                                     prediction.cpu().detach().numpy(),
                                     index.cpu().numpy().squeeze()):
                df = self.update_df(df, y, y_hat, idx)
            setattr(self, f'{train_or_val}_df', df)
    
    def evaluate_epoch(self, *args, **kwargs):
        train_or_val = 'train' if self.train else 'val'
        df = getattr(self, f'{train_or_val}_df')
        avg_df = df.divide(df['count'], axis = 0)
        y_cols = [col_name for col_name in df.columns if self.y_pattern.match(col_name)]
        y_hat_cols = [col_name for col_name in df.columns if self.y_hat_pattern.match(col_name)]
        y = avg_df[y_cols].to_numpy()
        y_hat = avg_df[y_hat_cols].to_numpy()
        
        setattr(type(self), f'{train_or_val}_y', y)
        setattr(type(self), f'{train_or_val}_y_hat', y_hat)
        
        setattr(self, f'{train_or_val}_df', pd.DataFrame())
        super().evaluate_epoch(*args, **kwargs)

class IndexAveragedMetric(Metric):
    
    PARENT_METRIC = AggregatePredictionLabelAndIndex
    
    def __init__(self, metric, _config_dict = {}, *args, **kwargs):
        if Metric not in getattr(metric, '__mro__', []):
            self.metric = Metric(metric, _config_dict, *args, **kwargs)
        else:
            self.metric = metric(_config_dict, *args, **kwargs)
        metric_name = getattr(self.metric, 'name', Metric.convert_to_snake(metric.__name__))
        self.name = f'index_averaged_{metric_name}'
        self.metric.name = self.name
        self.train = True
    
    def calculate_batch(self, *args, **kwargs):
        return
    
    def evaluate_batch(self, train = True, *args, **kwargs):
        self.train = train
    
    def evaluate_epoch(self, *args, **kwargs):
        train_or_val = 'train' if self.train else 'val'
        y, y_hat = (getattr(self.PARENT_METRIC, f'{train_or_val}_{attr}') for attr in ('y', 'y_hat'))
        pred, label = torch.tensor(y_hat), torch.tensor(y)
        self.metric.calculate_batch(prediction = pred, label = label, train = self.train, *args, **kwargs)
        self.metric.evaluate_batch(prediction = pred, label = label, train = self.train, *args, **kwargs)
        return self.metric.evaluate_epoch(*args, **kwargs)

class L1Loss(IndexAveragedMetric): # TODO
    
    def __init__(self, *args, **kwargs):
        super().__init__(torch.nn.L1Loss)
    
class RelativeL1Distance(IndexAveragedMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(metrics.RelativeL1Distance, *args, **kwargs)

class R2Score(IndexAveragedMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(metrics.R2Score, *args, **kwargs)

class AbsoluteOrRelativeError(IndexAveragedMetric):
    
    PARAMS = metrics.AbsoluteOrRelativeError.PARAMS
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(metrics.AbsoluteOrRelativeError, _config_dict, *args, **kwargs)

class AbsoluteOrRelativeAccuracy(IndexAveragedMetric):
    
    PARAMS = metrics.AbsoluteOrRelativeAccuracy.PARAMS
    
    def __init__(self, _config_dict, threshold = 0.05, *args, **kwargs):
        super().__init__(metrics.AbsoluteOrRelativeAccuracy, _config_dict, threshold = threshold, *args, **kwargs)

class ElementwiseAccuracies(IndexAveragedMetric):
    
    PARAMS = multidim_metrics.ElementwiseAccuracies.PARAMS
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(multidim_metrics.ElementwiseAccuracies, _config_dict, *args, **kwargs)

class WeightedAccuracy(IndexAveragedMetric):
    
    PARAMS = multidim_metrics.WeightedAccuracy.PARAMS
    
    def __init__(self, _config_dict, threshold = 0.05, *args, **kwargs):
        super().__init__(multidim_metrics.WeightedAccuracy, _config_dict, threshold = threshold, *args, **kwargs)

class StrictAccuracy(IndexAveragedMetric):
    
    PARAMS = multidim_metrics.StrictAccuracy.PARAMS
    
    def __init__(self, _config_dict, threshold = 0.05, *args, **kwargs):
        super().__init__(multidim_metrics.StrictAccuracy, _config_dict, threshold = threshold, *args, **kwargs)
    
class LenientStrictAccuracy(IndexAveragedMetric):
    
    PARAMS = multidim_metrics.LenientStrictAccuracy.PARAMS
    
    def __init__(self, _config_dict, threshold = 0.05, *args, **kwargs):
        super().__init__(multidim_metrics.LenientStrictAccuracy, _config_dict, threshold = threshold, *args, **kwargs)

class IndexAveragedPlot(IndexAveragedMetric):
    def __init__(self, plotter, _config_dict, *args, **kwargs):
        super().__init__(plotter, _config_dict, *args, **kwargs)
        self.metric.PARENT_METRIC = AggregatePredictionLabelAndIndex

class ScatterPlot(IndexAveragedPlot):
    
    PARAMS = metrics.ScatterPlot.PARAMS
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(metrics.ScatterPlot, _config_dict, *args, **kwargs)
        self.metric.base_fname = 'index_averaged_scatter_plot'
        
class RecordwiseAccuracyHistogram(IndexAveragedPlot):
    
    PARAMS = multidim_metrics.RecordwiseAccuracyHistogram.PARAMS
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(multidim_metrics.RecordwiseAccuracyHistogram, _config_dict, *args, **kwargs)
        self.metric.base_fname = 'index_averaged_accuracy_histogram'

class ThresholdAccuracyCurve(IndexAveragedPlot):
    
    PARAMS = metrics.ThresholdAccuracyCurve.PARAMS
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(metrics.ThresholdAccuracyCurve, _config_dict, *args, **kwargs)
        self.metric.fname = 'index_averaged_accuracy_curve_epoch_{}'

class ThresholdStrictAccuracyCurve(IndexAveragedPlot):
    
    PARAMS = multidim_metrics.ThresholdStrictAccuracyCurve.PARAMS
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(multidim_metrics.ThresholdStrictAccuracyCurve, _config_dict, *args, **kwargs)
        self.metric.fname = 'index_averaged_strict_accuracy_curve_epoch_{}'

class ThresholdLenientStrictAccuracyCurve(IndexAveragedPlot):
    
    PARAMS = multidim_metrics.ThresholdLenientStrictAccuracyCurve.PARAMS
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(multidim_metrics.ThresholdLenientStrictAccuracyCurve, _config_dict, *args, **kwargs)
        self.metric.fname = 'index_averaged_lenient_strict_accuracy_curve_epoch_{}'

class ThresholdWeightedAccuracyCurve(IndexAveragedPlot):
    
    PARAMS = multidim_metrics.ThresholdWeightedAccuracyCurve.PARAMS
    
    def __init__(self, _config_dict, *args, **kwargs):
        super().__init__(multidim_metrics.ThresholdWeightedAccuracyCurve, _config_dict, *args, **kwargs)
        self.metric.fname = 'index_averaged_weighted_accuracy_curve_epoch_{}'
