import numpy as np
from transform import TransformWrapper

class ThresholdLabel(TransformWrapper):

    PARAMS = {'threshold': 0.2, 'flip': False}

    def __init__(self, config_dict, *args, **kwargs):
        self.th = config_dict['threshold']
        if config_dict['flip']:
            def transform(input):
                input['label'] = (np.abs(input['label']) < self.th).astype(int)
                return input
        else:
            def transform(input):
                input['label'] = (np.abs(input['label'] > self.th)).astype(int)
                return input
        self.transform = transform

class AverageLabel(TransformWrapper):
    
    def __init__(self, *args, **kwargs):
        return
    
    def transform(self, input):
        input['label'] = np.mean(input['label'], axis = 0, keepdims = True)
        return input