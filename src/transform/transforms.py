import builtins
import importlib
import numpy as np
import torch
import utils

from collections.abc import Iterable

class TransformWrapper(object):
    """
    General wrapper object for transforms that transform certain types of input, while leave others in place.
    """
    def __init__(self, transform, config_dict = {}, data_to_transform = ['x'], **kwargs):
        """
        Arguments:
            `transform`: callable; constructor of transformation object that takes in keywords arguments, and returns a dictionary with the same keys
            `data_to_tranform`: iterable or str; list of keywords that specify the input types that will be passed on to `transform` and transformed; if set to 'all', the transform will be applied to all inputs (expects all batch to have the same input keywords)
            `kwargs`: keyword arguments that will be passed onto the `transform` constructor
        """
        trsf_kwargs = utils.get_kwargs(transform, config_dict)
        self.trsf_func = transform(**trsf_kwargs)
        self.data_to_transform = data_to_transform
        if list(self.data_to_transform) != ['x']:
            self.transform = lambda input: self.trsf_func(**{k: v for k, v in input.items() if k in self.data_to_transform})
        else:
            def transform(input):
                x = input['x']
                if isinstance(x, (Iterable, int, float, complex, bool)) and not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)
                return self.trsf_func(x)
            self.transform = transform
        self.data_to_transform = data_to_transform
    
    def __call__(self, k = 1, **input):
        """Applies the transform to the input keywords specified in `self.data_to_transform`, and passes the rest of the inputs through unchanged."""
        output = input
        if getattr(self, 'data_to_transform', 'all') == 'all':
            self.data_to_transform = [x for x in input.keys() if x != '_index']
        if k == 1:
            transformed_value = self.transform(input)
        else:
            transformed_value = []
            for i in range(k):
                transformed_value.append(self.transform({key: value[i] for key, value in input.items()}))
            first_value = transformed_value[0]
            if isinstance(first_value, dict):
                transformed_value = {key: [value[key] for value in transformed_value] for key in first_value.keys()}
        if isinstance(transformed_value, dict):
            output.update(transformed_value)
        else:
            output['x'] = transformed_value
        output['k'] = output.get('k') or k
        return output

class GeneralTransformWrapper(TransformWrapper):
    """Wrapper object for transforms that should transform everything in the input."""
    def __init__(self, trsf):
        def transform(**input):
            return {k: trsf(v) for k, v in input.items()}
        
        super().__init__(lambda **kwargs: transform, data_to_transform = 'all')

class ConvertToType(TransformWrapper):

    TENSOR_TYPES = (
        torch.FloatTensor,
        torch.DoubleTensor,
        torch.HalfTensor,
        torch.ByteTensor,
        torch.CharTensor,
        torch.ShortTensor,
        torch.IntTensor,
        torch.LongTensor,
        torch.BoolTensor
    )

    @staticmethod
    def default_transform(x):
        if isinstance(x, torch.Tensor):
            return x.to(dtype = torch.float32)
        if not isinstance(x, Iterable):
            x = [x]
        return torch.FloatTensor(x)

    def __init__(self, types_dict, default = None, *args, **kwargs):
        def get_caster(name_str):
            if name_str is None:
                return lambda x: x
            module, caster_name = '.'.join(name_str.split('.')[:-1]), name_str.split('.')[-1]
            caster_func = getattr(importlib.import_module(module), caster_name)
            def caster(x):
                if not isinstance(x, Iterable):
                    x = [x]
                return caster_func(x)
            return caster if caster_func in self.TENSOR_TYPES else caster_func
        
        if isinstance(types_dict, utils.config_dict.ConfigDict):
            types_dict = types_dict.trim().to_dict()
        self.types_dict = {datapoint_type: get_caster(caster_name) for datapoint_type, caster_name in types_dict.items()}
        self.default = default or self.default_transform

    def transform(self, input):
        return {k: self.types_dict.get(k, self.default)(v) for k, v in input.items()}

class Compose(object):
    """Class for composing a list of transforms. Transforms will be performed in order."""
    def __init__(self, *transforms):
        self.transforms = transforms
    
    def __call__(self, **input):
        output = input
        for transform in self.transforms:
            output = transform(**output)
        return output

class LambdaTransform(TransformWrapper):
    
    def __init__(self, config_dict, *args, **kwargs):
        self.func = config_dict.get_str('function')
        self.kw = config_dict.get_str('out_keyword')
        
        if not isinstance(self.func, str):
            msg = f'Lambda function must be str, not {type(self.func)} ({self.func}).'
            raise TypeError(msg)
        if not isinstance(self.kw, str):
            msg = f'Keyword must be str, not {type(self.kw)} ({self.kw}).'
            raise TypeError(msg)
        self.globals = {'__builtins__': {**builtins.__dict__, **np.__dict__}}
    
    def transform(self, input):
        input[self.kw] = eval(self.func, self.globals, input)
        return input

class RepeatChannels(object):

    PARAMS = {'repeats': 3}
    
    def __init__(self, repeats = 3, *args, **kwargs):
        self.repeats = repeats
    
    def __call__(self, x, **kwargs):
        return np.repeat(x, self.repeats, axis = 0)

class Mixup(TransformWrapper):
    
    PARAMS = {'distribution': 'symmetric_beta', 'probability': 1.0}

    ALPHA = 1.0
    MAX = 1.0

    num_datapoints = 2

    def __init__(self, config_dict = {}, seed = None, *args, **kwargs):
        distribution = config_dict.trim()['distribution']
        self.rng = np.random.default_rng(seed)
        if distribution == 'symmetric_beta':
            alpha = config_dict.get_or_update('alpha', self.ALPHA)
            self.get_lambda = lambda: self.rng.beta(alpha, alpha)
        elif distribution == 'uniform':
            high = config_dict.get_or_update('maximum', self.MAX)
            self.get_lambda = lambda: self.rng.uniform(0, high)
        else:
            raise ValueError(f'Distribution must be one of \'symmetric_beta\' or \'uniform\', not \'{distribution}\'.')
        self.p = config_dict['probability']
    
    def __call__(self, **input): # TODO: deal with more than two datapoints in a sample
        if self.rng.binomial(1, self.p):
            (x1, x2), (y1, y2) = input['x'], input['label']
            input = {k: v[0] for k, v in input.items() if isinstance(v, list)}
            lda = self.get_lambda()
            input['x'] = (1 - lda) * x1 + lda * x2
            input['label'] = (1 - lda) * y1 + lda * y2
            input['k'] = 1
        return input

class CutMix(TransformWrapper):
    
    PARAMS = {'distribution': 'uniform', 'axes': (1, 2), 'probability': 1}

    num_datapoints = 2

    def __init__(self, config_dict = {}, seed = None, *args, **kwargs):
        distribution = config_dict.trim()['distribution']
        self.rng = np.random.default_rng(seed)
        if distribution == 'uniform':
            self.get_split = lambda size: self.rng.integers(0, size)
        else:
            raise ValueError
        self.p = config_dict.get('probability') or 1
        self.axes = np.array(config_dict.get('axes'))
    
    def __call__(self, **input): # TODO: deal with more than two datapoints in a sample
        if self.rng.binomial(1, self.p):
            (x1, x2), (y1, y2) = input['x'], input['label']
            input = {k: v[0] for k, v in input.items() if isinstance(v, list)}
            axis = self.rng.choice(self.axes)
            
            size = x1.shape[axis]
            cut = self.get_split(size)
            lda = cut / size
            
            x = getattr(x1, 'clone', getattr(x1, 'copy', lambda: x1))()
            x_slices = [slice(None) for _ in x.shape]
            x_slices[axis] = slice(cut, None)
            x_slices = tuple(x_slices)
            x[x_slices] = x2[x_slices]

            input['x'] = x
            input['label'] = (1 - lda) * y1 + lda * y2
            input['k'] = 1
        return input