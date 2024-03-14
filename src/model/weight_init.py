from torch.nn import init
import utils

class InitWrapper(object):
    """Wrapper object for weight initialisation functions."""
    @staticmethod
    def convert(v):
        if isinstance(v, str):
            return v.replace(' ', '_')
        if isinstance(v, utils.ConfigDict):
            return InitWrapper.convert(v.key())
        return v

    def __init__(self, init_func,
                 bias_init = lambda tensor, **kwargs: init.zeros_(tensor),
                 *args, **kwargs):
        self.init_func = init_func
        self.bias_init = bias_init
        self.kwargs = {k: self.convert(v) for k, v in kwargs.items()}
    
    def __call__(self, layer, *args, **kwargs):
        self.init_func(layer.weight, **self.kwargs)
        if hasattr(layer, 'bias'):
            self.bias_init(layer.bias, **self.kwargs)

# dictionary of implented weight initialsiation schemes
# keys are the names of schemes, values are dicts
# every value should have an 'init' and 'arguments' entry
# 'init' is a callable that constructs an instance of the activation function
# 'arguments' is a dict of the hyperparameters of the activation function
# with keys corresponding to hyperparameters, and values being dicts
# with a 'default' value that sets the value given to the hyperparameter if not otherwise specified
# and an 'argument name' str that specifies what is the name of the argument
# corresponding to that hyperparameter in the constructor function 'init'
# 'argument name' can be omitted if it is the same as the key

# NOTE: initialiser names should be all lowercase

inits_dict = {
    'glorot uniform': {
        'init': lambda **kwargs: InitWrapper(init.xavier_uniform_, **kwargs),
        'arguments': {
            'gain': {
                'default': 1.0
                }
            }
        },
    'glorot normal': {
        'init': lambda **kwargs: InitWrapper(init.xavier_normal_, **kwargs),
        'arguments': {
            'gain': {
                'default': 1.0
                }
            }
        },
    'he uniform': {
        'init': lambda **kwargs: InitWrapper(init.kaiming_uniform_,
                                                   nonlinearity = 'relu',
                                                   **kwargs),
        'arguments': {
            'mode': {
                'default': 'fan in'
                }
            }
        },
    'he normal': {
        'init': lambda **kwargs: InitWrapper(init.kaiming_normal_,
                                                   nonlinearity = 'relu',
                                                   **kwargs),
        'arguments': {
            'mode': {
                'default': 'fan in'
                }
            }
        },
    'constant': {
        'init': lambda **kwargs: InitWrapper(init.constant_, **kwargs),
        'arguments': {
            'value': {
                'argument name': 'val',
                'default': 1.0
                }
            }
        },
    'fix uniform': {
        'init': lambda **kwargs: InitWrapper(init.uniform_, **kwargs),
        'arguments': {
            'minium': {
                'argument name': 'a',
                'default': 0.0
                },
            'maximum': {
                'argument name': 'b',
                'default': 1.0
                }
            }
        },
    'fix normal': {
        'init': lambda **kwargs: InitWrapper(init.normal_, **kwargs),
        'arguments': {
            'mean': {
                'default': 0.0
                },
            'std': {
                'default': 1.0
                }
            }
        },
    'torch default': {
        'init': lambda **kwargs: InitWrapper(lambda *args, **kwargs: None,
                                                  bias_init = lambda *args, **kwargs: None),
        'arguments': {}
        }
}