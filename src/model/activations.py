from torch import nn

# dictionary of implented activation functions
# keys are the names of activation functions, values are dicts
# every value should have an 'init' and 'arguments' entry
# 'init' is a callable that constructs an instance of the activation function
# 'arguments' is a dict of the hyperparameters of the activation function
# with keys corresponding to hyperparameters, and values being dicts
# with a 'default' value that sets the value given to the hyperparameter if not otherwise specified
# and an 'argument name' str that specifies what is the name of the argument
# corresponding to that hyperparameter in the constructor function 'init'

activation_funcs_dict = {
    'relu': {
        'init': lambda **kwargs: nn.ReLU(inplace = True),
        'arguments': {}
        },
    'sigmoid': {
        'init': nn.Sigmoid,
        'arguments': {}
        },
    'softmax': {
        'init': lambda **kwargs: nn.Softmax(dim = 1),
        'arguments': {}
        },
    'prelu': {
        'init': nn.PReLU,
        'arguments': {
            'initial negative slope': {
                'argument name': 'init',
                'default': 0.2
                }
            }
        },
    'leaky relu': {
        'init': nn.LeakyReLU,
        'arguments': {
            'negative slope': {
                'argument name': 'negative_slope',
                'default': 0.2
                }
            }
        },
    'gelu': {
        'init': nn.GELU,
        'arguments': {}
        },
    'linear': {
        'init': nn.Identity,
        'arguments': {}
        }
}