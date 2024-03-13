from torch import optim

def adam_init(beta_1 = 0.9, beta_2 = 0.999, decoupled_weight_decay = False, **kwargs):
    """Constructs either an Adam or AdamW (https://arxiv.org/abs/1711.05101) optimizer."""
    kwargs['betas'] = (beta_1, beta_2)
    if decoupled_weight_decay:
        return optim.AdamW(**kwargs)
    return optim.Adam(**kwargs)

# dictionary of implented optimizer schemes
# keys are the names of schemes, values are dicts
# every value should have an 'init' and 'arguments' entry
# 'init' is a callable that constructs an instance of the activation function
# 'arguments' is a dict of the hyperparameters of the activation function
# with keys corresponding to hyperparameters, and values being dicts
# with a 'default' value that sets the value given to the hyperparameter if not otherwise specified
# and an 'argument name' str that specifies what is the name of the argument
# corresponding to that hyperparameter in the constructor function 'init'
# 'argument name' can be omitted if it is the same as the key

# NOTE: optimiser names should be all lowercase

optimizers_dict = {'sgd': {
                        'init': optim.SGD,
                        'arguments': {
                            'momentum': {
                                'argument name': 'momentum',
                                'default': 0.9
                                },
                            'weight decay': {
                                'argument name': 'weight_decay',
                                'default': 0.0
                                },
                            'nesterov momentum': {
                                'argument name': 'nesterov',
                                'default': False
                                },
                            'momentum dampening': {
                                'argument name': 'dampening',
                                'default': 0.0
                                }
                            }
                        },
                   'adam': {
                        'init': adam_init,
                        'arguments': {
                            'beta_1': {
                                'argument name': 'beta_1',
                                'default': 0.9
                                },
                            'beta_2': {
                                'argument name': 'beta_2',
                                'default': 0.999
                                },
                            'weight decay': {
                                'argument name': 'weight_decay',
                                'default': 0.0
                                },
                            'decoupled weight decay': {
                                'argument name': 'decoupled_weight_decay',
                                'default': False
                                },
                            'amsgrad': {
                                'argument name': 'amsgrad',
                                'default': False
                                }
                            }
                        }
}
