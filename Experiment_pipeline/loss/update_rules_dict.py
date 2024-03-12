import update_rules

# dictionary of implented loss combination weight update rule schemes
# keys are the names of schemes, values are dicts
# every value should have an 'init' and 'arguments' entry
# 'init' is a callable that constructs an instance of the activation function
# 'arguments' is a dict of the hyperparameters of the activation function
# with keys corresponding to hyperparameters, and values being dicts
# with a 'default' value that sets the value given to the hyperparameter if not otherwise specified
# and an 'argument name' str that specifies what is the name of the argument
# corresponding to that hyperparameter in the constructor function 'init'
# 'argument name' can be omitted if it is the same as the key

# NOTE: update rule names should be added in all lowercase

update_rules_dict = {
    'linear change': {
        'init': update_rules.LinearChange,
        'arguments': {
            'loss to change': {
                'argument name': 'loss_to_change',
                'default': 'boundary loss'
                },
            'change factor': {
                'argument name': 'change_factor',
                'default': 0.001
                },
            'step frequency': {
                'argument name': 'step_frequency',
                'default': 1
                }
        }   
    },
    'linear rebalance': {
        'init': update_rules.LinearRebalance,
        'arguments': {
            'loss to increase': {
                'argument name': 'loss_to_increase',
                'default': 'boundary loss'
                },
            'loss to decrease': {
                'argument name': 'loss_to_decrease',
                'default': 'binary crossentropy'
                },
            'change factor': {
                'argument name': 'change_factor',
                'default': 0.001
                },
            'step frequency': {
                'argument name': 'step_frequency',
                'default': 1
                }
        }
    },
    'discontinuous change': {
        'init': update_rules.DiscontinousChange,
        'arguments': { # TODO
            'changes': {
                'argument name': 'change_dict',
                'default': {}
                }
            }
    }
}