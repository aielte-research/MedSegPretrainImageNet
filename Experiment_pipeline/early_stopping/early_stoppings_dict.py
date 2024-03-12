import early_stopping

# dictionary of implented early stopping schemes
# keys are the names of schemes, values are dicts
# every value should have an 'init' and 'arguments' entry
# 'init' is a callable that constructs an instance of the activation function
# 'arguments' is a dict of the hyperparameters of the activation function
# with keys corresponding to hyperparameters, and values being dicts
# with a 'default' value that sets the value given to the hyperparameter if not otherwise specified
# and an 'argument name' str that specifies what is the name of the argument
# corresponding to that hyperparameter in the constructor function 'init'

early_stoppings_dict = { # TODO: what metric to watch
    'watched metric stops improving': {
        'init': early_stopping.PatientImprovement,
        'arguments': {
            'patience': {
                'argument name': 'patience',
                'default': 0
                },
            'statistic to compare': {
                'argument name': 'statistic',
                'default': 'best'
                },
            'comparison criterium': {
                'argument name': 'mode',
                'default': 'max'
                },
            'minimal improvement': {
                'argument name': 'eps',
                'default': 1e-4
                }
        }
    }
}