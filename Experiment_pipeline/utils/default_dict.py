import re
from numpy.random import default_rng

def boolean(string):
    """Casts string to boolean."""
    string = string.lower()
    if string == 'true':
        return True
    if string == 'false':
        return False
    raise ValueError('\'{}\' is not a valid boolean value.'.format(string))

def my_list(list_type):
    """Creates caster function that transforms a string to a list containing list_type elements."""
        
    list_pattern = re.compile('\[\s*([\w. ,]+)\s*\]')
    empty_list_pattern = re.compile('\[\s*\]')
    split_pattern = re.compile(', *')

    def cast(list_as_str):
        if empty_list_pattern.match(list_as_str):
            return []
        m = list_pattern.match(list_as_str)
        if m:
            split_list = split_pattern.split(m.group(1))
            return list(map(list_type, split_list))
        raise ValueError('Cannot convert \'{}\' to list'.format(list_as_str))
    return cast

def my_tuple(tuple_type):
    """Creates caster function that transforms a string to a tuple containing tuple_type elements."""
        
    tuple_pattern = re.compile('\(\s*([\w. ,]+)\s*\)')
    empty_tuple_pattern = re.compile('\(\s*\)')
    split_pattern = re.compile(', *')

    def cast(tuple_as_str):
        if empty_tuple_pattern.match(tuple_as_str):
            return tuple()
        m = tuple_pattern.match(tuple_as_str)
        if m:
            split_list = split_pattern.split(m.group(1))
            return tuple(map(tuple_type, split_list))
        raise ValueError('Cannot convert \'{}\' to tuple.'.format(tuple_as_str))
    return cast

def my_dict(key_type, value_type):
    """Creates caster function that transforms a string to a dictionary containing key_type keys and value_type values."""

    dict_pattern = re.compile('\{\s*([\w\s:.,(){}]+)\s*\}')
    entry_pattern = '((?:[\w\s.()]*\w)|(?:\{[\w\s.,():]*\})|(?:\[[\w\s.,()]*\]))'
    k_v_pair_pattern = re.compile('\s*({}\s*:\s*{})\s*,?'.format(entry_pattern, entry_pattern))
    empty_dict_pattern = re.compile('\{\s*\}')

    def cast(dict_as_str):
        if empty_dict_pattern.match(dict_as_str):
            return {}
        m = dict_pattern.match(dict_as_str)
        if m:
            cast_dict = {}
            for _, key, value in k_v_pair_pattern.findall(m.group(1)):
                cast_dict[key_type(key)] = value_type(value)
            return cast_dict
        raise ValueError('Could not convert \'{}\' to dict'.format(dict_as_str))
    return cast

def multi_type(*types):
    """Creates a caster function that casts a string to either one of a list of types."""
    def cast(x):
        for caster_function in types:
            try:
                return caster_function(x)
            except ValueError:
                continue
        raise ValueError('Could not convert \'{}\' to any of {}'.format(x, types))
    return cast

# dictionary of general hyperparameters
# keys are the names of hyperparameters, values are dicts
# every value should have an 'init' and 'arguments' entry
# 'init' is a callable that constructs an instance of the activation function
# 'arguments' is a dict of the hyperparameters of the activation function
# with keys corresponding to hyperparameters, and values being dicts
# with a 'default' value that sets the value given to the hyperparameter if not otherwise specified
# and an 'argument name' str that specifies what is the name of the argument
# corresponding to that hyperparameter in the constructor function 'init'

neptune_dict = {
                'meta/neptune run/project name': { # TODO: neptune paramétereket külön gyűjteni
                    # name of neptune project
                    'default': 'aielte/UNeXt',
                    'type': str,
                    'argument name': 'project_name',
                    },
                'meta/neptune run/tags': {
                    # tags that will upload to neptune
                    'default': [],
                    'type': my_list(str),
                    'argument name': 'tags'
                    },
                'meta/neptune run/description': {
                    # short description of the experiment
                    'default': '',
                    'type': str,
                    'argument name': 'description'
                    },
                'meta/neptune run/log source code': { # TODO: ez lehetne more versatile
                    # whether to log the source code to neptune
                    'default': True,
                    'type': boolean,
                    'argument name': 'log_source_code'
                    },
                'meta/neptune run/log model': {
                    'default': False,
                    'type': boolean,
                    'argument name': 'log_model_to_neptune'
                    }
}

default_dict = {'model': {
                    'default': 'segmentation.models.UNet',
                    'type': str,
                    'argument name': 'model_name'
                    },
                'meta/technical/experiment name': {
                    'default': 'Experiment'
                    },
                'meta/technical/save destination': {
                    # parent directory where directory for the experiment logs will be created
                    'default': '/data/unext/logs/',
                    'type': str,
                    'argument name': 'save_destination',
                    },
                'meta/technical/seed': {
                    # random seed used by various (but not all) random number generators
                    # useful for reproducing experiments
                    # NOTE: when running several experiments with the same command
                    # the seed will only initialise once
                    'default': int(default_rng().integers(1000000)),
                    'type': int,
                    'argument name': 'seed'
                    },
                'experiment/number of epochs': {
                    'default': 150,
                    'type': int,
                    'argument name': 'num_epochs'
                    },
                'experiment/number of trials': {
                    # number of times to repeat the same experiment (with different seeds)
                    'default': 1,
                    'type': int,
                    'argument name': 'nmbr_of_trials'
                    },
                'training/optimizer': {
                    'default': 'sgd',
                    'type': str,
                    'argument name': 'optimizer_name'
                    },
                'training/loss': {
                    'default': 'torch.nn.BCELoss',
                    'type': multi_type(my_dict(str, float), str),
                    'argument name': 'loss'
                    },
                'metrics/metrics': {
                    'default': ('metrics.DiceIndex',
                                'metrics.BalancedAccuracy',
                                'metrics.Sensitivity',
                                'metrics.Specificity',
                                'metrics.JaccardIndex',
                                'metrics.ModifiedHausdorffDistance',
                                'metrics.AUROC',
                                'metrics.ROCCurve',
                                'metrics.AveragePrecision',
                                'metrics.PrecisionRecallCurve',
                                'metrics.MCC',
                                'metrics.Accuracy',
                                'segmentation.image_logging.ImageLogger'),
                    'type': my_list(str),
                    'argument name': 'metrics'
                    },
                'data/transforms': { # TODO: lehet, hogy ez most megzavar mindent
                    # list of transformations to be used during training
                    # dict containing keys 'train' and 'val'
                    # and values that specify the transformations to be used for the
                    # training and validation data respectively
                    #
                    # TODO: if either key is missing, an empty list should be assumed 
                    'default': {'train': ('segmentation.transforms.wrapped_transforms.RandomRotation',
                                          'segmentation.transforms.wrapped_transforms.CenterCrop',
                                          'segmentation.transforms.wrapped_transforms.RandomFlip'),
                                          # 'soundy.transforms.GaussianNoise'),
                                          #{'torchvision.transforms.ColorJitter': { # TODO
                                          #    'brightness': 0.25,
                                          #    'contrast': 0.25,
                                          #    'hue': 0.25
                                          #}}
                        
                                'val': tuple()},
                    'type': my_dict(str, my_list(str)),
                    'argument name': 'transforms'
                    },
                'data/data': {
                    'default': 'segmentation.datasets.DRIVEDataset'
                    },
                'meta/technical/log to device': {
                    'default': True,
                    'type': boolean,
                    'argument name': 'log_to_device'
                    },
                'meta/technical/log to neptune': {
                    'default': True,
                    'type': boolean,
                    'argument name': 'log_to_neptune'
                    },
                'meta/technical/number of data loader workers': { #TODO: ez jó helyen van?
                    # number of workers that load batch into the GPU
                    'default': 0,
                    'type': int,
                    'argument name': 'num_workers'
                    },
                'meta/technical/log metric and loss plots': {
                    # whether to log images of the plotted metrics and loss
                    'default': True,
                    'type': boolean,
                    'argument name': 'log_plots'
                    },
                'meta/technical/maximum actual batch size': {
                    # maximum number of datapoints to load into memory at once
                    # per GPU devices used
                    # used for gradient accumulation
                    'default': 24, # TODO
                    'type': int,
                    'argument name': 'max_actual_batch_size'
                    },
                'meta/technical/verbose': False,
                'meta/technical/use_cudnn_benchmarking': True,
                'meta/technical/use_deterministic_algorithms': False,
                'meta/technical/number_of_cpu_threads': 16,
                'meta/technical/export_plots_as': ('json', 'html'),
                'meta/technical/log_best_model': True,
                'meta/technical/log_last_model': True,
                'meta/technical/memory_usage_limit': -1,
                'training/gradient_clipping/max_value': None,
                'training/gradient_clipping/norm': 2.0
                }

model_eval = {'metric': 'val_metrics/accuracy', 'mode': 'max'}