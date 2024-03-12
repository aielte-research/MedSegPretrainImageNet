import copy
import gc
import json
import random
import resource
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import os
import socket
from datetime import datetime
import warnings

import neptune.new as neptune
import numpy as np
import pandas as pd
import torch
import yaml
from sympy import divisors
from collections.abc import Iterable

import data
import loss
import metrics
from metrics import early_stopping
import model
import optim
import transform
import utils
from exception_handling import handle_exception
from train_model import train_model
from utils.config_dict import ConfigDict


def experiment(config_dict : ConfigDict, original : Optional[Dict],
               series_id : str = str(random.randint(0, 1e16)),
               modifiers : Optional[List[str]] = None,
               *args, **kwargs):
    """
    Function that performs a number of trials with a given hyperparameter configuration and logs the results.

    NOTE: instead of directly calling this function, it is advised to run experiment.py with a config file containing the hyperparameters.
    """
    
    continued, run_start, epoch_start = check_for_continued(modifiers, config_dict)
    
    config_dict.expand() # transform str values of `config_dict` to ConfigDict({value: {}})
    config_dict.fill_with_defaults(utils.default_dict) # fill in the default technical parameters
    tech_params = config_dict['meta/technical']
    tech_params = tech_params.trim()
    config_dict['meta/technical'] = tech_params
    
    # limit the number of memory used
    memory_limit = tech_params.get('memory_usage_limit', -1)
    if isinstance(memory_limit, (int, float)):
        memory_limit = (memory_limit,) * 2
    memory_limit = [int(max(-1, 2**30 * limit)) for limit in memory_limit]
    resource.setrlimit(resource.RLIMIT_DATA, memory_limit)
    
    # limit the number of CPU threads
    torch.set_num_threads(tech_params['number_of_cpu_threads'])
    
    # use convolution benchmarking if desired
    torch.backends.cudnn.benchmark = tech_params['use_cudnn_benchmarking']
    # use a deterministic variant (if available) for every operation
    torch.use_deterministic_algorithms(tech_params['use_deterministic_algorithms'], warn_only = True)

    # get the number of cuda devices used in the experiment
    device_count = max(torch.cuda.device_count(), 1)

    data.BalancedDataLoader.fill_kwargs(config_dict.get_or_update('data/sampling', utils.ConfigDict({})))

    # if the actual batch size is too big for memory, use smaller effective batch size
    max_bs : int = tech_params['maximum actual batch size']
    batch_size : int = config_dict['data/sampling/batch size']
    bs = max(filter(lambda n: n <= device_count * max_bs, divisors(batch_size)))
    
    seed = tech_params['seed'] + epoch_start
    
    # create two lists of transforms that will be applied on the train and validation datasets
    transforms = {'train': [], 'val': []}

    # create transforms
    trsfs_dict = config_dict.get_or_update('data/transforms', ConfigDict({'train': [], 'val': []}))
    partition_count = {'train': None, 'val': None}
    datapoint_count = {'train': 1, 'val': 1}
    for ds_type in ('train', 'val'):
        for tr_config_dict in trsfs_dict.elements_of(ds_type):
            utils.fill_dict(tr_config_dict)
            transf = utils.create_object_from_dict(tr_config_dict.trim(),
                                                   seed = seed,
                                                   wrapper_class = transform.TransformWrapper)
            transforms[ds_type].append(transf)
            if 'number of partitions' in tr_config_dict.value():
                # get partition count (needed for later batching)
                partition_count[ds_type] = np.product(tr_config_dict.value().get_tuple('number_of_partitions'))
            num_datapoints = getattr(transf, 'num_datapoints', 1)
            datapoint_count[ds_type] = max(datapoint_count[ds_type], num_datapoints)
    
    train_transforms, val_transforms = tuple(transforms.values())
    # cast the values to float tensors
    transform_to_tensor = transform.ConvertToType(config_dict.get('data/transforms/casting', {}))
    train_transforms.append(transform_to_tensor)
    val_transforms.append(transform_to_tensor)

    # compose the train and val transformations
    train_transfs = transform.Compose(*train_transforms)
    val_transfs = transform.Compose(*val_transforms)

    # fill in the config dict with the default hyperparameteres of the model, loss, dataset, optimizer and metrics
    # (in case they are missing)
    keys = ('model', 'training/loss', 'data/data')

    for key in keys:
        utils.fill_dict(config_dict, key)
        
    model.Model.fill_weight_init_kwargs(config_dict['model'].value())

    optim.Optimizer.fill_kwargs(config_dict['training/optimizer'])
    metrics.MetricsCalculator.fill_kwargs(config_dict)

    try:
        added_tags = fill_dict_with_name_fields(config_dict)
    except Exception as e:
        handle_exception(e, 'Unexpected exception occured while trying to fill out name fields.')
        added_tags = []

    if not continued:    
        # create directory where the experiment logs will be saved
        save_destination = os.path.abspath(tech_params.get_or_update('save destination')) + '/'
        date_folder = str(datetime.now()).split(' ')[0]
        save_destination = save_destination + date_folder + '/'
        if not os.path.isdir(save_destination):
            os.mkdir(save_destination)
        if tech_params['log to device']:
            save_destination += tech_params['experiment_name']
            default_dest = save_destination
            i = 1
            while os.path.isdir(default_dest):
                i += 1
                default_dest = f'{save_destination}_{i}'
            save_destination = os.path.abspath(default_dest) + '/'
            os.mkdir(save_destination)
            tech_params['server'] = socket.gethostname()
            tech_params['absolute path'] = save_destination
            tech_params['series_id'] = series_id
    else:
        save_destination = tech_params['absolute_path']

    if not continued:
        # initialises neptune run    
        neptune_run = None
        if not continued:
            if tech_params['log to neptune']:
                config_dict.fill_with_defaults(utils.neptune_dict, final = True)
                neptune_params = config_dict['meta/neptune run'].trim()
                neptune_run = neptune.init_run(
                    neptune_params['project name'],
                    name = tech_params['experiment name'],
                    tags = neptune_params['tags'],
                    description = neptune_params['description'],
                    source_files = '*.py' if neptune_params['log source code'] else None
                )
                config_dict['meta/neptune_run/run_id'] = neptune_run.get_url().split('/')[-1]
                                        
                neptune_run['parameters'] = config_dict.trim().to_dict()
                neptune_run['series_id'] = series_id
            else:
                neptune_run = neptune.init(config_dict['meta/neptune_run/project_name'],
                                           run = config_dict['meta/neptune_run/run_id'])
    else:
        neptune_run = neptune.init(config_dict.get_str('meta/neptune_run/project_name'),
                                   run = config_dict.get_str('meta/neptune_run/run_id'))
    
    if tech_params['log_best_model']:
        tech_params.get_or_update('model_evaluation', ConfigDict()).fill_with_defaults(utils.model_eval)
    
    if not continued and tech_params['log to device']:
        # get start time
        config_dict['meta/technical/start time'] = ''.join(str(datetime.now()).split('.')[:-1])
                    
        with open(save_destination + 'config.yaml', 'w') as file: # log 
            yaml.dump(config_dict.trim().to_dict(lists_to_tuples = True), file, sort_keys = False)
        if tech_params['log to neptune']:
            neptune_run['config'].upload(save_destination + 'config.yaml')
        
        if original is not None:
            with open(save_destination + 'source_config.yaml', 'w') as file:
                yaml.dump(original, file, sort_keys = False)
            if tech_params['log to neptune']:
                neptune_run['source_config'].upload(save_destination + 'source_config.yaml')

    # run trials
    for i in range(run_start, config_dict['experiment/number of trials'] + 1):
        try:
            # change seed
            curr_seed = seed + i - 1
            curr_destination = save_destination + 'run_{}/'.format(i)

            # make the train/validation split using the current random seed
            ds_object = utils.create_object_from_dict(config_dict['data/data'].trim(),
                                                      wrapper_class = data.Dataset,
                                                      seed = curr_seed)
            train_data, val_data = ds_object.train, getattr(ds_object, 'val', {})
            test_data = getattr(ds_object, 'test', {})
            load_function = getattr(ds_object, 'load_function', None)
            class_names = getattr(ds_object, 'CLASSES',
                                  config_dict.get_str_tuple('metrics/calculation/class_names',
                                                            tuple(f'class_{i}' for i in range(config_dict.get('metrics/calculation/number_of_classes', 0)))))

            train_ds = data.BalancedDataLoader(train_data,
                                               config_dict['data/sampling/train'],
                                               bs = batch_size,
                                               actual_bs = bs,
                                               num_workers = tech_params['number of data loader workers'],
                                               transforms = train_transfs,
                                               partition_count = partition_count['train'],
                                               load_function = load_function,
                                               seed = curr_seed,
                                               datapoints_per_sample = datapoint_count['train'],
                                            )
            val_ds = data.BalancedDataLoader(val_data,
                                             config_dict['data/sampling/val'],
                                             bs = batch_size,
                                             actual_bs = bs,
                                             num_workers = tech_params['number of data loader workers'],
                                             transforms = val_transfs,
                                             partition_count = partition_count['val'],
                                             load_function = load_function,
                                             seed = curr_seed,
                                             datapoints_per_sample = datapoint_count['val']
                                            )
            
            if test_data:
                test_ds = data.BalancedDataLoader(test_data,
                                                  config_dict['data/sampling/val'],
                                                  bs = batch_size,
                                                  actual_bs = bs,
                                                  num_workers = tech_params['number of data loader workers'],
                                                  transforms = val_transfs,
                                                  partition_count = partition_count['val'],
                                                  load_function = load_function,
                                                  seed = curr_seed,
                                                  datapoints_per_sample = datapoint_count['val']
                                                )
            else:
                test_ds = None

            # run trial
            run_exp(train_ds, val_ds, test_ds,
                    curr_destination, curr_seed + epoch_start * (run_start != i),
                    config_dict.trim(), neptune_run,
                    batch_size, bs, save_destination = save_destination,
                    partition_count = partition_count, idx = i, class_names = class_names,
                    epoch_start = epoch_start * (i == run_start),
                    continued = continued and i == run_start)
            
            del ds_object, train_data, val_data, train_ds, val_ds
            gc.collect()
            
            extensions = tech_params.get_str_tuple('export_plots_as')
            compare_experiments(i, save_destination, neptune_run, extensions)

        # catch exception and print them
        # if the exception is too long, it will be logged to a file
        except Exception as e:
            exp_name = tech_params['experiment name']
            message = f'Exception occured in run {i} of experiment \'{exp_name}\'.'
            handle_exception(e, message)
        
    if tech_params['log to device']: # log the list of all used modules and their version
        modules = {name: module.__version__ for name, module in sys.modules.copy().items()
                   if hasattr(module, '__version__')}
        lines = (f'{name}=={version}\n' for name, version in modules.items())
        with open(save_destination + 'environment.txt', 'w') as file:
            file.writelines(lines)
        if tech_params['log to neptune']:
            neptune_run['environment'].upload(save_destination + 'environment.txt')
            
    if tech_params['log to neptune']:
        neptune_run.stop()
    
    log_data = {}

    if tech_params['log_to_device']:
        log_data = {'current_experiment': True,
                    'exp_name': tech_params['experiment_name'],
                    'save_path': save_destination,
                    'num_trials': config_dict['experiment/number_of_trials'],
                    'log_to_neptune': tech_params['log_to_neptune'],
                    'tags': added_tags}

        if tech_params['log_to_neptune']:
                log_data['neptune_project_name'] = neptune_params['project_name']
                log_data['neptune_run_id'] = config_dict['meta/neptune_run/run_id']
    
    return log_data
    

def run_exp(train_data : Dict[str, Any], val_data : Dict[str, Any],
            test_data : Optional[Dict[str, Any]], destination : str,
            curr_seed : int, config_dict : ConfigDict,
            run : Union[neptune.Run, None], batch_size : int,
            bs : int, idx : Optional[int] = None,
            class_names : Tuple[str] = tuple(),
            continued : bool = False, epoch_start : int = 0,
            **kwargs):
        
        tech_params = config_dict['meta/technical']
        name = 'run_{}'.format(idx)
        
        # create the destination directory
        if tech_params['log to device'] and not continued:
            os.mkdir(destination)
        
        # configures the random number generators used by PyTorch with the current seed
        # useful for reproducibility
        torch.manual_seed(curr_seed)
        np.random.seed(curr_seed)
        random.seed(curr_seed)
        
        # initialises model
        nn_model : model.Model = utils.create_object_from_dict(config_dict, key = 'model',
                                                               wrapper_class = model.Model)
        model_dict = config_dict['model'].value()
        if continued:
            weights_path = os.path.join(destination, 'last_model_state_dict.pt')
            model_dict['weight_init'] = ConfigDict(dict(weights = weights_path,
                                                        strict = False))
            torch.save(torch.load(weights_path), os.path.join(destination, 'last_model_state_dict_checkpoint.pt'))
            
        nn_model.init_weight(model_dict)
        nn_model.freeze_and_unfreeze(model_dict)
        if continued:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            nn_model = nn_model.to(device)

        # log GFLOP count and number of parameters to neptune
        if idx in (1, None) and tech_params['log to neptune'] and not continued:
            try:
                copied_data = copy.deepcopy(train_data)
                run['model/GFLOPs'] = nn_model.get_number_of_flops(copied_data) / 1e9
                del copied_data
                gc.collect()
            except Exception as e:
                handle_exception(e, 'Exception occured while trying to calculate FLOP count.')
            try:
                run['model/parameters'] = nn_model.get_num_params()
                run['model/trainable_parameters'] = nn_model.get_num_params(trainable_only = True)
            except Exception as e:
                handle_exception(e, 'Exception occured while trying to calculate number of parameters.')

        # initialises optimiser with given hyperparameters
        optim_dict = config_dict['training/optimizer']
        optimizer = optim.Optimizer(optim_dict, nn_model.parameters())

        # logs the number of batches in an epoch
        batches_per_epoch = math.ceil(len(train_data) / batch_size * bs)
        run[name + '/epoch_logs/batches per epoch/'] = batches_per_epoch

        scheduler = None
        scheduler_dict = optim_dict.value()['learning_rate']
        schedule = scheduler_dict.key() != 'constant'
        if schedule:
            scheduler = utils.create_object_from_dict(scheduler_dict,
                                                      wrapper_class = optim.scheduler.SchedulerWrapper,
                                                      optimizer = optimizer,
                                                      num_epochs = config_dict['experiment/number_of_epochs'],
                                                      batches_per_epoch = batches_per_epoch)
        
        if continued:
            optim_state_dict = torch.load(os.path.join(destination, 'optimizer_state_dict.pt'))
            optimizer.load_state_dict(optim_state_dict)
            torch.save(optim_state_dict, os.path.join(destination, 'optimizer_state_dict_checkpoint.pt'))
            if schedule:
                scheduler_state_dict = torch.load(os.path.join(destination, 'scheduler_state_dict.pt'))
                scheduler.load_state_dict(scheduler_state_dict)
                torch.save(scheduler_state_dict, os.path.join(destination, 'scheduler_state_dict_checkpoint.pt'))
              
        # generate loss function with the configured hyperparameters
        loss_fn = utils.create_object_from_dict(config_dict, key = 'training/loss', wrapper_class = loss.Loss)

        to_validate = len(getattr(val_data, 'dataloader', [])) > 0
        metric_calcs = metrics.MetricsCalculator(config_dict,
                                                 neptune_run = run,
                                                 neptune_save_path = name + '/plots/',
                                                 validate = to_validate,
                                                 exp_name = name,
                                                 loss = loss_fn,
                                                 class_names = class_names
                                                 )

        early_stopping_dict = config_dict.get('metrics/early_stopping')
        if early_stopping_dict is not None:
            early_stop = utils.create_object_from_dict(early_stopping_dict,
                                                       wrapper_class = early_stopping.EarlyStoppingWrapper)
        else:
            early_stop = None
        
        # trains the model and logs the results
        train_model(nn_model, train_data, val_data if to_validate else None,
                    test_data, config_dict, metrics_and_loss = metric_calcs,
                    prediction_index = config_dict.get('training/prediction_index', 0),
                    optimizer = optimizer, scheduler = scheduler, early_stopping = early_stop,
                    virtual_batch_size = batch_size, true_batch_size = bs, run = run, name = name,
                    verbose = config_dict.get('meta/technical/verbose'), epoch_start = epoch_start,
                    grad_clip_value = config_dict.get('training/gradient_clipping/max_value'),
                    grad_clip_norm_type = config_dict.get('training/gradient_clipping/norm'))
        
        if tech_params['log to device']:
            # creates plots for the metric and saves them
            if tech_params['log metric and loss plots']:
                plot_destination = destination + 'plots/'
                if not os.path.isdir(plot_destination):
                    os.mkdir(plot_destination)
                extensions = tech_params.get_str_tuple('export_plots_as')
                plot_and_save_history(logs_path = destination + 'epoch_logs.csv',
                                      destination = destination,
                                      plot_destination = plot_destination, neptune_run = run,
                                      baselines = config_dict.get('metrics/baselines', {}),
                                      name = name, extensions = extensions, **kwargs)

            # uploads the model state dictionary to neptune
            if tech_params['log to neptune'] and config_dict.get('neptune run/log model', False):
                run['models/model_{}_state_dict'.format(idx)].upload(destination + 'model_state_dict.pt')

def plot_and_save_history(logs_path : str, destination : str, plot_destination : str,
                          neptune_run : Optional[neptune.Run],
                          name : str = '', baselines = {}, extensions = [], **kwargs):
    """
    Helper function that logs metric plots.

    Parameters:
        logs_path: path to a .csv file where the metrics are logged
        destination: path where the plots should be saved
        train_colour: colour of the train curves for the metrics
        val_colour: colour of the validation curves for the metrics
    """
    logs = pd.read_csv(logs_path)
    metrics = [column_name for column_name in logs.columns if column_name[:4] != 'val_']
    argmixes = {}
    for metric in metrics:
        metric_name = metric.split('/')[-1]
        baselines_for_metric = get_baselines_for_metric(metric_name, baselines)
        ys = [logs[metric].tolist()]
        if 'val_' + metric in logs.columns:
            scores = logs['val_' + metric].tolist()
            ys.append(scores)
            argmixes[metric] = get_argmixes(scores)
            labels = ['train', 'validation']
        else:
            labels = []
        plotter = utils.framework.plotters.GeneralPlotter(
                    dict(Ys = ys,
                         xlabel = 'epoch', ylabel = metric_name,
                         legend = {'labels': labels},
                         dirname = plot_destination,
                         fname = metric_name + '_plot',
                         baselines = baselines_for_metric),
                    neptune_run[name + '/plots']
                    )
        utils.export_plot(plotter, extensions)
        
        if 'learning_rate' == metric or 'lr_param_group' == metric[:len('lr_param_group')]:
            plotter.yscale = 'log'
            plotter.fname = metric_name + '_log_plot'
            utils.export_plot(plotter, [ext for ext in extensions if ext.lower().strip('.') != 'json'])
    csv_path = destination + 'best_scores.csv'
    pd.DataFrame(argmixes).to_csv(csv_path)
    neptune_run[name + '/best_indices'].upload(csv_path)

def get_argmixes(scores):
    if len(scores) == 0:
        return {}
    mix = 'max' if scores[0] <= scores[-1] else 'min'
    argmix = getattr(np, f'arg{mix}')
    output = {'best_index': argmix(scores) + 1}
    scores_arr = (-1)**(mix == 'min') * np.array(scores)
    output['soft_best_index'] = np.argmax(scores_arr >= scores[-1]) + 1
    return output

def get_baselines_for_metric(metric, all_baselines):
    try:
        baselines = all_baselines.get(metric, {})
        if isinstance(baselines, utils.config_dict.ConfigDict):
            baselines = baselines.trim().to_dict()
        
        if isinstance(baselines, dict):
            pass
        elif isinstance(baselines, Iterable):
            baselines = {f'baseline {i+1}': baseline for i, baseline in enumerate(baselines)}
        else:
            baselines = {'baseline': baselines}
    except Exception as e:
        msg = f'Exception occured when trying to calculate baseline for {metric}.'
        handle_exception(e, msg)
        baselines = {}
    finally:  
        return {'labels': list(baselines.keys()), 'values': list(baselines.values())}
        
    
def transform_img_to_tensor(img, *args, **kwargs):
    """Wrapper function that casts an image to torch.FloatTensor."""
    return torch.FloatTensor(img)

def transform_mask_to_tensor(mask, *args, **kwargs):
    """Wrapper function that casts a mask to torch.FloatTensor."""
    return torch.FloatTensor(mask)

def fill_dict_with_name_fields(config_dict : ConfigDict, name_fields = None):

    name_field_values = {}

    name_field_dicts = name_fields or config_dict.get('meta/technical/name_fields', ())
    if not isinstance(name_field_dicts, (list, tuple)):
        name_field_dicts = [name_field_dicts]

    for name_field_dict in name_field_dicts:
        if isinstance(name_field_dict, (ConfigDict, dict)):
            name_field = name_field_dict.key()
            key = name_field_dict.config_dict[name_field].get('keyword', name_field.split('/')[-1])
            has_default = 'default' in name_field_dict.value()
            if has_default:
                default = name_field_dict.value()['default']
        else:
            name_field = name_field_dict.replace(config_dict.SLASH_SUBSTITUTE, '/')
            key = name_field.split('/')[-1]
            has_default = False

        if name_field in config_dict:
            value = config_dict[name_field]
            
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            
            if isinstance(value, ConfigDict):
                try:
                    value = value.key()
                except ValueError:
                    continue
            if isinstance(value, str):
                value = value.split('.')[-1]
            
            name_field_values[key] = value
        elif has_default:
            name_field_values[key] = default
    
    added_tags = []
    for key, value in name_field_values.items():
        new_tag = f'{key}: {value}'
        added_tags.append(new_tag)

    if config_dict['meta/technical/log_to_neptune']:
        tags = list(config_dict.get('meta/neptune_run/tags', []))
        config_dict['meta/neptune_run/tags'] = list(set((*tags, *added_tags)))
    
    if config_dict['meta/technical/log_to_device']:
        exp_name = config_dict['meta/technical/experiment name'].rstrip('_')
        for key, value in name_field_values.items():
            suffix = f'{key}_{value}'
            if suffix not in exp_name:
                exp_name = exp_name + '_' + suffix
        config_dict['meta/technical/experiment name'] = exp_name
    
    return added_tags

def get_logs_from_path(experiment, name_fields = None, project = None):
    
    if isinstance(experiment, utils.config_dict.ConfigDict):
        experiment = experiment.key()
    
    experiment = str(experiment).replace(utils.config_dict.ConfigDict.SLASH_SUBSTITUTE, '/')
    
    if os.path.isdir(experiment):
        cd_path = experiment.rstrip('/') + '/config.yaml'
        if not os.path.isfile(cd_path):
            raise FileNotFoundError(f'Couldn\'t open logs from \'{cd_path}\': no such file.')
        config_dict = utils.config_dict.ConfigDict.from_yaml(cd_path)
    elif os.path.isdir('/'.join(experiment.split('/')[:-2])):
        raise FileNotFoundError(f'Couldn\'t open logs from \'{experiment}\': no such directory.')
    else: # load config dict from neptune run id
        elems = experiment.split('/')
        run_id = elems[-1]
        project_name = '/'.join(elems[:-1]) or project
        run = neptune.init_run(project_name, run = run_id)
        params = run.fetch()['parameters']
        run.stop()
        config_dict = utils.config_dict.ConfigDict(params)
    
    config_dict = config_dict.trim()
    tech_params = config_dict['meta/technical']
    
    log_data = {}
    
    if tech_params['log_to_device']:
        log_data = {'current_experiment': False,
                    'exp_name': tech_params['experiment_name'],
                    'save_path': tech_params['absolute_path'],
                    'num_trials': config_dict['experiment/number_of_trials'],
                    'log_to_neptune': tech_params['log_to_neptune'],
                    'tags': fill_dict_with_name_fields(config_dict, name_fields)}

        if tech_params['log_to_neptune']:
                log_data['neptune_project_name'] = config_dict['meta/neptune_run/project_name']
                log_data['neptune_run_id'] = config_dict['meta/neptune_run/run_id']
                
    return log_data

def compare_experiments(num_trials, save_path, neptune_run = None, extensions = []):
    if num_trials < 2:
        return

    if not os.path.isdir(save_path + 'variance_comparisons'):
        os.mkdir(save_path + 'variance_comparisons')
    
    labels = [f'run {k}' for k in range(1, num_trials + 1)]
    metric_logs = [pd.read_csv(save_path + f'run_{i + 1}/epoch_logs.csv') for i in range(num_trials)]
    val_column_names = []
    columns = metric_logs[0].columns
    for column in columns:
        if 'val_' == column[:4] and column not in val_column_names:
            if all(column in logs.columns for logs in metric_logs):
                val_column_names.append(column)
    
    statistics = {}

    for metric_name in val_column_names:
        values = []
        for metric_log in metric_logs:
            if metric_name in metric_log.columns:
                values.append(metric_log[metric_name].to_list())
            else:
                values.append([])
            
        axis_name = metric_name.split('/')[-1]
        if axis_name[:4] == 'val_':
            axis_name = axis_name[4:]
        
        ex_logs = max(values, key = len)
        if len(ex_logs) == 0:
            msg = f'None of the runs had logs for {axis_name}.'
            warnings.warn(msg)
            continue
        
        try:
            mix = 'max' if ex_logs[0] <= ex_logs[-1] else 'min'
            last_logs = [logs[-1] for logs in values if len(logs) > 0]
            best_logs = [getattr(np, mix)(logs) for logs in values if len(logs) > 0]
            statistics[axis_name] = {'mean_last_epoch': np.mean(last_logs),
                                     'median_last_epoch': np.median(last_logs),
                                     'std_last_epoch': np.std(last_logs),
                                    f'mean_{mix}_values': np.mean(best_logs),
                                    f'median_{mix}_values': np.median(best_logs),
                                    f'std_{mix}_values': np.std(best_logs)}
        except Exception as e:
            msg = f'Exception occured while trying to calculate variance statistics for {axis_name}.'
            handle_exception(e, msg)
        
        try:
            neptune_log_path = neptune_run and neptune_run['variance_comparisons/plots']
            plotter = utils.framework.plotters.GeneralPlotter(
                                        dict(Ys = values,
                                            xlabel = 'epoch',
                                            ylabel = axis_name,
                                            legend = {'labels': labels},
                                            dirname = save_path + 'variance_comparisons/',
                                            fname = f'{axis_name}_comparison'),
                                        neptune_experiment = neptune_log_path)
            utils.export_plot(plotter, extensions = extensions)
        except Exception as e:
            msg = f'Exception occured while trying to plot variance comparisons for {axis_name}.'
            handle_exception(e, msg)
    
    with open(save_path + 'variance_comparisons/statistics.json', 'w') as stats_file:
        json.dump(statistics, stats_file, indent = 3, sort_keys = True)
    
    if neptune_run is not None:
        neptune_run['variance_comparisons/statistics'] = statistics

def check_for_continued(modifiers : List[str], config_dict : utils.config_dict.ConfigDict):
    continued = '--continued' in modifiers
    run_start, epoch_start = 1, 0
    if continued:
        absolute_path = config_dict.get_str('meta/technical/absolute_path')
        num_epochs = config_dict['experiment/number_of_epochs']
        while os.path.isdir(os.path.join(absolute_path, f'run_{run_start+1}')):
            run_start += 1
        epoch_start = len(pd.read_csv(os.path.join(absolute_path, f'run_{run_start}', 'epoch_logs.csv')))
        if num_epochs == epoch_start:
            run_start += 1
            epoch_start = 0
    return continued, run_start, epoch_start