import os
import random
import sys
import time
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd

import utils
from exception_handling import handle_exception
from run_experiment import experiment, get_logs_from_path

from utils import config_dict, config_parser

def main():
    args = iter(sys.argv[1:])
    arg = next(args)
    while arg:
        modifiers = []
        next_possible_arg = next(args, '')
        while next_possible_arg.startswith('-'):
            modifiers.append(next_possible_arg)
            next_possible_arg = next(args, '')
        run_experiment_from_dict(arg, modifiers)
        arg = next_possible_arg

# runs experiment with hyperparameters and other variables specified in a python dict
def run_experiment_from_dict(file_path, modifiers = []):
    config_dicts, original = config_parser.parse(file_path)
    logs = get_comparisons(config_dict.ConfigDict(original))
    series_id = utils.base64_str(int(time.time() * 1e8)) + utils.base64_str(random.randint(0, 64**7 - 1), num_digits = 8)
    for i, cd in enumerate(config_dicts):
        try:
            logs.append(experiment(config_dict.ConfigDict(cd), original = original, series_id = series_id, modifiers = modifiers))
        except Exception as e:
            msg = f'Exception occured while trying to run experiment {i + 1} of file {file_path}.'
            handle_exception(e, msg)
        try:
            compare_experiments(logs)
        except Exception as e:
            msg = 'Exception occured while trying to plot comparisons between experiments.'
            handle_exception(e, msg)

def compare_experiments(logs, extensions = ['html', 'json']):
    logs = list(filter(lambda log: log != {}, logs))
    num_exps = len(logs)
    if num_exps < 2:
        return
    logs_dict : Dict[str, list] = {k: [] for k in logs[0].keys()}
    for i, log in enumerate(logs):
        for k, v in log.items():
            if k in logs_dict:
                logs_dict[k] = [*logs_dict[k], v]
            else:
                logs_dict[k] = [*[None for _ in range(i - 1)], v]

    num_trials = min(logs_dict.get('num_trials', [1]))
    for k in range(1, num_trials + 1):
            
        for save_path in logs_dict['save_path']:
            dest_dir = save_path + f'run_{k}/comparisons/'
            if not os.path.isdir(dest_dir):
                os.mkdir(dest_dir)

        metric_logs = [pd.read_csv(logs_dict['save_path'][i] + f'run_{k}/epoch_logs.csv')
                       for i in range(num_exps)]
        val_column_names : List[str] = []
        columns = metric_logs[0].columns
        for column in columns:
            if 'val' in column and column not in val_column_names:
                if all(column in logs.columns for logs in metric_logs):
                    val_column_names.append(column)
        
        labels : List[str] = []
        for i in range(num_exps):
            label = '; '.join(logs_dict['tags'][i])
            neptune_run = logs_dict['neptune_run_id'][i]
            if neptune_run is not None:
                label = f'{label} ({neptune_run})'
            labels.append(label)

        best_values : Dict[str, Dict[str, float]] = {}
        last_values : Dict[str, Dict[str, float]] = {}
        
        for metric_name in val_column_names:
            values : List[List[float]] = []
            for metric_log in metric_logs:
                if metric_name in metric_log.columns:
                    values.append(metric_log[metric_name].to_list())
                else:
                    values.append([])
            
            axis_name = metric_name.split('/')[-1]
            if axis_name[:4] == 'val_':
                axis_name = axis_name[4:]
                
            last_values[axis_name] = {label: value_list[-1] if len(value_list) > 0 else np.nan  for label, value_list in zip(labels, values)}
            mixes = [value_list[0] <= value_list[-1] if len(value_list) > 0 else np.nan for value_list in values]
            is_max = np.nanmean(mixes) >= 0.5
            mix = max if is_max else min
            mix_name = 'max' if is_max else 'min'
            best_values[mix_name + '_' + axis_name] = {label: mix(value_list) if len(value_list) > 0 else np.nan for label, value_list in zip(labels, values)}
            
            for i in range(num_exps):
                
                plotter = utils.framework.plotters.GeneralPlotter(
                                                        dict(Ys = values,
                                                             xlabel = 'epoch',
                                                             ylabel = axis_name,
                                                             legend = {'labels': labels},
                                                             dirname = logs_dict['save_path'][i] + f'/run_{k}/comparisons',
                                                             fname = f'{axis_name}_comparison'))
                utils.export_plot(plotter, extensions)
                if logs_dict['log_to_neptune'][i]:
                    neptune_run.stop()
        
        bests_df = pd.DataFrame(best_values)
        lasts_df = pd.DataFrame(last_values)
        for i in range(num_exps):
            if not logs_dict['current_experiment'][i]:
                continue
            neptune_log_path = None
            bests_dest = logs_dict['save_path'][i] + f'/run_{k}/comparisons/best_values_comparison.csv'
            lasts_dest = logs_dict['save_path'][i] + f'/run_{k}/comparisons/last_values_comparison.csv'
            bests_df.to_csv(bests_dest)
            lasts_df.to_csv(lasts_dest)
                

def get_comparisons(cd : config_dict.ConfigDict):
    name_fields = cd.get('meta/technical/name_fields', [])
    project_name = cd.get('meta/neptune_run/project_name', utils.neptune_dict['meta/neptune run/project name']['default'])
    
    comparisons = []
    for path in cd.elements_of('meta/technical/compare_to'):
        try:
            comparisons.append(get_logs_from_path(path, name_fields, project_name))
        except Exception as e:
            if isinstance(path, config_dict.ConfigDict):
                path = path.key()
            path = path.replace(config_dict.ConfigDict.SLASH_SUBSTITUTE, '/')
            msg = f'An excpetion occured trying to load logs from experiment {path}. Comparisons to that experiment will not be logged.'
            if isinstance(e, FileNotFoundError):
                warnings.warn(msg + f'\n{e}')
            else:
                handle_exception(e, msg)
            
    return comparisons

if __name__ == '__main__':
    main()