import csv
import itertools
import math
from typing import Dict, Iterable, Literal, Optional, Union
import metrics

from tqdm import tqdm

import os
import torch
from optim.scheduler import SchedulerWrapper

from utils.config_dict import ConfigDict
from exception_handling import handle_exception

def predict(model : torch.nn.Module, ds : Iterable[Dict[str, torch.Tensor]],
            metrics_and_loss : metrics.MetricsCalculator,
            optimizer : Optional[torch.optim.Optimizer] = None,
            scheduler : Optional[SchedulerWrapper] = None,
            accumulation_scale : int = 1, pred_idx : int = 0,
            device : str = 'cuda' if torch.cuda.is_available() else 'cpu',
            train : bool = True, log_to_device : bool = True, destination : str = None,
            last : bool = False, learning_rate_keywords = [],
            grad_clip_value : Optional[float] = torch.inf,
            grad_clip_norm_type : Union[float, Literal['inf']] = 'inf',
            *args, **kwargs):
    """
    Helper function performing one epoch.

    Arguments:
        model: an instance of an object subclassing torch.nn.Module
        ds: iterable dataset object that loads batches
        metrics_and_loss: dictionary containing the metrics and loss that should be calculated
        accumulation_scale: number of iteration within one batch; used for gradient accumulation
        device: name of device containing the model
        optimzier: instance of an object subclassing torch.optim.Optimizer, that is configured with the model's parameters
        scheduler: optional instance of an object subclassing optim.SchedulerWrapper, that is configured with the optimizer's parameters; should be not None only if it updates in a batchwise manner
        train: bool; whether the epoch is a train loop; if set to False, batches will not be logged, and no optimizer step will be performed
        log_to_device: bool; whether to log the metrics and loss calculated on each batch to the device
        destination: path where the metrics should be logged if log_to_device is True
        log_to_neptune: bool; whether to log the metrics and loss calculated on each batch to neptune
        run: object for neptune runif log_to_neptune is true
        name: name of the current run
    """

    if train and not last:
        model.train()
    else:
        model.eval()

    for i, batch in enumerate(ds):

        try:
            # step: whether an optimizer step should be performed
            step = (i + 1) % accumulation_scale == 0 or i == len(ds) - 1

            if train and step and not last:
                optimizer.zero_grad()

            batch = {input_type: input.to(device) for input_type, input in batch.items()}

            if train and not last:
                pred = model(**batch)
                if isinstance(pred, (list, tuple)):
                    batch['predictions'] = pred
                    batch['prediction'] = pred[pred_idx]
                else:
                    batch['prediction'] = pred
            else:
                with torch.no_grad():
                    pred = model(**batch)
                    if isinstance(pred, (list, tuple)):
                        batch['predictions'] = pred
                        batch['prediction'] = pred[pred_idx]
                    else:
                        batch['prediction'] = pred

            metric_value_dict = metrics_and_loss.calculate_batch(batch,
                                                                 train = train,
                                                                 accumulation_scale = accumulation_scale,
                                                                 last = last)
            if step:
                metric_value_dict = metrics_and_loss.evaluate_batch(batch,
                                                                    train = train,
                                                                    accumulation_scale = accumulation_scale, last = last)
            
            # if it is the end of a batch, update metric_value_dict with the values calculated over the batch
            # and perform a step with the optimzier
            if train and step and not last:
                metric_value_dict.update({lr_kw: param_group['lr']
                                          for lr_kw, param_group
                                          in zip(learning_rate_keywords, optimizer.param_groups)})
                if grad_clip_value is not None:
                    try:
                        grad_magnitude = torch.nn.utils.clip_grad.clip_grad_norm_(parameters = model.parameters(),
                                                                                  max_norm = grad_clip_value,
                                                                                  norm_type = grad_clip_norm_type,
                                                                                  error_if_nonfinite = False)
                        try:
                            metric_value_dict['gradient_magnitude'] = grad_magnitude.item()
                        except Exception as e:
                            msg = 'Exception occurred while trying to log gradient magnitude.'
                            handle_exception(e, msg)
                    except Exception as e:
                        msg = 'Exception occurred while trying to clip gradients.'
                        handle_exception(e, msg)
                optimizer.step()
                if scheduler is not None:
                    # scheduler is not None only if it updates batchwise
                    scheduler.step()
                
                # log the metrics to the device
                if log_to_device:
                    write_header = not os.path.isfile(destination)

                    with open(destination, 'a') as file:
                        writer = csv.DictWriter(file, fieldnames = metric_value_dict.keys())
                        if write_header:
                            writer.writeheader()
                        writer.writerow(metric_value_dict)
        
        except Exception as e:
            if accumulation_scale == 1:
                handle_exception(e, f'Exception occured in batch {i}.')
            else:
                batch_num = i // accumulation_scale
                batch_fragment_num = i % accumulation_scale
                handle_exception(
                    e, f'Exception occured in batch {batch_num} in batch fragment {batch_fragment_num}.'
                    )

def train_model(model : torch.nn.Module,
                train_data : Iterable[Dict[str, torch.Tensor]],
                val_data : Iterable[Dict[str, torch.Tensor]],
                test_data : Optional[Iterable[Dict[str, torch.Tensor]]],
                config_dict : ConfigDict,
                optimizer : torch.optim.Optimizer,
                virtual_batch_size : int = 32,
                true_batch_size : int = 1,
                metrics_and_loss : metrics.MetricsCalculator = None,
                name : Optional[str]= None,
                scheduler : Optional[SchedulerWrapper]= None,
                verbose : bool = True,
                prediction_index : int = 0,
                epoch_start : int = 0,
                grad_clip_value : Optional[float] = torch.inf,
                grad_clip_norm_type : Union[float, Literal['inf']] = 'inf',
                *args, **kwargs):
    """
    Function that trains a model for a certain number of epochs.

    Arguments:
        model: instance of an object subclassing torch.nn.Module
        train_data: dataloader-type object containing the train dataset
        val_data: optional dataloader-type object containing the validation dataset; if None, no validation loop will be calculated
        virtual_batch_size: number of datapoints the model has to see before it performs an optimizer step during training
        true_batch_size: number of datapoints loaded into memory; should be used when virtual_batch_size is too large for memory; should be a divisor of virtual_batch_size
        optimzier: instance of an object subclassing torch.optim.Optimizer, that is configured with the model's parameters
        loss: object calculating the loss function; should be wrapped by losses.Loss
        metrics: object that calculates the metrics
        scheduler: learning rate scheduler object instance
        run: object for neptune runif log_to_neptune is true
        name: name of the current run
    """
    assert virtual_batch_size % true_batch_size == 0, 'Virtual batch size ({}) should divide true batch size ({})'.format(
        virtual_batch_size, true_batch_size
        )
    
    gradient_accumulation_scale = virtual_batch_size // true_batch_size
    if grad_clip_value is None:
        grad_clip_value = torch.inf

    tech_params : ConfigDict = config_dict['meta/technical']
    destination : str = tech_params.get('absolute path', '') + name + '/'
    num_epochs : int = config_dict['experiment/number of epochs']
    log_to_device : bool = tech_params['log to device']
    
    log_batch_to_device = log_to_device and log_to_device != 'epoch'
    
    log_last_model = tech_params['log_last_model']
    log_best_model = tech_params['log_best_model']
    checkpoints = tech_params.get_tuple('model_log_checkpoints', [])
    
    if log_best_model:
        model_eval_dict = tech_params.get('model_evaluation', {})
        watched_metric = model_eval_dict.get_str('metric')
        eval_mode = model_eval_dict.get_str('mode')
        mix = min if eval_mode in ('min', 'minimum') else max
        best_value = (-1)**(mix == max) * math.inf
        

    parallel_train = torch.cuda.device_count() > 1
    if parallel_train: # TODO: don't train on all available device
        model = torch.nn.parallel.DataParallel(model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    
    lr_kws = ['learning_rate'] if len(optimizer.param_groups) == 1 else [f'lr_param_group_{i+1}' for i in range(len(optimizer.param_groups))]
    
    counter = range(epoch_start, num_epochs) if num_epochs is not None else itertools.count(epoch_start)
    if verbose:
        counter = tqdm(counter, desc = 'Training model', unit = 'epoch')
    for i in counter:
        try:
            
            metric_value_dict = {lr_kw: param_group['lr'] for lr_kw, param_group in zip(lr_kws, optimizer.param_groups)}
            
            # run train loop
            predict(model, train_data,
                    device = device,
                    accumulation_scale = gradient_accumulation_scale,
                    pred_idx = prediction_index,
                    metrics_and_loss = metrics_and_loss,
                    optimizer = optimizer,
                    scheduler = scheduler if getattr(scheduler, 'batch_update', False) else None,
                    log_to_device = log_batch_to_device,
                    destination = destination + 'batch_logs.csv',
                    learning_rate_keywords = lr_kws,
                    grad_clip_value = grad_clip_value,
                    grad_clip_norm_type = grad_clip_norm_type)
            
            metric_value_dict.update(metrics_and_loss.evaluate_epoch())
            
            if val_data:
                # run validation loop
                predict(model, val_data,
                        device = device,
                        accumulation_scale = gradient_accumulation_scale,
                        pred_idx = prediction_index,
                        metrics_and_loss = metrics_and_loss,
                        log_to_device = False,
                        log_to_neptune = False,
                        train = False)
                
                metric_value_dict.update(
                                {'val_' + k: v for k, v in metrics_and_loss.evaluate_epoch().items()}
                )
            
            # log metric values to the device
            if log_to_device:
                dest = destination + 'epoch_logs.csv'
                write_header = not os.path.isfile(dest)
                with open(dest, 'a') as file:
                    writer = csv.DictWriter(file, fieldnames = metric_value_dict.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metric_value_dict)

            # change the learning rate according to the scheduler
            if scheduler is not None and scheduler.epoch_update:
                scheduler.step()
            
            # log the model weights to a state dictionary
            if log_to_device: # TODO: ezt kezeld ha multi_GPU futtatás van
                model_state_dict = model.state_dict() if not parallel_train else model.module.state_dict()
                if log_last_model:
                    torch.save(model_state_dict, destination + 'last_model_state_dict.pt')
                if log_best_model:
                    curr_value = metric_value_dict[watched_metric]
                    if mix(best_value, curr_value) == curr_value:
                        torch.save(model_state_dict, destination + 'best_model_state_dict.pt')
                j = i + 1
                if j in checkpoints:
                    torch.save(model_state_dict, destination + f'model_state_dict_epoch_{j}.pt')
                torch.save(optimizer.state_dict(), destination + 'optimizer_state_dict.pt')
                if scheduler is not None:
                    torch.save(scheduler.state_dict(), destination + 'scheduler_state_dict.pt')

        except Exception as e:
            msg = f'Exception occured in epoch {i}.'
            handle_exception(e, msg)

    if metrics_and_loss.requires_last_pass:
        predict(model, train_data,
                metrics_and_loss = metrics_and_loss,
                accumulation_scale = gradient_accumulation_scale,
                device = device,
                train = True,
                log_to_device = False,
                log_to_neptune = False,
                last = True)
        metrics_and_loss.evaluate_epoch(last = True)
        predict(model, val_data,
                metrics_and_loss = metrics_and_loss,
                accumulation_scale = gradient_accumulation_scale,
                device = device,
                train = False,
                log_to_device = False,
                log_to_neptune = False,
                last = True)
        metrics_and_loss.evaluate_epoch(last = True)
    
    metrics_and_loss.evaluate_at_end()
    if test_data:
        try:
            # run test loop
            predict(model, test_data,
                    device = device,
                    accumulation_scale = gradient_accumulation_scale,
                    pred_idx = prediction_index,
                    metrics_and_loss = metrics_and_loss,
                    log_to_device = False,
                    log_to_neptune = False,
                    train = False)
            
            metric_value_dict = metrics_and_loss.evaluate_epoch()
            
            if log_to_device:
                dest = destination + 'test_logs.csv'
                write_header = not os.path.isfile(dest)
                with open(dest, 'a') as file:
                    writer = csv.DictWriter(file, fieldnames = metric_value_dict.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metric_value_dict)
                    
        except Exception as e:
            msg = 'Exception occured while trying to evaluate the test data.'
            handle_exception(e, msg)
