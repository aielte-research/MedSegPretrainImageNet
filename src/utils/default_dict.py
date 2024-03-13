from numpy.random import default_rng

default_dict = {'model': 'segmentation.models.UNet',
                'meta/technical/experiment_name': 'Experiment',
                'meta/technical/save_destination': '../logs/',
                'meta/technical/seed': int(default_rng().integers(1000000)),
                'experiment/number_of_epochs': 150,
                'experiment/number_of_trials': 1,
                'training/optimizer': 'sgd',
                'training/loss': 'torch.nn.BCELoss',
                'metrics/metrics':  ('metrics.DiceIndex',),
                'data/transforms': {'train': ('segmentation.transforms.wrapped_transforms.RandomRotation',
                                              'segmentation.transforms.wrapped_transforms.CenterCrop'),
                                    'val': tuple()},
                'data/data':  'segmentation.datasets.ACDC',
                'meta/technical/log_to_device': True,
                'meta/technical/number_of_data_loader_workers': 0,
                'meta/technical/log_metric_and_loss_plots': False,
                'meta/technical/maximum_actual_batch_size': 24,
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