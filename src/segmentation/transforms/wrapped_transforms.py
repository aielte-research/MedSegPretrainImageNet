import cv2
from . import transforms
import albumentations

import utils

class RandomPick(transforms.AlbumentationWrapper):

    PARAMS = {
        'crop size': {
                'argument name': 'crop_size',
                'default': 256
                },
        'rotation limit': {
                'argument name': 'rotate_limit',
                'default': 180
                }
            }
    
    def __init__(self, config_dict, **kwargs):
        kwargs.update(
            {param_dict['argument name']: config_dict.get(param_name, param_dict['default'])
            for param_name, param_dict in self.PARAMS.items()}
            )
        return super().__init__(transforms.RandomPick, channels_first = True, **kwargs)

class RandomHorizontalFlip(transforms.AlbumentationWrapper):

    PARAMS = {}

    def __init__(self, *args, **kwargs):
        return super().__init__(albumentations.augmentations.HorizontalFlip)

class Partition(transforms.TransformWrapper):

    PARAMS = {
        'number of partitions': {
                'argument name': 'partition_count',
                'default': 4
                }
            }
    
    def __init__(self, config_dict, **kwargs):
        partition_count = config_dict.get('number of partitions', self.PARAMS['number of partitions']['default'])
        return super().__init__(transforms.Partition, partition_count = partition_count,
                                data_to_transform = ('x', 'mask', 'distance_map', '_index'))

class RandomRotation(transforms.AlbumentationWrapper):

    PARAMS = {
        'limit': 180,
        'border_mode': 'BORDER_CONSTANT'
    }

    PADDING_VALUE = 0
    MASK_PADDING_VALUE = 0

    @staticmethod
    def fill_kwargs(config_dict):
        border_mode = config_dict['border_mode']
        if isinstance(border_mode, utils.config_dict.ConfigDict):
            border_mode = border_mode.key()
        if border_mode in ('BORDER_CONSTANT', cv2.BORDER_CONSTANT):
            config_dict.get_or_update('value', RandomRotation.PADDING_VALUE)
            config_dict.get_or_update('mask_value', RandomRotation.MASK_PADDING_VALUE)

    def __init__(self, config_dict, **kwargs):
        alb_kwargs = utils.get_kwargs(albumentations.augmentations.geometric.Rotate, config_dict)
        
        border_mode = config_dict.trim()['border_mode']
        if isinstance(border_mode, str):
            border_mode = getattr(cv2, border_mode)
        alb_kwargs['border_mode'] = border_mode

        if 'interpolation' in alb_kwargs:
            interpolation = alb_kwargs['interpolation']
            if isinstance(interpolation, str):
                alb_kwargs['interpolation'] = getattr(cv2, interpolation)

        alb_kwargs['p'] = alb_kwargs.get('p') or 1
        
        super().__init__(albumentations.augmentations.geometric.Rotate, **alb_kwargs)

class RandomFlip(transforms.AlbumentationWrapper):

    PARAMS = {'horizontal': False, 'vertical': True, 'probability': 0.5}

    def __init__(self, config_dict, **kwargs):
        p = config_dict.get('probability', config_dict.get('p')) or 1
        horizontal, vertical = config_dict['horizontal'], config_dict['vertical']
        assert horizontal or vertical, 'At least one of horizontal or vertical flips should be allowed for random flip augmentation.'

        if not horizontal and not vertical:
            alb_tr = lambda *args, **kwargs: lambda _: _
        elif not horizontal:
            alb_tr = albumentations.augmentations.VerticalFlip
        elif not vertical:
            alb_tr = albumentations.augmentations.HorizontalFlip
        else:
            alb_tr = albumentations.augmentations.Flip
        
        super().__init__(alb_tr, p = p)

class CenterCrop(transforms.AlbumentationWrapper):

    PARAMS = {'size': 128}

    def __init__(self, config_dict, **kwargs):
        size = config_dict['size']
        if isinstance(size, int):
            size = (size, size)
        size_kwargs = dict(zip(('height', 'width'), size))
        p = config_dict.get('probability', config_dict.get('p')) or 1

        super().__init__(albumentations.augmentations.crops.transforms.CenterCrop, p = p, **size_kwargs)

class Resize(transforms.AlbumentationWrapper):

    PARAMS = {'size': 224}

    def __init__(self, config_dict, **kwargs):
        size = config_dict['size']
        super().__init__(transforms.Resize, size = size)