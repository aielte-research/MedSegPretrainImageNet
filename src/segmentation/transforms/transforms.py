import cv2
import numpy as np
import scipy.ndimage
import torch
from math import sqrt
import torchvision
import albumentations

import utils
from transform import TransformWrapper

class AlbumentationWrapper(TransformWrapper):
    """A wrapper object for `albumentation` transforms that should be transformed on the input x, and the mask and/or distance map if present."""
    def __init__(self, alb_transform, channels_first = False, **kwargs):
        """
        Arguments:
            `alb_transform`: constructor of the `albumentation` transform instance
            `kwargs`: keyword arguments that will be passed onto the `alb_transform` constructor
        """
        alb_transform_calc = alb_transform(**kwargs)
        def transform(x, mask = None, distance_map = None):
            has_mask = mask is not None
            has_dist = distance_map is not None
            mask_and_dist = has_mask and has_dist

            if mask_and_dist:
                y = np.concatenate((mask, distance_map), axis = 0)
            elif has_mask:
                y = mask
            elif has_dist:
                y = distance_map
            
            channel_dims = int(y.ndim > 2)
            if y.ndim == 2:
                y = np.expand_dims(y, 0)
            
            if not channels_first:
                x = np.moveaxis(x, 0, -1)
                y = np.moveaxis(y, 0, -1)

            if not (has_mask or has_dist):
                x = alb_transform_calc(image = x)['image']
                if not channels_first:
                    x = np.moveaxis(x, -1, 0)
                return {'x': x}
            
            transformed_pair = alb_transform_calc(image = x, mask = y)
            x = transformed_pair['image']
            if not channels_first:
                x = np.moveaxis(x, -1, 0)
            output = {'x': x}
            y = transformed_pair['mask']
            if channels_first:
                y_shape = y.shape[:-2] * channel_dims + y.shape[-2:]
            else:
                y_shape = y.shape[:2] + y.shape[2:] * channel_dims
            y = np.reshape(y, y_shape)
            if not channels_first and y.ndim > 2:
                y = np.moveaxis(y, -1, 0)

            if mask_and_dist:    
                mask, dist = y
                output.update({'mask': mask, 'distance_map': dist})
            
            elif has_mask:
                output.update({'mask': y})
            else:
                output.update({'distance_map': y})
            return output

        super().__init__(lambda *args, **kwargs: transform, data_to_transform = ('x', 'mask', 'distance_map'))



class RandomPick(object):
    """
    Callable object that samples from a larger image a random crop_size * crop_size square that is not necessarily parallel to the xy axes.

    Parameters:
        crop_size: int; the size of the image to crop; default: 256
        rotate_limit: int; maximum angle of rotation (in both directions) in degrees; default: 180
        seed: int; seed for the random number generator; optional

    __call__ method:
        input: image, mask; an image and its segmentation mask. The image and the mask are both expected to be an N * N square with shape (..., N, N)
        output: a D dictionary; D['image'] and D['mask'] are the transformed image and mask
    
    NOTE: for crop sizes too large relative to the original size, rotate_limit should be limited since not all rotation angles are possible, and the object's __init__ method does not check for that. (This is not a problem if the crop_size is at most half the image size.)

    NOTE: __call__ does not sample from a uniform distribution on all possible crops; rather, it samples the angle first, then the exact box to crop (both uniformly), resulting in a distribution that skews towards crops with angles close to multiples of 90 degrees.
    """

    def __init__(self, crop_size = 256, rotate_limit = 180, seed = None, *args, **kwargs):
        self.crop_size = crop_size
        self.rotate_limit = rotate_limit
        self.generator = np.random.default_rng(seed)
        
    def __call__(self, image, mask, *args, **kwargs):
        image, mask = torch.tensor(image), torch.tensor(mask)
        img_size = image.shape[1]
        angle = self.generator.integers(-self.rotate_limit, self.rotate_limit)
        x, y = self.calculate_cropping_coordinates(angle, img_size)
        aug_img = self.rotate_and_crop(image, angle, x, y)
        aug_mask = self.rotate_and_crop(mask, angle, x, y)
        aug_img, aug_mask = np.array(aug_img), np.array(aug_mask)
        return {'image': aug_img, 'mask': aug_mask}
        
    # sample the upper left point of the box
    def calculate_cropping_coordinates(self, angle, img_size):        
        alpha = np.deg2rad(angle % 90)
        sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)
        length = img_size - self.crop_size * (sin_alpha + cos_alpha)
        rotation_matrix = np.array([[sin_alpha, -cos_alpha], [cos_alpha, sin_alpha]])
        x_shift = (img_size - self.crop_size * cos_alpha) * cos_alpha
        y_shift = self.crop_size * sin_alpha * cos_alpha
        shift_array = np.array([x_shift, y_shift])
        x, y = rotation_matrix @ self.generator.integers(0, length, size = 2) + shift_array
        x, y = int(x), int(y)
        return x, y
    
    def rotate_and_crop(self, img, angle, left, top):
        rotated_img = torchvision.transforms.functional.rotate(img, int(angle), expand = True)
        cropped_img = torchvision.transforms.functional.crop(rotated_img, top, left, self.crop_size, self.crop_size)
        return cropped_img

class Partition(object):
    def __init__(self, partition_count = 4):
        if isinstance(partition_count, int):
            partition_count = (int(sqrt(partition_count)),) * 2
        self.partition_counts = partition_count
        self.num_partitions = np.product(partition_count)
    
    def __call__(self, x, _index, mask = None, distance_map = None):
        image_size = x.shape[1:]
        crop_idx = _index % self.num_partitions
        
        l = crop_idx % self.partition_counts[0]
        t = crop_idx // self.partition_counts[0]

        crop_size = (image_size[0] // self.partition_counts[0], image_size[1] // self.partition_counts[1])
        left = image_size[0] - crop_size[0] if (l + 1) * crop_size[0] > image_size[0] else l * crop_size[0]
        top = image_size[1] - crop_size[1] if (t + 1) * crop_size[1] > image_size[1] else t * crop_size[1]

        crop_slice = (slice(left, left + crop_size[0]), slice(top, top + crop_size[1]))
        x = x[(slice(None), *crop_slice)]
        output = {'x': x}

        if mask is not None:
            if mask.ndim == 2:
                output['mask'] = mask[crop_slice]
            else:
                output['mask'] = mask[(slice(None), *crop_slice)]
        
        if distance_map is not None:
            output['distance_map'] = distance_map[crop_slice]

        return output

class DiscreteOrRandomRot(TransformWrapper):

    PARAMS = {
        'rotation_limit': 20,
        'border_mode': 'BORDER_CONSTANT',
        'discrete_rotation_probability': 0.5,
        'continuous_rotation_probability': 0.5
    }

    PADDING_VALUE = 0
    MASK_PADDING_VALUE = 0
    INTERPOLATION = cv2.INTER_LINEAR

    @staticmethod
    def fill_kwargs(config_dict):
        border_mode = config_dict['border_mode']
        if isinstance(border_mode, utils.config_dict.ConfigDict):
            border_mode = border_mode.key()
        if border_mode in ('BORDER_CONSTANT', cv2.BORDER_CONSTANT):
            config_dict.get_or_update('padding_value', DiscreteOrRandomRot.PADDING_VALUE)
            config_dict.get_or_update('mask_padding_value', DiscreteOrRandomRot.MASK_PADDING_VALUE)

    def __init__(self, config_dict, seed = None, *args, **kwargs):
        self.rng = np.random.default_rng(seed)
        for param, default_value in self.PARAMS.items():
            setattr(self, param, config_dict.trim().get(param, default_value))
        if isinstance(self.border_mode, str):
            self.border_mode = getattr(cv2, self.border_mode, self.border_mode)
        if self.border_mode in ('BORDER_CONSTANT', cv2.BORDER_CONSTANT):
            self.value = config_dict.get('padding_value', self.PADDING_VALUE)
            self.mask_value = config_dict.get('mask_padding_value', self.MASK_PADDING_VALUE)
        interpolation = config_dict.get('interpolation_method', self.INTERPOLATION)
        if isinstance(interpolation, str):
            interpolation = getattr(cv2, interpolation, interpolation)
        
        self.cont_rotate = albumentations.augmentations.geometric.Rotate(
                                                                limit = self.rotation_limit,
                                                                interpolation = interpolation,
                                                                border_mode = self.border_mode,
                                                                value = self.value,
                                                                mask_value = self.mask_value,
                                                                p = 1)
        self.disc_rotate = albumentations.augmentations.geometric.rotate.RandomRotate90(p = 1)
    
    def __call__(self, **input):
        img, mask = input['x'], input['mask']
        has_colors = len(img.shape) > 2
        padded_mask = len(mask.shape) > 2

        if self.rng.binomial(n = 1, p = self.discrete_rotation_probability):
            if has_colors:
                img = np.moveaxis(img, 0, -1)
            if padded_mask:
                mask = np.moveaxis(mask, 0, -1)
            trsfd = self.disc_rotate(image = img, mask = mask)
            img, mask = trsfd['image'], trsfd['mask']
            if has_colors:
                img = np.moveaxis(img, -1, 0)
            if padded_mask:
                mask = np.moveaxis(mask, -1, 0)
            img, mask = img[..., ::-1].copy(), mask[..., ::-1].copy()
        elif self.rng.binomial(n = 1, p = self.continuous_rotation_probability):
            if has_colors:
                img = np.moveaxis(img, 0, -1)
            if padded_mask:
                mask = np.moveaxis(mask, 0, -1)
            trsfd = self.cont_rotate(image = img, mask = mask)
            img, mask = trsfd['image'], trsfd['mask']
            if has_colors:
                img = np.moveaxis(img, -1, 0)
            if padded_mask:
                mask = np.moveaxis(mask, -1, 0)
        input['x'] = img
        input['mask'] = mask
        return input

class Resize(object):
    def __init__(self, size = 224, *args, **kwargs):
        if isinstance(size, int):
            size = (size, size)
        self.size = np.array(size)
    
    def __call__(self, image, mask, *args, **kwargs):
        zoom_value =  tuple(self.size / image.shape[:2])
        img_zoom_value =  zoom_value + (1,) * (len(image.shape) - 2)
        mask_zoom_value = zoom_value + (1,) * (len(mask.shape) - 2)
        
        img = scipy.ndimage.zoom(image, img_zoom_value)
        mask = scipy.ndimage.zoom(mask, mask_zoom_value, order = 0)
        return {'image': img, 'mask': mask}
