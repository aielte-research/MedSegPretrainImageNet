import glob
import itertools
import os
import re
import cv2
import numpy as np
import scipy.ndimage

import nibabel as nib

import cv2

import socket

from PIL import Image

import data
import utils

class OKPIDataset(data.Dataset):
    PARAMS = {
        'val split percentage': 0.2,
        'load masks': True,
        'load distance maps': False
        }
    
    # path to data (arrays in .npy format)
    PATH = '/home/bakosbence237/Pat_segm/preprocessed_data/Train and val/'

    # prepare data to be loaded
    # (functions are created instead of loading data to decrease memory consumption)
    # nonzero masks values are manually cast to 1
    # necessary because all the train and validation regions of interest are marked as 2
    @staticmethod
    def load_imgs():
        return np.load(OKPIDataset.PATH + 'imgs.npy')
    
    @staticmethod
    def load_masks():
        return np.where(np.load(OKPIDataset.PATH + 'masks.npy') != 0, 1, 0)
    
    @staticmethod
    def load_distances():
        return np.load(OKPIDataset.PATH + 'distances.npy')

    @staticmethod
    def load_sources():
        return np.load(OKPIDataset.PATH + 'sources.npy')
    
    def __init__(self, ds_dict, seed = None, *args, **kwargs):

        return_masks = ds_dict['load masks']
        return_distances = ds_dict['load distance maps']
        split = ds_dict['val split percentage']
        

        imgs = self.load_imgs()
        imgs = np.moveaxis(imgs, -1, 1) / 255
        N = len(imgs)

        rnd = np.random.default_rng(ds_dict.get('seed') or seed)
        rnd_arr = np.arange(N)
        rnd.shuffle(rnd_arr)
        
        val_length = int(split * N)
        train_idcs = rnd_arr >= val_length
        val_idcs = ~train_idcs

        train_imgs = imgs[train_idcs]
        val_imgs = imgs[val_idcs]

        sources = self.load_sources()
        train_sources = sources[train_idcs]
        val_sources = sources[val_idcs]

        train_pos_idcs = train_sources[:,1] == '1'
        val_pos_idcs = val_sources[:,1] == '1'

        train_data = {'x': train_imgs, 'positives': train_pos_idcs}
        val_data = {'x': val_imgs, 'positives': val_pos_idcs}
        
        if return_masks:
            masks = self.load_masks()
            masks = masks.reshape((masks.shape[0], 1, *masks.shape[1:]))
            train_masks = masks[train_idcs]
            val_masks = masks[val_idcs]
            train_data['mask'] = train_masks
            val_data['mask'] = val_masks
        
        if return_distances:
            distances = self.load_distances()
            distances = distances.reshape((distances.shape[0], 1, *distances.shape[1:]))
            train_distances = distances[train_idcs]
            val_distances = distances[val_idcs]
            train_data['distance_map'] = train_distances
            val_data['distance_map'] = val_distances

        self.train, self.val = train_data, val_data


class COVIDQUDataset(data.Dataset):
    PARAMS = {
        'val split percentage': {
            'argument name': 'val_split_percentage',
            'default': 0.2
            },
        'load masks': {
            'argument name': 'return_masks',
            'default': True
            },
        'load distance maps': {
            'argument name': 'return_distances',
            'default': False
            },
        'lung or infection': {
            'argument name': 'task',
            'default': 'lung'      # options: lung, inf
            },
        'validate on test': {
            'argument name': 'validate_on_test',
            'default': False
            },
        'image sizes': {
            'argument name': 'image_sizes',
            'default': 256
            }
        }
    
    # path to data (arrays in .npy format)
    if socket.gethostname() != 'a100.cs.elte.hu':
        PATH = '/data/bakosbence237/datasets/COVID_QU'
    else:
        PATH = '/projects/SegmentationPretraining/data/COVID_QU'

    # prepare data to be loaded
    # (functions are created instead of loading data to decrease memory consumption)
    # nonzero masks values are manually cast to 1
    # necessary because all the train and validation regions of interest are marked as 2
    @staticmethod
    def load_imgs(task, partition):
        return np.load(f'{COVIDQUDataset.PATH}/{task}/{partition}/imgs.npy')
    
    @staticmethod
    def load_masks(task, segmentation_task, partition):
        mask_file = 'masks' if segmentation_task == 'lung' else 'inf_masks'
        return np.where(np.load(f'{COVIDQUDataset.PATH}/{task}/{partition}/{mask_file}.npy') != 0, 1, 0).astype(np.float)
    
    @staticmethod
    def load_distances(task):
        return np.load(COVIDQUDataset.PATH + task + '/train and val/distances.npy')

    @staticmethod
    def load_sources(task):
        return np.load(COVIDQUDataset.PATH +  task + '/train and val/sources.npy')
    
    def __init__(self, ds_dict, seed = None, *args, **kwargs):

        return_masks = ds_dict['load masks']
        return_distances = ds_dict['load distance maps']
        split = ds_dict['val split percentage']
        segmentation_task = ds_dict['lung or infection']
        validate_on_test = ds_dict['validate_on_test']
        size = ds_dict['image sizes']
        
        if segmentation_task =='lung':
            task_path = 'Lung_segm/data arrays'
        elif segmentation_task =='inf':
            task_path = 'Inf_segm/data arrays/COVID-19'
        else:
            raise ValueError(f'Task must be \'lung\' or \'infection\', not {segmentation_task}.')
        
        imgs = self.load_imgs(task_path, 'train and val')
        N = len(imgs)
        
        if validate_on_test:
            imgs = np.concatenate((imgs, self.load_imgs(task_path, 'test')), axis=0)
        
        imgs = imgs.squeeze() / 255
        
        if isinstance(size, int):
            size = (size, size)
        imgs = np.array([cv2.resize(img, size) for img in imgs])
        imgs = np.expand_dims(imgs, 1)
        
        if validate_on_test:
            train_idcs = np.concatenate((np.full(N, True), np.full(len(imgs)-N, False)), axis=0)
            val_idcs = ~train_idcs
        else:
            rnd = np.random.default_rng(ds_dict.get('seed') or seed)
            rnd_arr = np.arange(len(imgs))
            rnd.shuffle(rnd_arr)
            val_length = int(split * N)
            train_idcs = rnd_arr >= val_length
            val_idcs = ~train_idcs

        train_imgs = imgs[train_idcs]
        val_imgs = imgs[val_idcs]

        # sources = self.load_sources()
        # train_sources = sources[train_idcs]
        # val_sources = sources[val_idcs]

        # train_pos_idcs = train_sources[:,1] == '1'
        # val_pos_idcs = val_sources[:,1] == '1'

        train_data = {'x': train_imgs}
        val_data = {'x': val_imgs}
        
        if return_masks:
            masks = self.load_masks(task_path, segmentation_task, 'train and val')
            if validate_on_test:
                masks = np.concatenate((masks, self.load_masks(task_path, segmentation_task, 'test')), axis=0)
            masks = masks.squeeze()
            masks = np.array([cv2.resize(mask, size) for mask in masks]).astype(np.int)
            masks = np.expand_dims(masks, 1)
            train_masks = masks[train_idcs]
            val_masks = masks[val_idcs]
            train_data['mask'] = train_masks
            val_data['mask'] = val_masks
        
        if return_distances:
            distances = self.load_distances()
            distances = distances.reshape((distances.shape[0], 1, *distances.shape[1:]))
            train_distances = distances[train_idcs]
            val_distances = distances[val_idcs]
            train_data['distance_map'] = train_distances
            val_data['distance_map'] = val_distances

        self.train, self.val = train_data, val_data

class DRIVEDataset(data.Dataset):
    PARAMS = {
        'val split percentage': {
            'argument name': 'val_split_percentage',
            'default': 0.25
            },
        'load masks': {
            'argument name': 'return_masks',
            'default': True
            },
        'load eye masks': {
            'argument name': 'return_eye_masks',
            'default': False
            },
        'whole img validation': {
            'argument name': 'whole_img_val',
            'default': True
            },
        'validate on test': {
            'argument name': 'validate_on_test',
            'default': False
            }
        }
    
    # path to data (arrays in .npy format)
    if socket.gethostname() != 'a100.cs.elte.hu':
        PATH = '/data/bakosbence237/datasets/DRIVE/training/cropped_data'
    else:
        PATH = '/projects/SegmentationPretraining/data/DRIVE/training/cropped_data'

    # prepare data to be loaded
    # (functions are created instead of loading data to decrease memory consumption)
    # nonzero masks values are manually cast to 1
    # necessary because all the train and validation regions of interest are marked as 2
    @staticmethod
    def load_imgs(partition):
        return np.load(f'{DRIVEDataset.PATH}/{partition}/imgs.npy')
    
    @staticmethod
    def load_masks(partition):
        return np.where(np.load(f'{DRIVEDataset.PATH}/{partition}/masks.npy') != 0, 1, 0)
    
    @staticmethod
    def load_eye_masks(partition):
        return np.where(np.load(f'{DRIVEDataset.PATH}/{partition}/eye_masks.npy') != 0, 1, 0)
    
    def __init__(self, ds_dict, seed = None, *args, **kwargs):

        return_masks = ds_dict['load masks']
        return_eye_masks = ds_dict['load eye masks']
        split = ds_dict['val split percentage']
        whole_img_val = ds_dict['whole img validation']
        validate_on_test = ds_dict['validate_on_test']

        imgs = self.load_imgs('train and val')
        N = len(imgs)
        
        if validate_on_test:
            imgs = np.concatenate((imgs, self.load_imgs('test')), axis=0)

        imgs = np.moveaxis(imgs, -1, 1) / 255

        rnd = np.random.default_rng(ds_dict.get('seed') or seed)

        if validate_on_test:
            train_idcs = np.concatenate((np.full(N, True), np.full(len(imgs)-N, False)), axis=0)
            val_idcs = ~train_idcs
        elif whole_img_val:
            source_array = np.tile(np.arange(16), 210)      # 16 different whole image in train and validation set. Every 16th patch comes from the same image.
            nr_of_val_ims = int(16*split)
            rnd_arr = np.arange(16)
            rnd.shuffle(rnd_arr)
            val_imgs = rnd_arr[:nr_of_val_ims]
            val_idcs = np.isin(source_array, val_imgs)
            train_idcs = ~val_idcs
        else:
            rnd_arr = np.arange(N)
            rnd.shuffle(rnd_arr)
            val_length = int(split * N)
            train_idcs = rnd_arr >= val_length
            val_idcs = ~train_idcs

        train_imgs = imgs[train_idcs]
        val_imgs = imgs[val_idcs]

        train_data = {'x': train_imgs}
        val_data = {'x': val_imgs}
        
        if return_masks:
            masks = self.load_masks('train and val')
            if validate_on_test:
                masks = np.concatenate((masks, self.load_masks('test')), axis=0)
            masks = masks.reshape((masks.shape[0], 1, *masks.shape[1:]))
            # masks = masks.transpose((0,3,1,2))
            train_masks = masks[train_idcs]
            val_masks = masks[val_idcs]
            train_data['mask'] = train_masks
            val_data['mask'] = val_masks
        
        if return_eye_masks:
            eye_masks = self.load_eye_masks()
            if validate_on_test:
                eye_masks = np.concatenate((eye_masks, self.load_masks('test')), axis=0)
            eye_masks = eye_masks.reshape((eye_masks.shape[0], 1, *eye_masks.shape[1:]))
            train_eye_masks = eye_masks[train_idcs]
            val_eye_masks = eye_masks[val_idcs]
            train_data['eye_mask'] = train_eye_masks
            val_data['eye_mask'] = val_eye_masks

        self.train, self.val = train_data, val_data


class DRIVEDatasetNew(data.Dataset):

    
    PARAMS = dict(base_image_sizes = (512, 512),
                  cropped_image_sizes = (128, 128),
                  train_crop_stride = (32, 32),
                  validation_set = 'test')   # one of 'train', 'test', or 'random'
    
    DEFAULT_VAL_SIZE = 4
    
    @staticmethod
    def fill_kwargs(config_dict):
        if config_dict.get_str('validation_set') == 'random':
            config_dict.get_or_update('validation_set_size', DRIVEDatasetNew.DEFAULT_VAL_SIZE)
    if socket.gethostname() != 'a100.cs.elte.hu':
        BASE_PATH = '/data/bakosbence237/datasets/DRIVE/training/original_data/'
    else:
        BASE_PATH = '/projects/SegmentationPretraining/data/DRIVE/training/original_data/'
    
    def __init__(self, config_dict, seed = None, *args, **kwargs):
        
        img_paths = sorted(glob.glob(self.BASE_PATH + 'train and val/images/*')) + sorted(glob.glob(self.BASE_PATH + 'test/images/*'))
        mask_paths = sorted(glob.glob(self.BASE_PATH + 'train and val/vessel_masks/*')) + sorted(glob.glob(self.BASE_PATH + 'test/vessel_masks/*'))
        train_set_size = len(glob.glob(self.BASE_PATH + 'train and val/images/*'))
        ds_size = len(img_paths)
        
        imgs, masks, from_test = [], [], []
        
        base_sizes = config_dict.get_tuple('base_image_sizes')
        if len(base_sizes) == 1:
            base_sizes = base_sizes * 2
        
        for img_path, mask_path in zip(img_paths, mask_paths):
            assert mask_path.split('/')[-1].split('_')[0] == img_path.split('/')[-1].split('_')[0], f'Imgage and mask file names are not matching at image: {img_path}'
            
            img = Image.open(img_path)
            arr = np.asarray(img) / 255
            arr = cv2.resize(arr, base_sizes[::-1])
            arr = np.moveaxis(arr, -1, 0)
            
            mask_arr = np.asarray(Image.open(mask_path))
            if mask_arr.ndim == 3:
                if mask_arr.shape[-1] > 3:
                    mask_arr = mask_arr[:,  :, :3]
                mask_arr = mask_arr.max(axis = -1)
            mask_arr = cv2.resize(mask_arr, base_sizes[::-1])

            imgs.append(arr)
            masks.append(np.stack(np.where(mask_arr == 0, 0, 1)))
                
        imgs, masks, from_test = np.stack(imgs), np.stack(masks), np.array(np.arange(ds_size)>=train_set_size)
        masks = np.expand_dims(masks, axis=1)
        
        val_set = config_dict.get_str('validation_set')
        if val_set == 'test':
            val_idcs = from_test
        elif val_set == 'train':
            val_idcs = ~from_test
        elif val_set == 'random':
            num_imgs = len(from_test)
            val_set_size = config_dict['validation_set_size']
            val_idcs = np.arange(num_imgs) < val_set_size
            rng = np.random.default_rng(config_dict.get('seed') or seed)
            rng.shuffle(val_idcs)
        else:
            msg = f'Parameter \'validation_set\' must be one of \'test\', \'train\', or \'random\', not \'{val_set}\'.'
            raise ValueError(msg)
        train_imgs, val_imgs = imgs[~val_idcs], imgs[val_idcs]
        train_masks, val_masks = masks[~val_idcs], masks[val_idcs]
        
        crop_sizes = config_dict.get('cropped_image_sizes')
                                     
        if crop_sizes is None or crop_sizes == base_sizes:
            self.train = {'x': train_imgs, 'mask': train_masks}
            self.val = {'x': val_imgs, 'mask': val_masks}
            return
        if not isinstance(crop_sizes, (list, tuple)):
            crop_sizes = (int(crop_sizes),)
        if len(crop_sizes) == 1:
            crop_sizes = crop_sizes * 2
        
        stride = config_dict.get('train_crop_stride') or crop_sizes
        if not isinstance(stride, (list, tuple)):
            stride = (stride,)
        if len(stride) == 1:
            stride = stride * 2
        
        for set_type, strides, base_imgs, base_masks in zip(('train', 'val'),
                                                            (stride, crop_sizes),
                                                            (train_imgs, val_imgs),
                                                            (train_masks, val_masks)):
            imgs, masks = [], []
            num_crops = [(base_size - crop_size) // s + 1
                         for base_size, crop_size, s in zip(base_sizes, crop_sizes, strides)]

            for img, mask in zip(base_imgs, base_masks):
                for i, j in itertools.product(range(num_crops[0]), range(num_crops[1])):
                    x_start = min(i * strides[0], base_sizes[0] - crop_sizes[0])
                    y_start = min(j * strides[1], base_sizes[1] - crop_sizes[1])
                    slice_idcs = (slice(None), slice(x_start, x_start + crop_sizes[0]), slice(y_start, y_start + crop_sizes[1]))
                    cropped_img = img[slice_idcs]
                    imgs.append(cropped_img)
                    masks.append(mask[slice_idcs])
            
            imgs, masks = np.stack(imgs), np.stack(masks)
            # if masks.shape[1] == 1:
            #     masks = masks.squeeze(axis = 1)
            print(masks.shape)
            setattr(self, set_type, dict(x = imgs, mask = masks))


        

class DummySegmentationDataset(data.Dataset):
    PARAMS = {
        'image size': 256,
        'channel size': 3,
        'mask channel size': 1,
        'train size': 32,
        'val size': 32,
        'return binary class': True,
        'datatype': float
    }

    def __init__(self, ds_dict, *args, **kwargs):
        img_size = ds_dict['image size']
        channel_size = ds_dict['channel size']
        mask_channel_size = ds_dict['mask channel size']
        dtype = ds_dict['datatype']
        train_size = ds_dict['train size']
        val_size = ds_dict['val size']
        return_positives = ds_dict['return binary class']

        rng = np.random.default_rng()
        
        img_sizes = (channel_size, img_size, img_size)
        if dtype == int:
            gen_imgs = lambda n: rng.integers(0, 256, size = (n, *img_sizes))
        else:
            gen_imgs = lambda n: rng.uniform(0, 1, size = (n, *img_sizes))
        
        gen_mask = lambda n: rng.integers(0, 2, size = (n, mask_channel_size, img_size, img_size))

        train_data = {'x': gen_imgs(train_size), 'mask': gen_mask(train_size)}
        val_data = {'x': gen_imgs(val_size), 'mask': gen_mask(val_size)}

        if return_positives:
            gen_idcs = lambda n: rng.integers(0, 2, size = n, dtype = bool)
            train_data['positives'] = gen_idcs(train_size)
            val_data['positives'] = gen_idcs(val_size)
        
        self.train, self.val = train_data, val_data

class Synapse(data.Dataset):
    
    PARAMS = {
        'validation split type': {
            'argument name': 'validation_split_type',
            'default': 'n_out'
            },
        'validate on test': {
            'argument name': 'validate_on_test',
            'default': False
            },
        'image sizes': {
            'argument name': 'image_sizes',
            'default': 224
            },
        'number_of_train_samples': 'all',
        'return_sources': False
    }
    
    SPLIT = 0.25
    N_OUT = 4
    
    CLASSES = ('aorta', 'gallbladder', 'left kidney', 'right kidney',
               'liver', 'pancreas', 'spleen', 'stomach')

    @staticmethod
    def fill_kwargs(config_dict):
        val_type = config_dict['validation_split_type']
        if isinstance(val_type, utils.config_dict.ConfigDict):
            val_type = val_type.key()
        if val_type == 'n_out':
            config_dict.get_or_update('number_of_validation_samples', Synapse.N_OUT)
        elif val_type == 'random':
            config_dict.get_or_update('val_split_percentage', Synapse.SPLIT)
        elif val_type == 'test':
            config_dict['validate_on_test'] = True
        elif config_dict['validate_on_test']:
            config_dict['validation_split_type'] = 'test'
        else:
            raise ValueError(f'Validation split type should be \'n_out\' or \'random\', not {val_type}.')
    
    # path to data (arrays in .npy format)
    if socket.gethostname() != 'a100.cs.elte.hu':
        PATH = '/data/unext/data/Synapse/arrays'
    else:
        PATH = '/projects/SegmentationPretraining/data/Synapse/arrays'

    @staticmethod
    def load_imgs(partition):
        file_name = 'test_imgs' if partition == 'test' else 'imgs'
        return np.load(f'{Synapse.PATH}/{partition}/{file_name}.npy')
    
    @staticmethod
    def load_masks(partition):
        file_name = 'test_masks' if partition == 'test' else 'masks'
        return np.load(f'{Synapse.PATH}/{partition}/{file_name}.npy').astype(np.float)

    @staticmethod
    def load_sources(partition):
        file_name = 'test_sources' if partition == 'test' else 'sources'
        return np.load(f'{Synapse.PATH}/{partition}/{file_name}.npy')
    
    def __init__(self, ds_dict, seed = None, *args, **kwargs):
        
        # load arguments
        val_type = ds_dict['validation_split_type']
        validate_on_test = ds_dict['validate_on_test']
        size = ds_dict['image_sizes']
        
        # load images and masks
        imgs = self.load_imgs('train')
        masks = self.load_masks('train')
        sources = self.load_sources('train')
        N = len(imgs)
        
        if validate_on_test:
            imgs = np.concatenate((imgs, self.load_imgs('test')), axis=0)
            masks = np.concatenate((masks, self.load_masks('test')), axis=0)
            sources = np.concatenate((sources, self.load_sources('test')))
        
        if isinstance(size, int):
            size = (size, size)
        size = np.array(size)        
        imgs = np.array([cv2.resize(img, size) for img in imgs])
        masks = np.array([cv2.resize(mask, size) for mask in masks]).astype(np.int)

        # add color channel to image and mask
        imgs = np.expand_dims(imgs, 1)
        masks = np.expand_dims(masks, 1)

        rng = np.random.default_rng(ds_dict.get('seed') or seed)
        if validate_on_test:
            train_idcs = np.concatenate((np.full(N, True), np.full(len(imgs)-N, False)), axis=0)
            val_idcs = ~train_idcs
        elif val_type == 'random':
            rnd_arr = np.arange(N)
            rng.shuffle(rnd_arr)
            val_length = int(ds_dict['val_split_percentage'] * N)
            train_idcs = rnd_arr >= val_length
            val_idcs = ~train_idcs
        else:
            samples = np.unique(sources)
            val_samples = rng.choice(samples, size = ds_dict['number_of_validation_samples'], replace = False)
            val_idcs = np.isin(sources, val_samples)
            train_idcs = ~val_idcs
        
        train_sources = sources[train_idcs]
        num_train_samples = ds_dict.get_str('number_of_train_samples')
        if num_train_samples != 'all':
            all_sources = np.unique(train_sources)
            source_idcs = rng.permute(len(all_sources)) < num_train_samples
            train_samples = all_sources[source_idcs]
            train_idcs = train_idcs & np.isin(sources, train_samples)

        train_imgs, train_masks = imgs[train_idcs], masks[train_idcs]
        val_imgs, val_masks = imgs[val_idcs], masks[val_idcs]
        
        train_positives = np.any(train_masks > 0, axis = tuple(range(1, len(masks.shape))))
        val_positives = np.any(val_masks > 0, axis = tuple(range(1, len(masks.shape))))

        train_data = {'x': train_imgs, 'mask': train_masks, 'positives': train_positives}
        val_data = {'x': val_imgs, 'mask': val_masks, 'positives': val_positives}
        
        if ds_dict.get('return_sources', False):
            train_data['sources'] = sources[train_idcs]
            val_data['sources'] = sources[val_idcs]
        
        self.train, self.val = train_data, val_data

class IDRiD(data.Dataset):
    
    CLASSES_DICT = {'MA': 'microaneurysms',
                    'HE': 'haemorrhages',
                    'EX': 'hard exudates',
                    'SE': 'soft exudates',
                    'OD': 'optic disc'}
    
    PARAMS = dict(base_image_sizes = (1024, 2048),
                  cropped_image_sizes = (512, 512),
                  train_crop_stride = (256, 256),
                  darkness_threshold = 0.99, # drop crops that are darker than this (i. e. mostly background)
                  task = 'MA',               # one of 'MA', 'HE', 'EX', 'SE', 'OD', or a tuple containing these
                  validation_set = 'test',   # one of 'train', 'test', or 'random'
                  train_set_size = 54)
    
    DEFAULT_VAL_SIZE = 27
    
    @staticmethod
    def fill_kwargs(config_dict):
        if config_dict.get_str('validation_set') == 'random':
            config_dict.get_or_update('validation_set_size', IDRiD.DEFAULT_VAL_SIZE)
        if len(config_dict.get_str_tuple('task')) == 1:
            config_dict.get_or_update('include_negatives', False)
        else:
            config_dict.get_or_update('multilabel', False)        
            # There are 4018 pixels total in the 81 2848x4288 images
            # which are annotated as several different masks.
            # If the 'multilabel' key is set to False, these overlaps will be ignored,
            # and the output masks will be single label classification masks,
            # so only one of the two marked classes will be present in the output mask.
            # If set to True, then the masks will be 0-1 matrices,
            # and the model output should not be a distribution,
            # but a separate probability for each class
            # (e. g. the end activation should be a sigmoid instead of a softmax layer)
    
    SUPER_DIR = '/data/SegmentationPretraining/datasets/' if socket.gethostname() != 'a100.cs.elte.hu' else '/projects/SegmentationPretraining/data/'
    BASE_PATH = SUPER_DIR + 'idrid/Segmentation/A. Segmentation/'
    IMGS_PATH = BASE_PATH + '1. Original Images/'
    MASKS_PATH = BASE_PATH + '2. All Segmentation Groundtruths/'
    
    FNAME_PATTERN = re.compile(f'{IMGS_PATH}(?P<source_set>a. Training Set|b. Testing Set)/IDRiD_(?P<id>\d+).jpg')
    
    LEFT_CROP, RIGHT_CROP = 96, 96
    
    def __init__(self, config_dict, seed = None, *args, **kwargs):
        
        tasks = config_dict.get_str_tuple('task')
        self.CLASSES = [self.CLASSES_DICT[task] for task in tasks]
        task_dirs = [f'{list(self.CLASSES_DICT.values()).index(task) + 1}. {task.title()}' for task in self.CLASSES]
        
        img_paths = sorted(glob.glob(self.IMGS_PATH + 'a. Training Set/*') + glob.glob(self.IMGS_PATH + 'b. Testing Set/*'))
        
        include_negatives = len(tasks) > 1 or config_dict['include_negatives']
        multilabel = len(tasks) == 1 or config_dict['multilabel']
        imgs, masks, from_test = [], [], []
        
        base_sizes = config_dict.get_tuple('base_image_sizes')
        if len(base_sizes) == 1:
            base_sizes = base_sizes * 2
        
        for img_path in img_paths:
            match = self.FNAME_PATTERN.match(img_path)
            if match is None:
                continue
            groupdict = match.groupdict()
            source_set, img_id = groupdict['source_set'], groupdict['id']
            
            img = Image.open(img_path)
            arr = np.asarray(img)[:, self.LEFT_CROP : -self.RIGHT_CROP] / 255
            arr = cv2.resize(arr, base_sizes[::-1])
            arr = np.moveaxis(arr, -1, 0)
            
            mask = []
            if not multilabel:
                mask.append(np.zeros(base_sizes, dtype = int))
            include_img = True
            for task_abbrev, task_dir_name in zip(tasks, task_dirs):
                mask_path = os.path.join(self.MASKS_PATH, source_set, task_dir_name, f'IDRiD_{img_id}_{task_abbrev}.tif')
                if not os.path.isfile(mask_path):
                    if include_negatives:
                        mask.append(np.zeros(base_sizes, dtype = int))
                    else:
                        include_img = False
                        break
                else:
                    mask_arr = np.asarray(Image.open(mask_path))[:, self.LEFT_CROP : - self.RIGHT_CROP]
                    if mask_arr.ndim == 3:
                        if mask_arr.shape[-1] > 3:
                            mask_arr = mask_arr[:,  :, :3]
                        mask_arr = mask_arr.max(axis = -1)
                    mask_arr = cv2.resize(mask_arr, base_sizes[::-1])
                    mask.append(np.where(mask_arr == 0, 0, 1))
            
            if include_img:
                imgs.append(arr)
                masks.append(np.stack(mask))
                from_test.append(source_set == 'b. Testing Set')
                
        imgs, masks, from_test = np.stack(imgs), np.stack(masks), np.array(from_test)
        
        if not multilabel:
            masks = masks.argmax(axis = 1, keepdims = True)
        
        rng = np.random.default_rng(config_dict.get('seed') or seed)
        
        val_set = config_dict.get_str('validation_set')
        if val_set == 'test':
            val_idcs = from_test
        elif val_set == 'train':
            val_idcs = ~from_test
        elif val_set == 'random':
            num_imgs = len(from_test)
            val_set_size = config_dict['validation_set_size']
            val_idcs = rng.permutation(num_imgs) < val_set_size
        else:
            msg = f'Parameter \'validation_set\' must be one of \'test\', \'train\', or \'random\', not \'{val_set}\'.'
            raise ValueError(msg)
        
        train_imgs, val_imgs = imgs[~val_idcs], imgs[val_idcs]
        train_masks, val_masks = masks[~val_idcs], masks[val_idcs]
        
        train_set_size = config_dict.get('train_set_size')
        if train_set_size > len(train_imgs):
            raise ValueError(f'Specified train set size {train_set_size} is larger than the total number of train images ({len(train_imgs)}).')
        
        train_idcs = rng.permutation(len(train_imgs)) < train_set_size
        train_imgs, train_masks = train_imgs[train_idcs], train_masks[train_idcs]
        
        crop_sizes = config_dict.get('cropped_image_sizes')
        if crop_sizes is None or crop_sizes == base_sizes:
            if train_masks.shape[1] == 1:
                train_masks, val_masks = train_masks.squeeze(axis = 1), val_masks.squeeze(axis = 1)
            self.train = {'x': train_imgs, 'mask': train_masks}
            self.val = {'x': val_imgs, 'mask': val_masks}
            return
        if not isinstance(crop_sizes, (list, tuple)):
            crop_sizes = (int(crop_sizes),)
        if len(crop_sizes) == 1:
            crop_sizes = crop_sizes * 2
        
        stride = config_dict.get('train_crop_stride') or crop_sizes
        if not isinstance(stride, (list, tuple)):
            stride = (stride,)
        if len(stride) == 1:
            stride = stride * 2
        
        threshold = 1 - config_dict['darkness_threshold']
        
        for set_type, strides, base_imgs, base_masks in zip(('train', 'val'),
                                                            (stride, crop_sizes),
                                                            (train_imgs, val_imgs),
                                                            (train_masks, val_masks)):
            imgs, masks = [], []
            num_crops = [(base_size - crop_size) // s + 1
                         for base_size, crop_size, s in zip(base_sizes, crop_sizes, strides)]
            
            for img, mask in zip(base_imgs, base_masks):
                for i, j in itertools.product(range(num_crops[0]), range(num_crops[1])):
                    x_start = min(i * strides[0], base_sizes[0] - crop_sizes[0])
                    y_start = min(j * strides[1], base_sizes[1] - crop_sizes[1])
                    slice_idcs = (slice(None), slice(x_start, x_start + crop_sizes[0]), slice(y_start, y_start + crop_sizes[1]))
                    cropped_img = img[slice_idcs]
                    if np.mean(cropped_img) < threshold:
                        continue
                    imgs.append(cropped_img)
                    masks.append(mask[slice_idcs])
            
            imgs, masks = np.stack(imgs), np.stack(masks)
            if masks.shape[1] == 1:
                masks = masks.squeeze(axis = 1)
            setattr(self, set_type, dict(x = imgs, mask = masks))



class ACDC(data.Dataset):
    PARAMS = {
        'val split percentage': {
            'argument name': 'val_split_percentage',
            'default': 0.2
            },
        'validate on test': {
            'argument name': 'validate_on_test',
            'default': False
            },
        'image sizes': {
            'argument name': 'image_sizes',
            'default': 256
            }
        }
    

    CLASSES = ('RV cavity','myocardium', 'LV cavity')

    # path to data (arrays in .npy format)
    if socket.gethostname() != 'a100.cs.elte.hu':
        PATH = '/data/.../ACDC/'
    else:
        PATH = '/projects/SegmentationPretraining/data/ACDC/database_raw/'


    
    def __init__(self, ds_dict, seed = None, *args, **kwargs):

        split = ds_dict['val split percentage']
        validate_on_test = ds_dict['validate_on_test']
        size = ds_dict['image sizes']

        if isinstance(size, int):
            size = (size, size)

        patient_paths = [p  for p in glob.glob(ACDC.PATH+'training/*') if os.path.isdir(p)]
        N = len(patient_paths)

        if not validate_on_test:
            rnd = np.random.default_rng(ds_dict.get('seed') or seed)
            rnd_arr = np.arange(len(patient_paths))
            rnd.shuffle(rnd_arr)
            val_length = int(split * N)
            train_idcs = rnd_arr >= val_length
            val_idcs = ~train_idcs
        else:
            patient_paths += [p  for p in glob.glob(ACDC.PATH+'testing/*') if os.path.isdir(p)]
            train_idcs = np.concatenate((np.full(N, True), np.full(len(patient_paths)-N, False)), axis=0)
            val_idcs = ~train_idcs

        # print(patient_paths)
        pattern = '*frame*.nii.gz'
        train_frame_paths = []
        # print(np.array(patient_paths)[train_idcs])
        for train_patient in np.array(patient_paths)[train_idcs]:
            train_frame_paths+=sorted(glob.glob(f'{train_patient}/{pattern}'))
            # print(train_frame_paths)
            train_img_paths, train_label_paths = train_frame_paths[0::2],train_frame_paths[1::2]
            for i, l in zip(train_img_paths, train_label_paths):
                assert i[:-7]+'_gt.nii.gz' == l, f'Wrong path pairing! img path: {i}, labels path: {l}'
                # print(i)
            
        train_imgs = [cv2.resize(load_nii(img_path)[0], size) for img_path in train_img_paths]
        train_labels = [cv2.resize(load_nii(label_path)[0], size) for label_path in train_label_paths]
        
        train_imgs = np.concatenate(train_imgs, axis=2).transpose(2,0,1)
        train_labels = np.concatenate(train_labels, axis=2).transpose(2,0,1).astype(np.int)
        
        
        val_frame_paths = []
        for val_patient in np.array(patient_paths)[val_idcs]:
            val_frame_paths+=sorted(glob.glob(f'{val_patient}/{pattern}'))
            val_img_paths, val_label_paths = val_frame_paths[0::2], val_frame_paths[1::2]
            for i, l in zip(val_img_paths, val_label_paths):
                assert i[:-7]+'_gt.nii.gz' == l, f'Wrong path pairing! img path: {i}, labels path: {l}'
            
        val_imgs = [cv2.resize(load_nii(img_path)[0], size) for img_path in val_img_paths]
        val_labels = [cv2.resize(load_nii(label_path)[0], size) for label_path in val_label_paths]

        val_imgs = np.concatenate(val_imgs, axis=2).transpose(2,0,1)
        val_labels = np.concatenate(val_labels, axis=2).transpose(2,0,1).astype(np.int)
        

        self.train = {'x': np.expand_dims(train_imgs, 1),
                      'mask': np.expand_dims(train_labels, 1)}
        
        print(self.train['x'].shape)
        print(self.train['mask'].shape)

        self.val = {'x': np.expand_dims(val_imgs, 1),
                    'mask': np.expand_dims(val_labels, 1)}

        


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header
