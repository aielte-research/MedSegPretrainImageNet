import glob
import json
import re
import pandas as pd
import torchvision
import numpy as np

from PIL import Image

import socket

import data

class CIFAR10(data.Dataset):

    PARAMS = {'validate_on_test': True}

    ROOT = '/data/hidygabor/' if socket.gethostname() != 'a100.cs.elte.hu' else '/home/hidygabor/'

    def __init__(self, config_dict, *args, **kwargs):
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        train_ds = torchvision.datasets.CIFAR10(root = self.ROOT, train = True, download = True)
        self.CLASSES = train_ds.classes
        imgs = np.moveaxis(train_ds.data, -1, 1) / 255
        labels = [[y] for y in train_ds.targets]

        if self.validate_on_test:
            self.train = {'x': imgs, 'label': labels}
            
            test_ds = torchvision.datasets.CIFAR10(root = self.ROOT, train = False, download = True)
            val_imgs = np.moveaxis(test_ds.data, -1, 1) / 255
            val_labels = [[y] for y in test_ds.targets]
            self.val = {'x': val_imgs, 'label': val_labels}

class MNIST(data.Dataset):
    
    PARAMS = {'validate_on_test': True}

    ROOT = '/data/hidygabor/' if socket.gethostname() != 'a100.cs.elte.hu' else '/home/hidygabor/'

    def __init__(self, config_dict, *args, **kwargs):
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        train_ds = torchvision.datasets.MNIST(root = self.ROOT, train = True, download = True)
        self.CLASSES = list(range(10))
        imgs = train_ds.data / 255
        labels = [[y] for y in train_ds.targets]

        if self.validate_on_test:
            self.train = {'x': imgs, 'label': labels}
            
            test_ds = torchvision.datasets.MNIST(root = self.ROOT, train = False, download = True)
            val_imgs = test_ds.data / 255
            val_labels = [[y] for y in test_ds.targets]
            self.val = {'x': val_imgs, 'label': val_labels}

class ImageNet(data.Dataset):

    PARAMS = {'use_official_validation': True}
    SPLIT = 0.2

    ARRAYS_PATH = '/home/hidygabor/datasets/imagenet/data/'
    LABELS_PATH = '/home/hidygabor/datasets/imagenet/labels.json'

    HOST = 'a100.cs.elte.hu'

    @staticmethod
    def fill_kwargs(config_dict):
        if not config_dict['use_official_validation']:
            config_dict.get_or_update('split', ImageNet.SPLIT)
    
    def __init__(self, config_dict, seed = None, *args, **kwargs):

        if socket.gethostname() != self.HOST:
            raise FileNotFoundError(f'The ImageNet data is only available on {self.HOST}, not {socket.gethostname()}.')
        
        with open(self.LABELS_PATH, 'r') as labels_file:
            labels_dict = json.load(labels_file)
        img_paths, labels = list(labels_dict.keys()), list(labels_dict.values())
        img_paths, labels = np.array(img_paths), np.array(labels)
        
        labels = labels - 1 # official labels start numbering at 1

        if config_dict['use_official_validation']:
            val_idcs = np.array(['val' in img_path.split('/')[-1] for img_path in img_paths])
        else:
            split = config_dict['split']
            rng = np.random.default_rng(seed)

            N = len(img_paths)
            rnd_arr = np.arange(N)
            rng.shuffle(rnd_arr)
            val_idcs = rnd_arr < int(split * N)
        
        train_idcs = ~val_idcs
        self.train = {'x': img_paths[train_idcs], 'label': labels[train_idcs]}
        self.val = {'x': img_paths[val_idcs], 'label': labels[val_idcs]}
        self.load_function = lambda fname: np.load(fname) / 255

class KaggleDRDataset(data.Dataset):
    
    # Dataset class of https://www.kaggle.com/competitions/diabetic-retinopathy-detection/
    
    PATH = '/data/SegmentationPretraining/datasets/KaggleDR/'
    
    PARAMS = dict(validation_set = 'random', # one of 'random', 'train', or 'test'
                  labels = 'symptomaticity') # one of 'symptomaticity', 'referability', or 'multilevel' 
                  # 'symptomaticity' labels:
                            # 0: no lesions
                            # 1: lesions present
                  # 'referability' labels:
                            # 0: no or mild DR
                            # 1: at least moderate DR
                  # 'multilevel' labels:
                            # 0: no DR
                            # 1: mild DR
                            # 2: moderate DR
                            # 3: severe DR
                            # 4: proliferative DR
    
    DEFAULT_SPLIT = 0.02
    
    FPATH_PATTERN = re.compile(PATH + '(?P<type>train|test)/(?P<img_id>(?P<patient_id>\d+)_(?P<side>right|left)).npy')
    
    @staticmethod
    def fill_kwargs(config_dict):
        if config_dict.get_str('validation_set') == 'random':
            config_dict.get_or_update('validation_split', KaggleDRDataset.DEFAULT_SPLIT)
    
    def __init__(self, config_dict, seed = None, *args, **kwargs):
        pathnames = glob.glob(self.PATH + 'train/*') + glob.glob(self.PATH + 'test/*')
        
        train_df = pd.read_csv(self.PATH + 'trainLabels.csv.zip')
        test_df = pd.read_csv(self.PATH + 'retinopathy_solution.csv')[train_df.columns]
        df = pd.concat([train_df, test_df])
        unordered_labels : pd.Series = df.set_index('image').squeeze()
        
        val_set = config_dict.get_str('validation_set')
        if val_set == 'random':
            rng = np.random.default_rng(config_dict.get('seed') or seed)
            num_patients = len(pathnames) // 2 # for every patient we have an image of the left and right eye
            num_val_records = np.round(config_dict['validation_split'] * num_patients).astype(int)
            val_idcs = 1 + rng.choice(num_patients, num_val_records, replace = False)
            
            def validate(patient_id, **kwargs):
                return int(patient_id) in val_idcs
        elif val_set in ('train', 'test'):
            def validate(type, **kwargs):
                return type == val_set
        else:
            msg = f'Validation set type must be one of \'train\', \'test\', or \'random\', not \'{val_set}\'.'
            raise ValueError(msg)
        
        is_val, labels, filtered_pathnames = [], [], []
        for pathname in pathnames:
            match = self.FPATH_PATTERN.match(pathname)
            if match is None:
                continue
            properties = match.groupdict()
            is_val.append(validate(**properties))
            labels.append(unordered_labels[properties['img_id']])
            filtered_pathnames.append(pathname)
        labels = np.array(labels)
        pathnames = np.array(filtered_pathnames)
        is_val = np.array(is_val)
        
        label_type = config_dict.get_str('labels')
        if label_type == 'symptomaticity':
            labels = np.where(labels == 0, 0, 1)
        elif label_type == 'referability':
            labels = np.where(labels <= 1, 0, 1)
        elif label_type != 'multilevel':
            msg = f'Label argument must be one of \'symptomaticity\', \'referability\', or \'multilevel\', but got \'{label_type}\'.'
            raise ValueError(msg)
        
        self.train = {'x': pathnames[~is_val], 'label': labels[~is_val]}
        self.val = {'x': pathnames[is_val], 'label': labels[is_val]}
        
        self.load_function = np.load