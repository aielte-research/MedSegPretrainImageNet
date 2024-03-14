import glob
import json
import re
import pandas as pd
import torchvision
import numpy as np

import data

class ImageNet(data.Dataset):

    PARAMS = {'use_official_validation': True}
    SPLIT = 0.2

    BASE_PATH = '../data/imagenet/'
    ARRAYS_PATH = BASE_PATH + 'data/'
    LABELS_PATH = BASE_PATH + 'labels.json'

    @staticmethod
    def fill_kwargs(config_dict):
        if not config_dict['use_official_validation']:
            config_dict.get_or_update('split', ImageNet.SPLIT)
    
    def __init__(self, config_dict, seed = None, *args, **kwargs):
        
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
