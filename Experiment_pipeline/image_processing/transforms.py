import numpy as np
import cv2

from transform import TransformWrapper

class CLAHE(object):
    
    rgb_channels = ('red', 'green', 'blue')
    
    def __init__(self, clipLimit = 40.0, tileGridSize = (8, 8), channels = 'all', *args, **kwargs):
        if not isinstance(tileGridSize, (tuple, list)):
            tileGridSize = (tileGridSize,) * 2
        self.clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize = tileGridSize)
        if channels != 'all':
            if not isinstance(channels, (list, tuple)):
                channels = (channels,)
            self.channels = []
            for channel in channels:
                if channel in self.rgb_channels:
                    self.channels.append(self.rgb_channels.index(channel))
                elif isinstance(channel, int):
                    self.channels.append(channel)
                else:
                    raise ValueError(f'Each channels should either be one one of {self.rgb_channels} or an int, not {channel}.')
        else:
            self.channels = 'all'

    def __call__(self, x, *args, **kwargs):
        x = np.asarray(x*255, dtype = np.uint8)
        # print(x.ndim, x.shape, x.max())
        if x.squeeze().ndim == 2:
            # print(x)
            return self.clahe.apply(x)/255
        else:
            axes = self.channels if self.channels != 'all' else range(x.shape[0])
            for i in axes:
                x[i] = self.clahe.apply(x[i])
            return x/255
        
class ToNumpyArray(TransformWrapper):
    
    def __init__(self, *args, **kwargs):
        return
    
    def transform(self, input):
        input['x'] = np.asarray(input['x'])
        return input

class ComplementChannels(object):
    
    def __init__(self, num_channels = 3, *args, **kwargs):
        self.num_channels = num_channels
    
    def __call__(self, x, *args, **kwargs):
        if len(x.shape) <= 2:
            x = x.reshape(1, *x.shape)
        num_channels = x.shape[0]
        if num_channels < self.num_channels: # TODO: do we want it to only work on numpy arrays?
            x = np.concatenate([x] * (self.num_channels // num_channels) + [x[: self.num_channels % num_channels]], axis = 0)
        elif num_channels > self.num_channels:
            raise ValueError(f'Expected input to have at most {self.num_channels} channels, but got {num_channels}.')
        return x

class RandomPatchErase(object):
    
    def __init__(self, erasure_rate = 0.2, min_patch_size = 16, seed = None, *args, **kwargs):
        self.rng = np.random.default_rng(seed)
        
        if isinstance(erasure_rate, (int, float)):
            erasure_rate = (erasure_rate,) * 2
        self.erasure_rate = erasure_rate
        self.min_patch_size = min_patch_size
    
    def __call__(self, img, *args, **kwargs):
        num_patches = (img.shape[-2] // self.min_patch_size, img.shape[-1] // self.min_patch_size)
        curr_erase_rate = self.rng.uniform(*self.erasure_rate)
        mask = self.rng.binomial(1, curr_erase_rate, size = num_patches)
        mask = cv2.resize(mask, img.shape[-2:], interpolation = cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, 0)
        return np.where(mask == 0, img, np.zeros_like(img))