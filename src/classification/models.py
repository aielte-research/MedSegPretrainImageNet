import numpy as np
from typing import Literal, Optional, Tuple, Union
import torch
from torch import nn

import utils
from segmentation.models import drop_path

class DeepResNet(nn.Module):
    
    def __init__(self, version = 'v1', bottleneck = True, channel_sizes = (256, 512, 1024, 2048),
                 widths = (3, 4, 6, 3), in_channels = 3, base_channel_size = 64, bias = True,
                 head = False, stochastic_depth_rate = 0, *args, **kwargs):
        
        if isinstance(version, int):
            version = f'v{version}'
        if version not in ('v1', 'v2'):
            msg = f'`version` parameter of ResNet must be \'v1\' or \'v2\', but got \'{version}\'.'
            raise ValueError(msg)
        
        if len(widths) != len(channel_sizes):
            w_d = len(widths)
            c_d = len(channel_sizes)
            msg = f'Each level of the ResNet needs one channel size and one width associated with it, but got {w_d} width valuesand {c_d} channel size values.'
            raise ValueError(msg)
    
        self.version = version
        self.bottleneck = bottleneck
        self.channel_sizes = channel_sizes
        self.widths = widths
        self.in_channels = in_channels
        self.base_channel_size = base_channel_size
        self.bias = bias
        self.head = head
        self.stochastic_depth_rate = stochastic_depth_rate
        if head:
            self.output_size = kwargs['output_size']

        super().__init__()
        
        if version == 'v1':
            self.stem = nn.Sequential(
                                nn.Conv2d(in_channels = in_channels,
                                          out_channels = base_channel_size,
                                          kernel_size = 7,
                                          stride = 2, padding = 3, bias = bias),
                                nn.BatchNorm2d(num_features = base_channel_size),
                                nn.ReLU()
                                )
        else:
            self.stem = nn.Conv2d(in_channels = in_channels,
                                  out_channels = base_channel_size,
                                  kernel_size = 7,
                                  stride = 2, padding = 3, bias = bias)
        
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        unit_size = 3 if bottleneck else 2
        drop_probabilities = np.linspace(0, stochastic_depth_rate or 0, sum(widths))
        
        self.levels = nn.ModuleList()
        for i, (width, in_channel_size, out_channel_size) in enumerate(zip(widths,
                                                                           (base_channel_size,
                                                                            *channel_sizes[:-1]),
                                                                           channel_sizes)):
            self.levels.append(ResBlock(width * unit_size, in_channel_size, out_channel_size,
                                        version = version, bottleneck = bottleneck,
                                        downsample = bool(i), bias = bias,
                                        drop_probabilities = drop_probabilities[sum(widths[:i]) : sum(widths[:i+1])]))
        
        if head:
            if version == 'v1':
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size = 1),
                    nn.Flatten(),
                    nn.Linear(in_features = channel_sizes[-1], out_features = kwargs['output_size'])
                    )
            else:
                self.classifier = nn.Sequential(
                    nn.BatchNorm2d(num_features = channel_sizes[-1]),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(output_size = 1),
                    nn.Flatten(),
                    nn.Linear(in_features = channel_sizes[-1], out_features = kwargs['output_size'])
                    )
        else:
            self.classifier = nn.Identity()
    
    def forward(self, x, return_skip_vals = False, *args, **kwargs):
        y = self.stem(x)
        skip_values = [y]
        
        y = self.max_pool(y)
        for level in self.levels:
            y = level(y)
            skip_values.append(y)
        
        y = self.classifier(y)
        if return_skip_vals:
            return y, skip_values[:-1]
        
        else:
            return y
    
    def __str__(self):
        kws = ['version', 'bottleneck', 'channel_sizes', 'widths', 'base_channel_size', 'in_channels', 'bias']
        if self.stochastic_depth_rate:
            kws.append('stochastic_depth_rate')
        if self.head:
            kws += ['head', 'output_size']
        kwargs = ', '.join(f'{kw} = {getattr(self, kw)}' for kw in kws)
        return f'DeepResNet({kwargs})'
        

class ResBlock(nn.Sequential):
    
    def __init__(self, size, in_channels, out_channels, version = 'v1', bottleneck = True,
                 downsample = False, bias = True, drop_probabilities = None, *args, **kwargs):
        if isinstance(version, int):
            version = f'v{version}'
        if version not in ('v1', 'v2'):
            msg = f'`version` parameter of ResBlock must be \'v1\' or \'v2\', but got \'{version}\'.'
            raise ValueError(msg)
        
        unit_size = 3 if bottleneck else 2
        if size % unit_size != 0:
            msg = f'Size of residual block must be divisible by {unit_size}, but got {size}.'
            raise ValueError(msg)
        n = size // unit_size
        
        if drop_probabilities is None:
            drop_probabilities = (0,) * n
        elif len(drop_probabilities) != n:
            msg = f'Number of drop probabilities given must equal the number of blocks, but got {n} blocks and {len(drop_probabilities)} probabilities ({drop_probabilities}).'
            raise ValueError(msg)

        super().__init__()
        
        if version == 'v1':
            basic_block = BottleNeckBlock if bottleneck else BasicBlock
        else:
            basic_block = BottleNeckBlockV2 if bottleneck else BasicBlockV2

        layers = []
        for i, p in enumerate(drop_probabilities):
            layers.append(basic_block(
                            in_channels = in_channels if i == 0 else out_channels,
                            out_channels = out_channels,
                            downsample = downsample and i == 0,
                            bias = bias,
                            drop_probability = p)
                          )
        
        super().__init__(*layers)

class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsample = False, bias = True,
                 drop_probability = 0, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        kernel_size = 3,
                        stride = 2 if downsample else 1,
                        padding = 1,
                        bias = bias
                        )
        self.bn1 = nn.BatchNorm2d(num_features = out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
                        in_channels = out_channels,
                        out_channels = out_channels,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1,
                        bias = bias
                        )
        self.bn2 = nn.BatchNorm2d(num_features = out_channels)
        self.relu2 = nn.ReLU()
        
        if downsample:
            downsample_fn = nn.AvgPool2d(kernel_size = 1, stride = 2)
        else:
            downsample_fn = nn.Identity()
        
        if out_channels != in_channels:
            zero_fill_channels = out_channels - in_channels
            if zero_fill_channels < 0:
                raise ValueError('Out channel size should not be smaller than in channel size.')
            def shortcut(x):
                shape = list(x.shape)
                shape[1] = zero_fill_channels
                zeros = torch.zeros(shape, device = x.device)
                return torch.cat([x, zeros], axis = 1)
        else:
            shortcut = nn.Identity()
        
        self.skip_connection = lambda x: shortcut(downsample_fn(x))
        self.drop_path = nn.Identity() if drop_probability == 0 else DropPath(drop_probability)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        
        y = self.relu2(self.drop_path(y) + self.skip_connection(x))
        return y
    
class BasicBlockV2(BasicBlock):
    
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels, *args, **kwargs)
        self.bn1 = nn.BatchNorm2d(num_features = in_channels)
    
    def forward(self, x):
        y = self.bn1(x)
        y = self.relu1(y)
        y = self.conv1(x)

        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv2(y)
        return self.drop_path(y) + self.skip_connection(x)

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = False, bias = True, drop_probability = 0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                                     out_channels = out_channels // 4,
                                     kernel_size = 1,
                                     bias = bias)
        self.bn1 = nn.BatchNorm2d(num_features = out_channels // 4)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels = out_channels // 4,
                                     out_channels = out_channels // 4,
                                     kernel_size = 3,
                                     padding = 1,
                                     stride = 2 if downsample else 1,
                                     bias = bias)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels // 4)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels = out_channels // 4,
                                     out_channels = out_channels,
                                     kernel_size = 1,
                                     bias = bias)
        self.bn3 = nn.BatchNorm2d(num_features = out_channels)
        self.relu3 = nn.ReLU()
        
        if downsample:
            downsample_fn = nn.AvgPool2d(kernel_size = 1, stride = 2)
        else:
            downsample_fn = nn.Identity()
        
        if out_channels != in_channels:
            zero_fill_channels = out_channels - in_channels
            if zero_fill_channels < 0:
                raise ValueError('Out channel size should not be smaller than in channel size.')
            def shortcut(x):
                shape = list(x.shape)
                shape[1] = zero_fill_channels
                zeros = torch.zeros(shape, device = x.device)
                return torch.cat([x, zeros], axis = 1)
        else:
            shortcut = nn.Identity()
        
        self.skip_connection = lambda x: shortcut(downsample_fn(x))
        self.drop_path = nn.Identity() if drop_probability == 0 else DropPath(drop_probability)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(self.drop_path(y) + self.skip_connection(x))
        
        return y

class BottleNeckBlockV2(BottleNeckBlock):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.bn1 = nn.BatchNorm2d(num_features = in_channels)
        self.bn3 = nn.BatchNorm2d(num_features = out_channels // 4)
    
    def forward(self, x):
        y = self.bn1(x)
        y = self.relu1(y)
        y = self.conv1(y)
        
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv2(y)
        
        y = self.bn3(y)
        y = self.relu3(y)
        y = self.conv3(y)
        
        return self.drop_path(y) + self.skip_connection(x)

class DropPath(nn.Module):
    
    def __init__(self, p = 0, *args, **kwrags):
        super().__init__()
        self.keep_prob = 1 - p
        self.p = p
    
    def forward(self, x, *args, **kwargs):
        if self.training:
            p_shape = (x.shape[0],) + (1,) * len(x.shape[1:])
            return torch.bernoulli(self.keep_prob * torch.ones(p_shape)).to(x.device) * x
        else:
            return self.keep_prob * x
    
    def __str__(self):
        return f'DropPath(p={self.p})'
    
    def __repr__(self):
        return str(self)