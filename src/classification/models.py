import numpy as np
from typing import Literal, Optional, Tuple, Union
import torch
from torch import nn

import utils
from segmentation.models import drop_path

class ResNet(nn.Module):
    def __init__(self, size, version = 'v1', channel_sizes = [16, 32, 64], in_channels = 3, output_size = 10, *args, **kwargs):
        assert len(channel_sizes) == 3, 'Channel sizes should change two times.'
        assert size % 3 == 2, f'Size {size} is not of shape 3n + 2.'
        m = int((size - 2) / 3)

        super().__init__()
        
        self.first_layer = nn.Conv2d(
                                in_channels = in_channels,
                                out_channels = 16,
                                kernel_size = 3,
                                stride = 1,
                                padding = 'same'
                                )
        self.bn1 = nn.BatchNorm2d(num_features = channel_sizes[0])
        self.relu = nn.ReLU()
        
        self.res_block1 = ResBlock(size = m, in_channels = channel_sizes[0], out_channels = channel_sizes[0])

        self.res_block2 = ResBlock(size = m, in_channels = channel_sizes[0], out_channels = channel_sizes[1])

        self.res_block3 = ResBlock(size = m, in_channels = channel_sizes[1], out_channels = channel_sizes[2])
        self.global_pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.final_layer = nn.Linear(in_features = channel_sizes[-1], out_features = output_size)
    
    def forward(self, x):
        y = self.first_layer(x)
        y = self.bn1(y)
        y = self.relu(y)
        
        y = self.res_block1(y)

        y = self.res_block2(y)

        y = self.res_block3(y)
        y = self.global_pool(y)

        y = y.squeeze()

        y = self.final_layer(y)
        return y

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


class ResNet3D(nn.Module):
    def __init__(self, size, version = 'v1', channel_sizes = (16, 32, 64),
                 in_channels = 3, head = True, output_size = 10, bottleneck = False, *args, **kwargs):
        assert len(channel_sizes) == 3, 'Channel sizes should change two times.'
        assert size % 3 == 2, f'Size {size} is not of shape 3n + 2.'
        m = int((size - 2) / 3)

        super().__init__()
        
        self.head = head
        self.first_layer = nn.Conv3d(
                                in_channels = in_channels,
                                out_channels = 16,
                                kernel_size = 3,
                                stride = 1,
                                padding = 'same'
                                )
        self.bn1 = nn.BatchNorm3d(num_features = channel_sizes[0])
        self.relu = nn.ReLU()
        
        self.res_block1 = ResBlock3D(size = m, in_channel_size = channel_sizes[0], out_channel_size = channel_sizes[0])

        self.res_block2 = ResBlock3D(size = m, in_channel_size = channel_sizes[0], out_channel_size = channel_sizes[1])

        self.res_block3 = ResBlock3D(size = m, in_channel_size = channel_sizes[1], out_channel_size = channel_sizes[2])
        self.global_pool = nn.AdaptiveAvgPool3d(output_size = 1)
        if self.head:
            self.final_layer = nn.Linear(in_features = channel_sizes[-1], out_features = output_size)
    
    def forward(self, x):
        y = self.first_layer(x)
        y = self.bn1(y)
        y = self.relu(y)
        
        y = self.res_block1(y)

        y = self.res_block2(y)

        y = self.res_block3(y)
        y = self.global_pool(y)
        
        if self.head:
            y = y.squeeze()
            y = self.final_layer(y)
        return y

class ResBlock3D(nn.Module):

    @staticmethod
    def fill_kwargs(config_dict):
        if 'activation_function' in config_dict:
            utils.fill_dict(config_dict['activation_function'])
    
    def __init__(self, size, in_channel_size = 16, out_channel_size = 16,
                 bottleneck = False, kernel_sizes = 3, dilations = 1, version = 'v1',
                 layer_scaling = False, drop_probabilities = None, dropout = False, res_connection = 'identity', *args, **kwargs):
        if not bottleneck:
            assert size % 2 == 0, f'Size must be even, not {size}'
            assert in_channel_size in (out_channel_size, out_channel_size // 2), 'Channel size must either be constant or doubled.'
        else:
            assert size % 3 == 0, f'Size must be divisible by 3 in case of bottleneck residual block. Given size is {size}.'

        n = size // 2 if not bottleneck else size // 3
        downsample = kwargs.pop('downsample', None)
        if downsample is None:
            downsample = (out_channel_size != in_channel_size)
        
        drop_probabilities = drop_probabilities if drop_probabilities is not None else [0 for _ in range(n)]
        if len(drop_probabilities) != n:
            raise ValueError(f'Number of drop probabilites ({len(drop_probabilities)}) does not match size {size}; expected {n} drop probabilites.')
        
        super().__init__()

        layers = []
        if version == 'v1':
            basic_block = BasicBlock3D if not bottleneck else BottleNeckBlock3D
        else:
            basic_block = BasicBlock3Dv2 if not bottleneck else BottleNeckBlockBlock3Dv2
        for i, drop_prob in enumerate(drop_probabilities):
            if dropout:
                layers.append(nn.Dropout(p = dropout))
            layers.append(basic_block(
                            in_channel_size = in_channel_size if i == 0 else out_channel_size,
                            out_channel_size = out_channel_size,
                            downsample = downsample and i == 0,
                            kernel_sizes = kernel_sizes,
                            dilations = dilations,
                            layer_scaling = layer_scaling,
                            drop_probability = drop_prob,
                            res_connection = res_connection,
                            *args, **kwargs
                        )
            )
        self.res_block = nn.Sequential(*layers)
    
    def forward(self, x):
        y = self.res_block(x)
        return y

class BasicBlock3D(nn.Module):

    @staticmethod
    def fill_kwargs(config_dict):
        utils.fill_dict(config_dict['activation_function'])
    
    def __init__(self, in_channel_size = 16, out_channel_size = 16, downsample = False,
                 batch_norm = True, kernel_sizes = 3, dilations = 1, activation_function = 'nn.ReLU',
                 layer_scaling = False, drop_probability = 0.0, res_connection = 'identity', *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv3d(
                        in_channels = in_channel_size,
                        out_channels = out_channel_size,
                        kernel_size = kernel_sizes,
                        stride = 2 if downsample else 1,
                        padding = ((kernel_sizes - 1) * dilations + 1) // 2,
                        dilation = dilations
                        )
        self.act1 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)

        self.conv2 = nn.Conv3d(
                        in_channels = out_channel_size,
                        out_channels = out_channel_size,
                        kernel_size = kernel_sizes,
                        stride = 1,
                        padding = 'same',
                        dilation = dilations
                        )
        self.act2 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm3d(num_features = out_channel_size)
            self.bn2 = nn.BatchNorm3d(num_features = out_channel_size)
        
        self.res_con = res_connection != False

        if self.res_con:
            if res_connection == 'identity':
                if downsample:
                    downsample_fn = nn.AvgPool3d(kernel_size = 1, stride = 2)
                    def shortcut(x):
                        y = downsample_fn(x)
                        zeros = torch.zeros_like(y)
                        return torch.cat([y, zeros], axis = 1)
                    self.skip_connection = shortcut
                elif out_channel_size == 2 * in_channel_size:
                    def shortcut(x):
                        return torch.cat([x, torch.zeros_like(x)], axis = 1)
                    self.skip_connection = shortcut
                else:
                    self.skip_connection = nn.Identity()
            elif res_connection == 'conv':
                self.skip_connection = []
                self.skip_connection.append(nn.Conv3d(in_channels = in_channel_size,
                                                out_channels = out_channel_size,
                                                kernel_size = 2 if downsample else 1,
                                                stride = 2 if downsample else 1,
                                                padding = 'same')
                                            )
                if self.batch_norm:
                    self.skip_connection.append(nn.BatchNorm3d(num_features = out_channel_size))
                self.skip_connection = nn.Sequential(*self.skip_connection)
        self.scaling = LayerScale3D(out_channel_size, layer_scaling) if layer_scaling else nn.Identity()
        self.drop_prob = drop_probability
    
    def forward(self, x):
        y = self.conv1(x)
        if self.batch_norm:
            y = self.bn1(y)
        y = self.act1(y)

        y = self.conv2(y)
        if self.batch_norm:
            y = self.bn2(y)
        
        y = self.scaling(y)
        if self.res_con:
            y = self.act2(drop_path(y, self.drop_prob, self.training) + self.skip_connection(x))
        else:
            y = self.act2(drop_path(y, self.drop_prob, self.training))
        return y

class BasicBlock3Dv2(BasicBlock3D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_norm = kwargs.get('batch_norm', True)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm3d(num_features = kwargs.get('in_channel_size', 16))
    
    def forward(self, x):
        y = self.bn1(x) if self.batch_norm else x
        y = self.act1(y)
        y = self.conv1(y)

        if self.batch_norm:
            y = self.bn2(y)
        y = self.act2(y)
        y = self.conv2(y)

        y = self.scaling(y)
        return drop_path(y, self.drop_prob, self.training) + self.skip_connection(x)


class BottleNeckBlock3D(nn.Module):
    def __init__(self, in_channel_size = 256, out_channel_size = 256, downsample = False,
                 batch_norm = True, activation_function = 'nn.ReLU', layer_scaling = False,
                 drop_probability = 0.0, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv3d(
                        in_channels = in_channel_size,
                        out_channels = out_channel_size // 4,
                        kernel_size = 1)
        self.act1 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)

        self.conv2 = nn.Conv3d(
                        in_channels = out_channel_size // 4,
                        out_channels = out_channel_size // 4,
                        kernel_size = 3,
                        stride = 2 if downsample else 1,
                        padding = 1)
        self.act2 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)

        self.conv3 = nn.Conv3d(
                        in_channels = out_channel_size // 4,
                        out_channels = out_channel_size,
                        kernel_size = 1)
        self.act3 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm3d(num_features = out_channel_size // 4)
            self.bn2 = nn.BatchNorm3d(num_features = out_channel_size // 4)
            self.bn3 = nn.BatchNorm3d(num_features = out_channel_size)
        
        if downsample:
            downsample_fn = nn.AvgPool3d(kernel_size = 1, stride = 2)
            def shortcut(x):
                y = downsample_fn(x)
                zeros = torch.zeros_like(y)
                return torch.cat([y, zeros], axis = 1)
            self.skip_connection = shortcut
        elif out_channel_size != in_channel_size:
            def shortcut(x : torch.Tensor):
                zeros = torch.zeros((x.shape[0], out_channel_size - in_channel_size, *x.shape[2:]),
                                    device = x.device, dtype = x.dtype)
                return torch.cat([x, zeros], axis = 1)
            self.skip_connection = shortcut
        else:
            self.skip_connection = nn.Identity()
        
        self.scale = LayerScale3D(out_channel_size, layer_scaling) if layer_scaling else nn.Identity()
        self.drop_prob = drop_probability
    
    def forward(self, x):
        y = self.conv1(x)
        if self.batch_norm:
            y = self.bn1(y)
        y = self.act1(y)

        y = self.conv2(y)
        if self.batch_norm:
            y = self.bn2(y)
        y = self.act2(y)

        y = self.conv3(y)
        if self.batch_norm:
            y = self.bn3(y)
        
        y = self.scale(y)
        y = self.act3(drop_path(y, self.drop_prob, self.training) + self.skip_connection(x))
        return y

class BottleNeckBlockBlock3Dv2(BottleNeckBlock3D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_norm = kwargs.get('batch_norm', True)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm3d(num_features = kwargs.get('in_channel_size', 256))
            self.bn3 = nn.BatchNorm3d(num_features = kwargs.get('out_channel_size', 256) // 4)
    
    def forward(self, x):
        y = self.bn1(x) if self.batch_norm else x
        y = self.act1(y)
        y = self.conv1(y)

        if self.batch_norm:
            y = self.bn2(y)
        y = self.act2(y)
        y = self.conv2(y)

        if self.batch_norm:
            y = self.bn3(y)
        y = self.act3(y)
        y = self.conv3(y)

        y = self.scale(y)
        return drop_path(y, self.drop_prob, self.training) + self.skip_connection(x)

class FixedSizeResNet3D(nn.Module):

    @staticmethod
    def fill_kwargs(config_dict):
        utils.fill_dict(config_dict['activation_function'])
        if 'final_activation' in config_dict:
            utils.fill_dict(config_dict['final_activation'])

    def __init__(self, size : int,
                 version : Literal['v1', 'v2'] = 'v1',
                 channel_sizes : Tuple[int] = (16, 32, 64),
                 in_channels : int = 2,
                 output_size : int = 1,
                 pooling_output_size : Union[int, Tuple[int]] = 1,
                 batch_norm : bool = False,
                 bottleneck : bool = False,
                 activation_function : utils.ConfigDict = {'nn.LeakyReLU': {'negative_slope': 0.2}},
                 final_activation : Optional[utils.ConfigDict] = None,
                 head : bool = True,
                 kernel_sizes : Union[int, Tuple[int]] = 3,
                 dilations : int = 1,
                 layer_scaling : Union[Literal[False], float] = False,
                 stochastic_depth_rate : float = 0,
                 dropout : Union[Literal[False], float] = False,
                 res_connection : Union[Literal[False], str] = 'identity',
                 *args, **kwargs):

        k = len(channel_sizes)
        assert size % k == 2, f'Size {size} is not of shape kn + 2.'
        m = (size - 2) // k

        l = 3 if bottleneck else 2
        n = (size - 2) // l
        self.head = head

        if version not in (1, 2, 'v1', 'v2'):
            raise ValueError(f'Version must be `v1` or `v2`, not `{version}`.')
        if isinstance(version, int):
            version = f'v{version}'
        self.v1 = version == 'v1'

        super().__init__()
        
        self.first_layer = nn.Conv3d(
                                in_channels = in_channels,
                                out_channels = channel_sizes[0] if not bottleneck else channel_sizes[0] // 4,
                                kernel_size = kernel_sizes,
                                stride = 1,
                                padding = 'same',
                                dilation = dilations
                                )

        self.batch_norm = batch_norm

        if version == 'v1':
            if batch_norm:
                self.bn1 = nn.BatchNorm3d(num_features = channel_sizes[0])
            self.act1 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)
        
        if not bottleneck:
            channel_sizes = (channel_sizes[0], *channel_sizes)
        else:
            channel_sizes = (channel_sizes[0] // 4, *channel_sizes)

        drop_probs = np.linspace(0, stochastic_depth_rate, num = n, endpoint = True)

        res_blocks = []
        for i in range(k):
            res_blocks.append(ResBlock3D(
                                    size = m,
                                    in_channel_size = channel_sizes[i],
                                    out_channel_size = channel_sizes[i + 1],
                                    batch_norm = batch_norm,
                                    activation_function = activation_function,
                                    downsample = False,
                                    kernel_sizes = kernel_sizes,
                                    dilations = dilations,
                                    bottleneck = bottleneck,
                                    version = version,
                                    layer_scaling = layer_scaling,
                                    drop_probabilities = drop_probs[i * m // l : (i + 1) * m // l],
                                    dropout = dropout,
                                    res_connection = res_connection
                                    )
                                )
        self.res_blocks = nn.Sequential(*res_blocks)

        if version == 'v2':
            if self.batch_norm:
                self.final_bn = nn.BatchNorm3d(num_features = channel_sizes[-1])
            self.second_to_final_act = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)

        if isinstance(pooling_output_size, int):
            self.grid_output = False
            self.global_pool = nn.AdaptiveAvgPool3d(output_size = pooling_output_size)
            if head:
                self.optional_dropout = nn.Dropout(p = dropout) if dropout else lambda x: x
                self.final_layer = nn.Linear(in_features = channel_sizes[-1], out_features = output_size)
                if final_activation is not None:
                    self.final_act = utils.create_object_from_dict(final_activation, convert_to_kwargs = True)
                else:
                    self.final_act = None
        else:
            self.grid_output = True
            self.global_pool = nn.AdaptiveAvgPool3d(output_size = pooling_output_size if len(pooling_output_size)==3 else (*pooling_output_size, 1))
    
    def forward(self, x):
        y = self.first_layer(x)

        if self.v1:
            if self.batch_norm:
                y = self.bn1(y)
            y = self.act1(y)
        
        y = self.res_blocks(y)

        if not self.v1:
            if self.batch_norm:
                y = self.final_bn(y)
            y = self.second_to_final_act(y)
        if not self.grid_output:
            y = self.global_pool(y)
            y = y.flatten(1)

            if self.head:
                y = self.optional_dropout(y)
                y = self.final_layer(y)
                if self.final_act:
                    y = self.final_act(y)
        else:
            y = self.global_pool(y)
            y = torch.squeeze(y, dim=-1)
        
        # print(y.shape)

        return y


class LayerScale3D(nn.Module):

    def __init__(self, n_channels, init_value = 1e-6, *args, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(n_channels, 1, 1, 1))
    
    def forward(self, x, *args, **kwargs):
        return self.scale * x

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