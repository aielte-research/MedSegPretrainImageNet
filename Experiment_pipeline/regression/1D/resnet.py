import numpy as np
from typing import Literal, Optional, Tuple, Union
import torch
from torch import nn

import utils
from segmentation.models import drop_path

class ResBlock1D(nn.Module):

    @staticmethod
    def fill_kwargs(config_dict):
        if 'activation_function' in config_dict:
            utils.fill_dict(config_dict['activation_function'])
    
    def __init__(self, size, in_channel_size = 16, out_channel_size = 16,
                 bottleneck = False, kernel_sizes = 3, dilations = 1, version = 'v1',
                 layer_scaling = False, drop_probabilities = None, *args, **kwargs):
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
            basic_block = BasicBlock1D if not bottleneck else BottleNeckBlock1D
        else:
            basic_block = BasicBlock1Dv2 if not bottleneck else BottleNeckBlockBlock1Dv2
        for i, drop_prob in enumerate(drop_probabilities):
            layers.append(basic_block(
                            in_channel_size = in_channel_size if i == 0 else out_channel_size,
                            out_channel_size = out_channel_size,
                            downsample = downsample and i == 0,
                            kernel_sizes = kernel_sizes,
                            dilations = dilations,
                            layer_scaling = layer_scaling,
                            drop_probability = drop_prob,
                            *args, **kwargs
                        )
            )
        self.res_block = nn.Sequential(*layers)
    
    def forward(self, x):
        y = self.res_block(x)
        return y

class BasicBlock1D(nn.Module):

    @staticmethod
    def fill_kwargs(config_dict):
        utils.fill_dict(config_dict['activation_function'])
    
    def __init__(self, in_channel_size = 16, out_channel_size = 16, downsample = False,
                 batch_norm = True, kernel_sizes = 3, dilations = 1, activation_function = 'torch.nn.ReLU',
                 layer_scaling = False, drop_probability = 0.0, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv1d(
                        in_channels = in_channel_size,
                        out_channels = out_channel_size,
                        kernel_size = kernel_sizes,
                        stride = 2 if downsample else 1,
                        padding = ((kernel_sizes - 1) * dilations + 1) // 2,
                        dilation = dilations
                        )
        self.act1 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)

        self.conv2 = nn.Conv1d(
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
            self.bn1 = nn.BatchNorm1d(num_features = out_channel_size)
            self.bn2 = nn.BatchNorm1d(num_features = out_channel_size)
        
        if downsample:
            downsample_fn = nn.AvgPool1d(kernel_size = 1, stride = 2)
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
        
        self.scaling = LayerScale1D(out_channel_size, layer_scaling) if layer_scaling else nn.Identity()
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
        y = self.act2(drop_path(y, self.drop_prob, self.training) + self.skip_connection(x))
        return y

class BasicBlock1Dv2(BasicBlock1D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_norm = kwargs.get('batch_norm', True)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(num_features = kwargs.get('in_channel_size', 16))
    
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


class BottleNeckBlock1D(nn.Module):
    def __init__(self, in_channel_size = 256, out_channel_size = 256, downsample = False,
                 batch_norm = True, activation_function = 'torch.nn.ReLU', layer_scaling = False,
                 drop_probability = 0.0, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv1d(
                        in_channels = in_channel_size,
                        out_channels = out_channel_size // 4,
                        kernel_size = 1)
        self.act1 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)

        self.conv2 = nn.Conv1d(
                        in_channels = out_channel_size // 4,
                        out_channels = out_channel_size // 4,
                        kernel_size = 3,
                        stride = 2 if downsample else 1,
                        padding = 1)
        self.act2 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)

        self.conv3 = nn.Conv1d(
                        in_channels = out_channel_size // 4,
                        out_channels = out_channel_size,
                        kernel_size = 1)
        self.act3 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(num_features = out_channel_size // 4)
            self.bn2 = nn.BatchNorm1d(num_features = out_channel_size // 4)
            self.bn3 = nn.BatchNorm1d(num_features = out_channel_size)
        
        if downsample:
            downsample_fn = nn.AvgPool1d(kernel_size = 1, stride = 2)
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
        
        self.scale = LayerScale1D(out_channel_size, layer_scaling) if layer_scaling else nn.Identity()
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

class BottleNeckBlockBlock1Dv2(BottleNeckBlock1D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_norm = kwargs.get('batch_norm', True)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(num_features = kwargs.get('in_channel_size', 256))
            self.bn3 = nn.BatchNorm1d(num_features = kwargs.get('out_channel_size', 256) // 4)
    
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

class FixedSizeResNet1D(nn.Module):

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
                 batch_norm : bool = False,
                 bottleneck : bool = False,
                 activation_function : utils.ConfigDict = {'torch.nn.LeakyReLU': {'negative_slope': 0.2}},
                 final_activation : Optional[utils.ConfigDict] = None,
                 head : bool = True,
                 kernel_sizes : int = 3,
                 dilations : int = 1,
                 layer_scaling : Union[Literal[False], float] = False,
                 stochastic_depth_rate : float = 0,
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
        
        self.first_layer = nn.Conv1d(
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
                self.bn1 = nn.BatchNorm1d(num_features = channel_sizes[0])
            self.act1 = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)
        
        if not bottleneck:
            channel_sizes = (channel_sizes[0], *channel_sizes)
        else:
            channel_sizes = (channel_sizes[0] // 4, *channel_sizes)

        drop_probs = np.linspace(0, stochastic_depth_rate, num = n, endpoint = True)

        res_blocks = []
        for i in range(k):
            res_blocks.append(ResBlock1D(
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
                                    drop_probabilities = drop_probs[i * m // l : (i + 1) * m // l]
                                    )
                                )
        self.res_blocks = nn.Sequential(*res_blocks)

        if version == 'v2':
            if self.batch_norm:
                self.final_bn = nn.BatchNorm1d(num_features = channel_sizes[-1])
            self.second_to_final_act = utils.create_object_from_dict(activation_function, convert_to_kwargs = True)
        
        self.global_pool = nn.AdaptiveAvgPool1d(output_size = 1)
        if head:
            self.final_layer = nn.Linear(in_features = channel_sizes[-1], out_features = output_size)
            if final_activation is not None:
                self.final_act = utils.create_object_from_dict(final_activation, convert_to_kwargs = True)
            else:
                self.final_act = None
    
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

        y = self.global_pool(y)

        if self.head:
            y = y.squeeze()

            y = self.final_layer(y)
            if self.final_act:
                y = self.final_act(y)
                
        return y


class LayerScale1D(nn.Module):

    def __init__(self, n_channels, init_value = 1e-6, *args, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(n_channels, 1))
    
    def forward(self, x, *args, **kwargs):
        return self.scale * x
