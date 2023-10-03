import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial

class Conv2d_gaussian(torch.nn.Conv2d):

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        # torch.nn.init.kaiming_normal_(self.weight, a= math.sqrt(5))
        #W has shape out, in, h, w
        torch.nn.init.normal_(self.weight, 0, np.sqrt(2)/np.sqrt(self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            # print(fan_in)
            if fan_in != 0:
                # bound = 0 * 1 / math.sqrt(fan_in)
                # torch.nn.init.uniform_(self.bias, -bound, bound)
                # torch.nn.init.uniform_(self.bias, -bound, bound)
                torch.nn.init.normal_(self.bias, 0, .1)    

def build_conv2d_gaussian(in_channels, out_channels, kernel=3, padding=1, mean=None, std=None):
    layer = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
    if mean is None:
        mean = 0
    if std is None:
        std = np.sqrt(2)/np.sqrt(layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3])
    # print('Initializing Conv. Mean=%.2f, std=%.2f'%(mean, std))
    torch.nn.init.normal_(layer.weight, mean, std)
    torch.nn.init.normal_(layer.bias, 0, .1)
    return layer

class GaussianLinear_old(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, funny = False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GaussianLinear, self).__init__()
        self.funny = funny
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # torch.nn.init.kaiming_normal_(self.weight, a=1 * np.sqrt(5))
        torch.nn.init.normal_(self.weight, 0, np.sqrt(2)/np.sqrt(self.in_features))
        # torch.nn.init.normal_(self.weight, 0, 3/np.sqrt(self.in_features))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(self.bias, -bound, bound)
            torch.nn.init.normal_(self.bias, 0, .1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class GaussianLinear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, funny = False, mu=None, sigma=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GaussianLinear, self).__init__()
        self.funny = funny
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(mu, sigma)

    def reset_parameters(self, mu=None, sigma=None) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # torch.nn.init.kaiming_normal_(self.weight, a=1 * np.sqrt(5))
        if mu is None:
            mu = 0
        if sigma is None:
            sigma = np.sqrt(2)/np.sqrt(self.in_features)
        torch.nn.init.normal_(self.weight, mu, sigma)
        # torch.nn.init.normal_(self.weight, 0, 3/np.sqrt(self.in_features))
        if self.bias is not None:
            # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(self.bias, -bound, bound)
            torch.nn.init.normal_(self.bias, 0, .1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ConvNet_wide(nn.Module):
    def __init__(self, input_dim, n_random_features, mu=None, sigma=None, k = 4, net_width = 128, net_depth = 3,
                 net_act = 'relu', net_norm = 'none', net_pooling = 'avgpooling', im_size = (32,32), chopped_head = False):
        self.k = k
        # print('Building Conv Model')
        super().__init__()
        
        # net_depth = 1
        self.features, shape_feat = self._make_layers(input_dim, net_width, net_depth, net_norm,
                                                      net_act, net_pooling, im_size, mu, sigma)
        # print(shape_feat)
        self.chopped_head = chopped_head
        if not chopped_head:
            num_feat = shape_feat[0] *shape_feat[1]*shape_feat[2]
            self.classifier = GaussianLinear(num_feat, n_random_features)

    def forward(self, x):
        out = self.features(x)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        # print(out.size())
        if not self.chopped_head:
            out = self.classifier(out)
        # print(out.size())
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'gelu':
            return nn.SiLU()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size, mu, sigma):
        k = self.k
        
        layers = []
        in_channels = channel
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [build_conv2d_gaussian(in_channels, int(k * net_width), 3, 
                                             1, mean=mu, std=sigma)]
            shape_feat[0] = int(k* net_width)
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = int(k * net_width)
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, input_dim, n_random_features, mu=None, sigma=None, k = 4, n_channels = 128, net_depth = 3, 
                 net_act = 'relu', net_norm = 'none', net_pooling = 'maxpooling', im_size = (32,32), chopped_head = False):
        super(ResNet, self).__init__()
        block = BasicBlock
        layers = [2, 2, 2, 2]
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(input_dim, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, n_random_features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class Linear_wide(nn.Module):
    def __init__(self, n_channels, input_dim, n_random_features, net_depth=3, net_act='relu', mu=None, sigma=None):
        super().__init__()
        # print(type(n_channels), type(net_depth))
        self.net = self._make_layers(n_channels, input_dim, n_random_features, net_depth, net_act, mu, sigma)
        
    def _make_layers(self, n_channels, input_feature_size, output_feature_size, net_depth, net_act, mean, std):        
        layers = []
        in_channels = input_feature_size
        for d in range(net_depth):
            layers += [GaussianLinear(in_features = in_channels,out_features=n_channels, mu=mean, sigma=std)]
            in_channels = n_channels
            layers += [self._get_activation(net_act)]
        
        layers += [GaussianLinear(in_features = in_channels,out_features=output_feature_size)]

        return nn.Sequential(*layers)
    
    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'gelu':
            return nn.SiLU()
        else:
            exit('unknown activation function: %s'%net_act)
    
    def forward(self, x):
        out = self.net(x)
        return out
    
def build_model(model_type, **args):
    print(args)
    # print(*args)
    if model_type == 'linear':
        model = partial(Linear_wide, **args)
    elif model_type == 'conv':
        model = partial(ConvNet_wide, **args)
    elif model_type == "resnet":
        model = partial(ResNet, **args)
    else:
        model = None
        print("Model Type '%s' NOT Implemented!")
    return model
