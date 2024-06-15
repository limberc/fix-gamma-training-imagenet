import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math

WarpWeight = 1.0 / 8
n = WarpWeight
_warp_kernel0 = [[0, 0, 0, 0, 0],
                 [0, 0, n, n, n],
                 [0, n, n, n, 0],
                 [n, n, n, 0, 0],
                 [0, 0, 0, 0, 0]]

_warp_kernel1 = [[0, 0, 0, n, 0],
                 [0, 0, n, n, 0],
                 [0, n, n, n, 0],
                 [0, n, n, 0, 0],
                 [0, n, 0, 0, 0]]

_warp_kernel2 = [[0, n, 0, 0, 0],
                 [0, n, n, 0, 0],
                 [0, n, n, n, 0],
                 [0, 0, n, n, 0],
                 [0, 0, 0, n, 0]]

_warp_kernel3 = [[0, 0, 0, 0, 0],
                 [n, n, n, 0, 0],
                 [0, n, n, n, 0],
                 [0, 0, n, n, n],
                 [0, 0, 0, 0, 0]]
_warp_kernel = [_warp_kernel0, _warp_kernel1, _warp_kernel2, _warp_kernel3]


class WarpConv(nn.Conv2d):
    def __init__(self, channels, mode: int):
        super().__init__(channels, channels, kernel_size=5, stride=1, padding=2, bias=False)
        kernel = np.array(_warp_kernel[mode], dtype=np.float32)

        kernel = kernel[np.newaxis, np.newaxis] * np.ones((channels, channels, 5, 5), dtype=np.float32)
        self.weight.data = torch.from_numpy(kernel)
        self.weight.requires_grad = False


class WarpLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        for i in range(4):
            self.add_module(f'warp_conv{i}', WarpConv(channels, mode=i))

    def forward(self, x):
        part0 = x * 9.0 / 8.0
        part1 = F.avg_pool2d(x, 3, stride=1, padding=1) * 9.0 / 8.0
        part2 = self.warp_conv0(x)
        part3 = self.warp_conv1(x)
        part4 = self.warp_conv2(x)
        part5 = self.warp_conv3(x)
        # return torch.cat([part0, part1, part2, part3, part4, part5], dim=1)
        return part0 + part1 + part2 + part3 + part4 + part5


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding
    bias=True
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution
    bias=True
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FixGammaBN(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, mode=9):
        super(FixGammaBN, self).__init__(num_features, eps, momentum, affine=True,
                                         track_running_stats=True)
        self.num_features = num_features
        if not mode:
            base = [2 ** -2, 2 ** -3]
        elif mode == 1:
            base = [2 ** -2, 2 ** -3, 2 ** -3, 2 ** -3, 2 ** -4, 2 ** -4, 2 ** -4, 2 ** -5]
        elif mode == 8:
            base = [ 1.0 ]
        elif mode == 9:
            base = [ 1.0 ] # Init -9.0?
        elif mode == 10:
            base = [ 0.0 ]
        nn.init.ones_(self.weight)
        self.weight.requires_grad = False

        self.gamma = nn.Parameter(
            torch.tensor(data=base * (num_features // len(base))).reshape((1, num_features, 1, 1)))
            #torch.tensor(data=base).reshape((1, 1, 1, 1)))
        #self.beta = nn.Parameter(
        #    torch.tensor(data=base * (num_features // len(base))).reshape((1, num_features, 1, 1)))
        #self.beta.requires_grad = True
        if mode != 9 and mode != 10 and mode != 8:
            self.gamma.requires_grad = False
        else:
            self.gamma.requires_grad = True
            #print("Gamma Trainable")
        #         self.proxybias = nn.Parameter(
        #             torch.tensor(data=[ 0.0 for i in range(num_features)]))
        #self.bias = self.bias * np.log(self.num_features) / np.log(2)
        #self.fixscale = torch.tensor(math.log2(self.num_features))
        #self.fixscale.requires_grad = False
    def forward(self, input):
        '''
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        out = F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, 
            self.bias * self.fixscale,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        '''
        out = super(nn.BatchNorm2d, self).forward(input)
        out *= self.gamma
        return out

class FixGammaLN(nn.LayerNorm):
    def __init__(self, normalized_shape, mode=9):
        super(FixGammaLN, self).__init__(normalized_shape, eps=1e-5, elementwise_affine=True)
        self.normalized_shape  = normalized_shape
        if not mode:
            base = [2 ** -2, 2 ** -3]
        elif mode == 1:
            base = [2 ** -2, 2 ** -3, 2 ** -3, 2 ** -3, 2 ** -4, 2 ** -4, 2 ** -4, 2 ** -5]
        elif mode == 9:
            base = [ 1.0 ] # Init -9.0?
        elif mode == 10:
            base = [ 0.0 ]
        nn.init.ones_(self.weight)
        self.weight.requires_grad = False

        self.gamma = nn.Parameter(
            torch.tensor(data=base).reshape((1, 1, 1, 1)))
        #self.beta = nn.Parameter(
        #    torch.tensor(data=base * (num_features // len(base))).reshape((1, num_features, 1, 1)))
        #self.beta.requires_grad = True
        if mode != 9 and mode != 10:
            self.gamma.requires_grad = False
        else:
            self.gamma.requires_grad = True
    def forward(self, input):
        out = super(nn.LayerNorm, self).forward(input)
        out *=  self.gamma
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = FixGammaBN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = FixGammaBN(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, warp=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        if warp:
            self.warp = WarpLayer(planes)
        else:
            self.warp = None
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = FixGammaBN(planes, mode=9)
        self.conv2 = conv3x3(planes, planes, stride)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = FixGammaBN(planes, mode=9)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        # self.warp = WarpLayer(planes * self.expansion)
        #self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3 = FixGammaBN(planes * self.expansion, mode=10)
        self.relu = nn.ReLU6(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.warp:
            out = self.warp(out)
        out = self.bn1(out)
        out = self.relu(out) 

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out) 

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out) 

        return out


class BottleneckWarp(Bottleneck):
    expansion = 3

    def __init__(self, inplanes, planes, stride=1, downsample=None, warp=True):
        super(BottleneckWarp, self).__init__(inplanes, planes, stride, downsample, warp)


class SEModule(nn.Module):
    def __init__(self, inplanes, reduction):
        super(SEModule, self).__init__()
        planes = inplanes // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = conv1x1(inplanes, planes)
        self.relu = nn.ReLU6(inplace=True)
        self.conv2 = conv1x1(planes, inplanes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        module_input = input
        out = self.avg_pool(input)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return module_input * out


class SEResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, warp=False):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        if warp:
            self.warp = WarpLayer(planes)
        else:
            self.warp = None
        self.bn1 = FixGammaBN(planes, mode=9)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False, stride=stride)
        self.bn2 = FixGammaBN(planes, mode=9)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = FixGammaBN(planes * 4, mode=9)
        self.relu = nn.ReLU6(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.warp:
            out = self.warp(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        #print(out.size(),self.se_module(out).size(),residual.size())
        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNetBottleneckWarp(SEResNetBottleneck):
    expansion = 3

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, warp=True):
        super(SEResNetBottleneckWarp, self).__init__(inplanes, planes, groups, reduction, stride, downsample, warp)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = FixGammaBN(64, mode=8)
        self.relu = nn.ReLU6(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Normalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.xavier_normal_(m.weight, gain=1.0)
            # elif isinstance(m, FixGammaBN):
            # nn.init.constant_(m.weight, 1)
            # nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #print(stride,self.inplanes,planes*block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                FixGammaBN(planes * block.expansion, mode=8),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SENet(nn.Module):
    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', conv3x3(in_planes=3, out_planes=64, stride=2)),
                ('bn1', FixGammaBN(64, mode=9)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', conv3x3(in_planes=64, out_planes=64)),
                ('bn2', FixGammaBN(num_features=64, mode=9)),
                ('relu2', nn.ReLU6(inplace=True)),
                ('conv3', conv3x3(in_planes=64, out_planes=inplanes)),
                ('bn3', FixGammaBN(num_features=inplanes, mode=9)),
                ('relu3', nn.ReLU6(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', FixGammaBN(inplanes, mode=9)),
                ('relu1', nn.ReLU6(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            stride=1,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.MaxPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #print(stride,self.inplanes,planes*block.expansion)
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                FixGammaBN(planes * block.expansion,mode=9),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)


    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def get_warp_bottleneck(warp: bool, se: bool):
    if se:
        if not warp:
            return SEResNetBottleneck
        else:
            return SEResNetBottleneckWarp
    else:
        if not warp:
            return Bottleneck
        else:
            return BottleneckWarp


def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


def resnet50(warp=False, half=False):
    block = get_warp_bottleneck(warp, False)
    model = ResNet(block, [3, 4, 6, 3])
    if half:
        model = model.half()
        print("Creating Half Model ... ")
    return model


def resnet101(warp=False, half=False):
    block = get_warp_bottleneck(warp, False)
    model = ResNet(block, [3, 4, 23, 3])
    if half:
        model = model.half()
        print("Creating Half Model ... ")
    return model


def se_resnet50(warp=False, half=False):
    block = get_warp_bottleneck(warp, True)
    model = SENet(block, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if half:
        model = model.half()
        print("Creating Half Model ... ")
    return model


def se_resnet68(warp=False, half=False):
    block = get_warp_bottleneck(warp, True)
    print(block)
    model = SENet(block, [3, 3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=True,
                  downsample_kernel_size=1, downsample_padding=0)
    #print(model)
    if half:
        model = model.half()
        print("Creating Half Model ... ")
    return model


def se_resnet101(warp=False, half=False):
    block = get_warp_bottleneck(warp, True)
    model = SENet(block, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if half:
        model = model.half()
        print("Creating Half Model ... ")
    return model


if __name__ == '__main__':
    model = resnet50()

