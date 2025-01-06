import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import models.base as base
import models.base_memory_efficient as base_me

with open(".strage_path.txt") as f:
    STRAGE_PATH = f.read()

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2


AvgPool2d =  nn.AvgPool2d  # base_me.AvgPool2d
BatchNorm =  nn.SyncBatchNorm
BinaryConv = base.BinaryConv  # base_me.BinaryConv
BinaryLinear = base.BinaryLinear  # base_me.BinaryLinear
BinaryActivation = base.BinaryActivation  # base_me.BinaryActivation
PReLU = nn.PReLU  # base_me.PReLU


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def binaryconv3x3(in_planes, out_planes, stride=1, xnor=False):
    """3x3 convolution with padding"""
    return BinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, xnor=xnor)


def binaryconv1x1(in_planes, out_planes, stride=1, xnor=False):
    """1x1 convolution"""
    return BinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, xnor=xnor)


class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = BatchNorm(oup)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return out


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, xnor=False, approx=None):
        super(BasicBlock, self).__init__()

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3 = binaryconv3x3(inplanes, inplanes, stride=stride, xnor=xnor)
        self.bn1 = BatchNorm(inplanes)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = binaryconv1x1(inplanes, planes, xnor=xnor)
            self.bn2 = BatchNorm(planes)
        else:
            self.binary_pw_down1 = binaryconv1x1(inplanes, inplanes, xnor=xnor)
            self.binary_pw_down2 = binaryconv1x1(inplanes, inplanes, xnor=xnor)
            self.bn2_1 = BatchNorm(inplanes)
            self.bn2_2 = BatchNorm(inplanes)

        self.move22 = LearnableBias(planes)
        self.prelu2 = PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation = BinaryActivation(approx=approx)
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = AvgPool2d(2)

    def forward(self, x):

        out1 = self.move11(x)

        out1 = self.binary_activation(out1)
        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)
        out2 = self.binary_activation(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2


class AdamBNN(nn.Module):
    def __init__(self, num_classes, xnor, approx):
        super(AdamBNN, self).__init__()
        
        # Feature extractor
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2, xnor=xnor, approx=approx))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1, xnor=xnor, approx=approx))
        self.pool1 = nn.AdaptiveAvgPool2d(1)

        # Header
        layers = [nn.Linear(1024, num_classes, bias=False), 
                    BatchNorm(num_classes, affine=False, track_running_stats=False)]
        self.fc = nn.Sequential(*layers)
        
        self._initialize_weights()

    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def update_rectified_scale(self, t):
        assert 0<=t<=1
        for i in range(1,14):
            self.feature[i].binary_activation.scale = torch.tensor(1+t*2).float().to()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BatchNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def get_adambnn(out_dim, xnor=False, approx=None, freeze_num=0):

    model = AdamBNN(num_classes=out_dim, xnor=xnor, approx=approx)

    # load checkpoint
    state_dict = torch.load(os.path.join(STRAGE_PATH, "model/AdamBNN.mobilenet.pth.tar"))
    state_dict["state_dict_new"] = OrderedDict()
    for key, value in state_dict["state_dict"].items():
        state_dict["state_dict_new"][key.split("module.")[-1]] = value
    checkpoint = state_dict["state_dict_new"]

    # initialize parameters
    for name, param in model.named_parameters():
        if name in checkpoint.keys():
            param.data = checkpoint[name]

    # freeze parameters
    for i in range(freeze_num):
        for name, param in model.feature[i].named_parameters():
            param.requires_grad = False

    return model