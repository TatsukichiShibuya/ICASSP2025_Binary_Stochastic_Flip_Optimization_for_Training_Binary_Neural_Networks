import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import models.base as base
import models.base_memory_efficient as base_me
from models.AdamBNN import BinaryActivation, conv3x3, conv1x1, binaryconv3x3, binaryconv1x1, firstconv3x3, LearnableBias, BasicBlock

with open(".strage_path.txt") as f:
    STRAGE_PATH = f.read()

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 4 + [1024] * 1 # 6 2


AvgPool2d =  nn.AvgPool2d  # base_me.AvgPool2d
BatchNorm =  nn.SyncBatchNorm
BinaryConv = base.BinaryConv  # base_me.BinaryConv
BinaryLinear = base.BinaryLinear  # base_me.BinaryLinear
BinaryActivation = base.BinaryActivation  # base_me.BinaryActivation
PReLU = nn.PReLU  # base_me.PReLU


class Shift(nn.Module):
    def __init__(self):
        super(Shift, self).__init__()
        self.pad1 = nn.ZeroPad2d(padding=(0, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        self.pad3 = nn.ZeroPad2d(padding=(1, 0, 0, 0))
        self.pad4 = nn.ZeroPad2d(padding=(0, 1, 0, 0))

    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim = 1)

        x1 = torch.roll(x1, 1, dims=2)#[:,:,1:,:]
        #x1 = self.pad1(x1)
        x2 = torch.roll(x2, -1, dims=2)#[:,:,:-1,:]
        #x2 = self.pad2(x2)
        x3 = torch.roll(x3, 1, dims=3)#[:,:,:,1:]
        #x3 = self.pad3(x3)
        x4 = torch.roll(x4, -1, dims=3)#[:,:,:,:-1]
        #x4 = self.pad4(x4)
        
        x = torch.cat([x1, x2, x3, x4], 1)

        return x

class GlobalShift(nn.Module):
    def __init__(self):
        super(GlobalShift, self).__init__()
        self.pad1 = nn.ZeroPad2d(padding=(0, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        self.pad3 = nn.ZeroPad2d(padding=(1, 0, 0, 0))
        self.pad4 = nn.ZeroPad2d(padding=(0, 1, 0, 0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2, x3, x4 = x.chunk(4, dim = 1)

        x1 = torch.roll(x1, 3, dims=2)#[:,:,1:,:]
        #x1 = self.pad1(x1)
        x2 = torch.roll(x2, -3, dims=2)#[:,:,:-1,:]
        #x2 = self.pad2(x2)
        x3 = torch.roll(x3, 3, dims=3)#[:,:,:,1:]
        #x3 = self.pad3(x3)
        x4 = torch.roll(x4, -3, dims=3)#[:,:,:,:-1]
        #x4 = self.pad4(x4)
        
        x = torch.cat([x1, x2, x3, x4], 1)

        return x

class ShiftBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1, padding=0, xnor=False, approx=None):
        super(ShiftBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.shift1 = Shift()
        self.shift2 = GlobalShift()
        self.binary_activation = BinaryActivation(approx=approx)

        self.binary_conv = BinaryConv(inplanes, planes, stride=stride, kernel_size=kernel_size, padding=padding, xnor=xnor)
        self.binary_conv1 = BinaryConv(inplanes, planes, stride=stride, kernel_size=kernel_size, padding=padding, xnor=xnor)
        self.binary_conv2 = BinaryConv(inplanes, planes, stride=stride, kernel_size=kernel_size, padding=padding, xnor=xnor)

        self.bn1 = BatchNorm(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)

        out1 = self.binary_conv(out)

        out2 = self.shift1(out)
        out2 = self.binary_conv1(out2)

        out3 = self.shift2(out)
        out3 = self.binary_conv2(out3)

        out = self.bn1(out1 + out2 + out3)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out

class PointBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1, padding=0, xnor=False, approx=None):
        super(PointBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation(approx=approx)
        self.binary_conv = BinaryConv(inplanes, planes, stride=stride, kernel_size=kernel_size, padding=padding, xnor=xnor)
        self.bn1 = BatchNorm(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out


class Block(nn.Module):
    expansion = 1

    def __init__(self, channel, xnor, approx):
        super(Block, self).__init__()
        mlp_layer = []

        mlp_layer.append(ShiftBlock(channel, channel, xnor=xnor, approx=approx))
        mlp_layer.append(ShiftBlock(channel, channel, xnor=xnor, approx=approx))
        mlp_layer.append(ShiftBlock(channel, channel, xnor=xnor, approx=approx))
        mlp_layer.append(PointBlock(channel, channel, xnor=xnor, approx=approx))

        self.mlp_layer = nn.Sequential(*mlp_layer)

    def forward(self, x):
        x = self.mlp_layer(x)
        return x


class BCDNet(nn.Module):
    def __init__(self, num_classes=1000, xnor=False, approx=None):
        super(BCDNet, self).__init__()

        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2, approx=approx))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1, approx=approx))

        self.mlp_layer = self.build(3, 1024, xnor, approx=approx)

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        
        layers = [nn.Linear(1024, num_classes, bias=False), BatchNorm(num_classes, affine=False, track_running_stats=False)]
        self.fc = nn.Sequential(*layers)
    
    def build(self, num, channel, xnor, approx):
        mlp_layer = []
        for i in range(num):
            mlp_layer.append(Block(channel, xnor, approx=approx))
        return nn.Sequential(*mlp_layer)

    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)
        
        x = self.mlp_layer(x)

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        
        x_res = self.fc(x)
        return x_res
    
    def update_rectified_scale(self, t):
        assert 0<=t<=1
        for i in range(1,11):
            self.feature[i].binary_activation.scale = torch.tensor(1+t*2).float().to()

def get_bcdnet(out_dim, xnor=False, approx=None):
    
    model = BCDNet(num_classes=out_dim, xnor=xnor, approx=approx)
    
    # load checkpoint
    state_dict = torch.load(os.path.join(STRAGE_PATH, "model/BCDNet.pth.tar"))["model"]
    checkpoint = OrderedDict()
    for key, value in state_dict.items():
        checkpoint[key.split("module.")[-1]] = value

    # initialize parameters
    for name, param in model.named_parameters():
        if name in checkpoint.keys():
            param.data = checkpoint[name]

    return model