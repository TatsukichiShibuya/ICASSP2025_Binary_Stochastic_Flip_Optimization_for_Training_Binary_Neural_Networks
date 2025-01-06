import os
import pdb
import pickle
import torch
import torch.nn as nn
from models.base import *


class MLP(nn.Module):
    def __init__(self, in_dim=12288, hid_dim=8192, out_dim=200, depth=4, approx=None):
        super(MLP, self).__init__()
        self.depth = depth

        layers = []
        for d in range(depth-1):
            dim1 = in_dim if d==0 else hid_dim
            dim2 = hid_dim
            layers += [BinaryLinear(dim1, dim2),
                       nn.SyncBatchNorm(dim2, affine=False, track_running_stats=False),
                       BinaryActivation(approx=approx)]
        self.feature = nn.Sequential(*layers)

        dim1 = hid_dim if depth>1 else in_dim
        out_layer = [BinaryLinear(dim1, out_dim),
                     nn.SyncBatchNorm(out_dim, affine=False, track_running_stats=False)]
        self.classifier = nn.Sequential(*out_layer)

        self._initialize_weights()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.feature(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BinaryLinear) or (isinstance(m, nn.Linear)):
                nn.init.normal_(m.weight, 0, 1e-2)

            elif isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def update_rectified_scale(self, t):
        assert 0<=t<=1
        for i in range(self.depth-1):
            self.feature[3*i+2].scale = torch.tensor(1+t*2).float()


def get_mlp(in_shape=(64,3), hid_dim=256, out_dim=200, depth=4, approx=None):
    in_dim = in_shape[0]**2 * in_shape[1]
    
    model = MLP(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, depth=depth, approx=approx)
    
    return model