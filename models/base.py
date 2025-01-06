import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_memory_efficient import SignFunction, SignFunctionReSTE, SignFunctionNoSTE, SignFunctionXNOR


class BinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size, stride, padding, xnor=False):
        super(BinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.01, requires_grad=True)
        self.xnor = xnor

    def forward(self, x):
        if self.xnor:
            scaling_factor = torch.mean(torch.abs(self.weight)).detach()
            binary_weights = SignFunctionXNOR.apply(self.weight, scaling_factor)
        else:
            binary_weights = SignFunction.apply(self.weight)
        return F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)


class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, xnor=False):
        super(BinaryLinear, self).__init__()
        self.weight = nn.Parameter(torch.rand((out_features, in_features)) * 0.01, requires_grad=True)
        self.xnor = xnor

    def forward(self, x):
        if self.xnor:
            scaling_factor = torch.mean(torch.abs(self.weight)).detach()
            b_weight = SignFunctionXNOR.apply(self.weight, scaling_factor)
        else:
            b_weight = SignFunction.apply(self.weight)
        
        return F.linear(x, b_weight, None)


class BinaryActivation(nn.Module):
    def __init__(self, approx):
        super(BinaryActivation, self).__init__()
        self.approx = approx
        self.scale = torch.tensor(1).float()

    def forward(self, x):
        if self.approx == "STE":
            out = SignFunction.apply(x)
        elif self.approx == "ApproxSign":
            out = ApproxSign.apply(x)
        elif self.approx == "ReSTE":
            out = SignFunctionReSTE.apply(x, self.scale)
        else:
            raise NotImplementedError()
        return out


class ApproxSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        result = sign_function(x)
        return result

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        mask1 = (-1 <= x) & (x < 0)
        mask2 = (0 <= x) & (x <= 1)
        tmp = torch.zeros_like(grad)
        tmp[mask1] = 2 + 2*x[mask1]
        tmp[mask2] = 2 - 2*x[mask2]
        x_grad = grad * tmp
        return x_grad
