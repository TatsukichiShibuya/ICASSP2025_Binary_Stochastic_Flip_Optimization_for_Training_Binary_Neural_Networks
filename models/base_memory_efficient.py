import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from functools import reduce

#cudnn_convolution = load(name="cudnn_convolution", sources=["cudnn_convolution.cpp"], verbose=True)

def sign_function(x):
    return torch.sign(torch.sign(x) + 0.5)


def BoolEncoder(x):
    assert x.dtype==torch.bool
    shape = x.shape
    temp = x.reshape(-1).to(torch.uint8)
    temp = torch.cat((torch.zeros((8-temp.shape[0]%8)%8, dtype=x.dtype, device=x.device), temp))
    temp = temp.reshape(-1,8)
    weights = torch.tensor([2**i for i in reversed(range(8))], dtype=torch.uint8, device=x.device)
    x_code = torch.sum(temp * weights, dim=1, dtype=torch.uint8)
    return x_code, shape


def BoolDecoder(x, shape):
    assert x.dtype==torch.uint8
    temp = x.unsqueeze(-1)
    mask = torch.tensor([1 << i for i in reversed(range(8))], dtype=torch.uint8, device=x.device).unsqueeze(0)
    x_bool = (temp & mask).ne(0)
    x_bool.view(-1)
    length = reduce(lambda x, y: x * y, shape)
    return x_bool.view(-1)[-length:].reshape(shape)

#########################################################################################################

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return avg_pool2d.apply(x, self.kernel_size)


class avg_pool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size):
        output = F.avg_pool2d(x, kernel_size)
        ctx.kernel_size = kernel_size
        return output

    @staticmethod
    def backward(ctx, grad):
        kernel_size = ctx.kernel_size
        grad_x = F.interpolate(grad, scale_factor=kernel_size, mode='nearest') / (kernel_size**2)
        return grad_x, None

#########################################################################################################

class PReLU(nn.Module):
    def __init__(self, num_parameters):
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = nn.parameter.Parameter(torch.ones(num_parameters)*0.25, requires_grad=True)

    def forward(self, x):
        return prelu.apply(x, self.weight)


class prelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        mask_code, mask_shape = BoolEncoder(x>0)
        ctx.save_for_backward(mask_code, weight)
        ctx.mask_shape = mask_shape
        return F.prelu(x, weight)

    @staticmethod
    def backward(ctx, grad):
        mask_code, weight = ctx.saved_tensors
        mask = BoolDecoder(mask_code, ctx.mask_shape)
        
        grad_x = grad.clone()
        _w = weight.view(1,-1,1,1)
        grad_x = torch.where(mask, grad_x, grad_x * _w)

        grad_w = grad.clone()
        grad_w[mask] = 0
        grad_w[~mask] *= -1
        dims = tuple([i for i in range(grad.dim()) if i != 1])
        grad_w = grad_w.sum(dim=dims)
        return grad_x, grad_w

#########################################################################################################

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
            binary_weights = SignFunctionNoSTE.apply(self.weight)
        return conv2d.apply(x, binary_weights, self.stride, self.padding)


class conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, stride, padding, dilation=1, groups=1):
        x_code, x_shape = BoolEncoder(x==1)
        w_code, w_shape = BoolEncoder(weight==1)
        ctx.save_for_backward(x_code, w_code)
        ctx.x_shape = x_shape
        ctx.w_shape = w_shape
        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups
        }
        cudnn_convolution.convolution(x, weight, None, stride, padding, dilation, groups, False, False)

    @staticmethod
    def backward(ctx, grad):
        x_code, w_code = ctx.saved_tensors
        x = BoolDecoder(x_code, ctx.x_shape).to(torch.float32)*2-1
        weight = BoolDecoder(w_code, ctx.w_shape).to(torch.float32)*2-1
        grad_x = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_x = cudnn_convolution.convolution_backward_input(x.shape, weight, grad, conf["stride"], conf["padding"], conf["dilation"], conf["groups"], False, False, False)
        if ctx.needs_input_grad[1]:
            grad_weight = cudnn_convolution.convolution_backward_weight(x, weight.shape, grad, conf["stride"], conf["padding"], conf["dilation"], conf["groups"], False, False, False)
        return grad_x, grad_weight, None, None, None, None

#########################################################################################################

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
            b_weight = SignFunctionNoSTE.apply(self.weight)
        
        return linear.apply(x, b_weight)


class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        output = torch.matmul(x, weight.T)
        x_code, x_shape = BoolEncoder(x==1)
        w_code, w_shape = BoolEncoder(weight==1)
        ctx.save_for_backward(x_code, w_code)
        ctx.x_shape = x_shape
        ctx.w_shape = w_shape
        return output

    @staticmethod
    def backward(ctx, grad):
        x_code, w_code = ctx.saved_tensors
        x = BoolDecoder(x_code, ctx.x_shape).to(torch.float32)*2-1
        weight = BoolDecoder(w_code, ctx.w_shape).to(torch.float32)*2-1
        grad_x = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(grad, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad.T, x)
        return grad_x, grad_weight

#########################################################################################################

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


class SignFunctionNoSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = sign_function(x)
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad


class SignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mask_code, mask_shape = BoolEncoder(torch.abs(x) <= 1)
        ctx.save_for_backward(mask_code)
        ctx.mask_shape = mask_shape
        result = torch.sign(x)
        return result

    @staticmethod
    def backward(ctx, grad):
        mask_code, = ctx.saved_tensors
        mask = BoolDecoder(mask_code, ctx.mask_shape)
        return grad * mask.to(torch.float32)


class ApproxSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mask1_code, mask1_shape = BoolEncoder(torch.abs(x) <= 1)
        mask2_code, mask2_shape = BoolEncoder(torch.abs(x) <= 0.5)
        ctx.save_for_backward(mask1_code, mask2_code)
        ctx.mask1_shape = mask1_shape
        ctx.mask2_shape = mask2_shape
        result = torch.sign(x)
        return result

    @staticmethod
    def backward(ctx, grad):
        mask1_code, mask2_code = ctx.saved_tensors
        mask1 = BoolDecoder(mask1_code, ctx.mask1_shape)
        mask2 = BoolDecoder(mask2_code, ctx.mask2_shape)

        tmp = torch.zeros_like(grad)
        tmp[mask1] = 0.5
        tmp[mask2] = 1
        x_grad = grad * tmp
        return x_grad


class SignFunctionXNOR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        mask_code, mask_shape = BoolEncoder(torch.abs(x) <= 1)
        ctx.save_for_backward(mask_code)
        ctx.mask_shape = mask_shape
        scale = scale.to(x.device)
        result = scale*torch.sign(x)
        return result

    @staticmethod
    def backward(ctx, grad):
        mask_code, = ctx.saved_tensors
        mask = BoolDecoder(mask_code, ctx.mask_shape)
        x_grad = grad * mask.to(torch.float32)
        return x_grad, None


class SignFunctionReSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        result = sign_function(x)
        ctx.save_for_backward(x, scale)

        return result

    @staticmethod
    def backward(ctx, grad):
        sign_code, n_log2, mean, std, scale = ctx.saved_tensors
        sign = BoolDecoder(sign_code, ctx.sign_shape).to(torch.float32)*2-1
        temp = 2**(n_log2.to(torch.float32)/10+5) * sign
        dims = tuple([i for i in range(temp.dim()) if i != 1])
        temp_mean = temp.mean(dim=dims, keepdim=True)
        temp_var = temp.var(dim=dims, unbiased=False, keepdim=True)
        temp_std = torch.sqrt(temp_var + 1e-12)
        normalized = (temp - temp_mean) / temp_std
        x = normalized * std + mean

        scale = scale.to(grad.device)
        th = 1.5
        interval = 0.1

        mask1 = (-th <= x) & (x < -interval)
        mask2 = (-interval <= x) & (x <= interval)
        mask3 = (interval < x) & (x <= th)

        tmp = torch.zeros_like(grad)
        tmp[mask1] = 1/scale * (-x[mask1])**(1/scale-1)
        tmp[mask2] = 1/scale * interval**(1/scale-1)
        tmp[mask3] = 1/scale * (x[mask3])**(1/scale-1)

        x_grad = grad * tmp
        return x_grad, None