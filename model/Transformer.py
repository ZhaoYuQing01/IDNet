
from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from einops.layers.torch import Rearrange
import math
Conv2d = nn.Conv2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DWConv3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, L, N, C = x.shape
        tx = x.transpose(2, 3).transpose(1, 2).view(B, C, L, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(3).transpose(1, 2).transpose(2, 3)


class MixFFN_skip3(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv3(c2) 
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class LaplacianPyramid(nn.Module):
    def __init__(self, in_channels=64, pyramid_levels=3):

        super().__init__()
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        sigma = 1.6
        s_value = 2 ** (1 / 3)

        self.sigma_kernels = [
            self.get_gaussian_kernel(3* i + 4, sigma * s_value ** i)
            for i in range(pyramid_levels)
        ]

    def get_gaussian_kernel(self, kernel_size, sigma):
        kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
        kernel_weights = kernel_weights * kernel_weights.T
        kernel_weights = np.repeat(kernel_weights[None, ...], self.in_channels, axis=0)[:, None, ...]

        return torch.from_numpy(kernel_weights).float().to(device)

    def _calculate_same_padding(self, input_height, input_width, kernel_height, kernel_width, stride=1):
        h_pad_total = max((kernel_height - stride) % 2, 0)
        w_pad_total = max((kernel_width - stride) % 2, 0)
        h_pad_total = max(kernel_height - input_height, 0)
        w_pad_total = max(kernel_width - input_width, 0)
        if h_pad_total % 2 == 0:
            pad_along_height = h_pad_total // 2
            pad_along_width = w_pad_total // 2

        else:
            pad_along_height = h_pad_total // 2 + 1
            pad_along_width = w_pad_total // 2 + 1
        return (pad_along_height, pad_along_width)

    def forward(self, x):
        G = x
        attention_maps = [G]
        pyramid = [G]
        
        for kernel in self.sigma_kernels:
            padding = self._calculate_same_padding(x.size(2), x.size(3), kernel.size(2), kernel.size(3))
            G = F.conv2d(input=x, weight=kernel, bias=None, padding=padding, groups=self.in_channels)
            pyramid.append(G)

        for i in range(1, self.pyramid_levels + 1):
            scale_factor = pyramid[i - 1].shape[2] / pyramid[i].shape[2]
            pyramid[i] = F.interpolate(pyramid[i], scale_factor=scale_factor, mode='bilinear', align_corners=True)
            L = torch.sub(pyramid[i - 1], pyramid[i])
            attention_maps.append(L)

        return sum(attention_maps)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, in_channels, pyramid_levels=3, ifBox=True):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.reprojection = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_dw = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1), bias=False,
                                 groups=in_channels)
        self.freq_attention = LaplacianPyramid(in_channels=in_channels, pyramid_levels=pyramid_levels)
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        x=self.freq_attention(x)
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, :c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace
        return out    

class Attention3(nn.Module):
    def __init__(self, dim, num_heads, bias, in_channels, hsi_inchannel, pyramid_levels=3, ifBox=True):
        super(Attention3, self).__init__()
        self.in_channels = in_channels
        self.reprojection = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_dw = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1), bias=False,
                                 groups=in_channels)
        self.freq_attention = LaplacianPyramid(in_channels=in_channels*hsi_inchannel, pyramid_levels=pyramid_levels)
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)


    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)
    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hwc = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hwc)" if ifBox else "b (head c) (hwc factor)"
        shape_tar = "b head (c factor) hwc"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hwc=hwc, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hwc=hwc, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hwc=hwc, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hwc=hwc, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out
 
    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2],x.shape[3], x.shape[4])
        x=self.freq_attention(x)
        x = x.reshape(x.shape[0], c, d,x.shape[2], x.shape[3])
        x_sort, idx_d = x[:, :c // 2].sort(-3)
        x_sort, idx_h = x_sort.sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, :c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, d, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, d, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out_replace = torch.scatter(out_replace, -3, idx_d, out_replace)
        out[:, :c // 2] = out_replace

        return out
class SSICTransformerBlock(nn.Module):

    def __init__(self, in_dim, hsi_inchannel,pyramid_levels=3, token_mlp='mix_skip'):
        super().__init__()

        self.in_dim = in_dim
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = Attention3(dim=in_dim, num_heads=1, bias=False, in_channels=in_dim,
                                         hsi_inchannel=hsi_inchannel, pyramid_levels=pyramid_levels, ifBox=True)

        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip3(in_dim, int(in_dim * 4))
        elif token_mlp == 'ffn':
            self.mlp = FeedForwardNetwork(in_dim, int(in_dim * 4), dropout=0.1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor: 
        norm_1 = self.norm1(x)  # 归一化
        norm_1 = Rearrange('b c (h w) d -> b d c h w', h=H, w=W)(norm_1) 

        attn = self.attn(norm_1)
        attn = Rearrange('b d c h w -> b c (h w) d')(attn) 
       
        tx = x + attn
        mx = tx + self.mlp(self.norm2(tx), H, W)

        return mx
    
class FeedForwardNetwork(nn.Module):  
    def __init__(self, d_model, d_ff, dropout=0.1):  
        super(FeedForwardNetwork, self).__init__()  
        # 第一个线性层  
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gl=nn.GELU()  
        # Dropout层  
        self.dropout = nn.Dropout(dropout)  
        # 第二个线性层  
        self.linear2 = nn.Linear(d_ff, d_model)  
  
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:  
        x = self.linear1(x) 
        x = self.gl(x)
        x = self.dropout(x)
        x = self.linear2(x)  
        return x 
 

class ICTransformerBlock(nn.Module):

    def __init__(self, in_dim, pyramid_levels=3, token_mlp='mix_skip'):
        super().__init__() 
        self.in_dim = in_dim
        self.norm1 = nn.LayerNorm(in_dim)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.attn = Attention(dim=in_dim, num_heads=1, bias=False, in_channels=in_dim,
                                        pyramid_levels=pyramid_levels, ifBox=True)

        self.norm2 = nn.LayerNorm(in_dim)
        self.bn2 = nn.BatchNorm2d(in_dim)
        if token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim * 4))
        elif token_mlp == 'ffn':
            self.mlp = FeedForwardNetwork(in_dim, int(in_dim * 4), dropout=0.07)

    def forward(self, x):
        _, _, h,w = x.shape
        x = Rearrange('b d h w -> b (h w) d')(x)  
        norm_1 = self.norm1(x)  # 归一化
        norm_1 = Rearrange('b (h w) d -> b d h w', h=h, w=w)(norm_1)  
        attn = self.attn(norm_1)
        attn = Rearrange('b d h w -> b (h w) d')(attn) 
        tx = x + attn
        mx = tx + self.mlp(self.norm2(tx), h, w)
        mx = Rearrange('b (h w) d -> b d h w', h=h, w=w)(mx) 
        return mx
