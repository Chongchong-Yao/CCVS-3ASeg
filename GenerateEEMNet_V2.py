# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, laplace


def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize), dtype=np.float32)
    center = kernelsize // 2
    n[:, :, center, center] = kernelvalue
    g = gaussian_filter(n, sigma)
    return torch.from_numpy(g).float()


def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize), dtype=np.float32)
    center = kernelsize // 2
    n[:, :, center, center] = kernelvalue
    l = laplace(n)
    return torch.from_numpy(l).float()


class SEM(nn.Module):
    def __init__(self, ch_out, reduction=16):
        super(SEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, max(1, ch_out // reduction), kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(max(1, ch_out // reduction), ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y.expand_as(x)


class EEM_Generator(nn.Module):
    def __init__(self, ch_in, ch_out, kernel=5, groups=1, reduction=16):
        super(EEM_Generator, self).__init__()

        # ===== 自动修正 groups =====
        if ch_in % groups != 0:
            groups = 1  # fallback
        self.groups = groups

        # ===== 固定核 =====
        gk = gaussiankernel(ch_in, ch_in // self.groups, kernel, sigma=kernel - 2, kernelvalue=0.9)
        lk = laplaceiankernel(ch_in, ch_in // self.groups, kernel, kernelvalue=0.9)
        self.register_buffer("gk", gk)
        self.register_buffer("lk", lk)

        # ===== 基础卷积（自动修正 groups）=====
        g1 = 1 if ch_in < 4 else 2  # 防止 in=3, group=2 报错
        g2 = 1 if ch_in < 4 else 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, max(1, ch_out // 2), kernel_size=1, padding=0, groups=g1),
            nn.PReLU(num_parameters=max(1, ch_out // 2), init=0.05),
            nn.InstanceNorm2d(max(1, ch_out // 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, max(1, ch_out // 2), kernel_size=1, padding=0, groups=g2),
            nn.PReLU(num_parameters=max(1, ch_out // 2), init=0.05),
            nn.InstanceNorm2d(max(1, ch_out // 2))
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(max(1, ch_out // 2), ch_out, kernel_size=1, padding=0, groups=1),
            nn.PReLU(num_parameters=ch_out, init=0.01),
            nn.GroupNorm(1, ch_out)
        )

        self.x_conv = nn.Conv2d(ch_in, ch_out, 1)  # 1x1卷积转换通道

        self.sem1 = SEM(ch_out, reduction=reduction)
        self.sem2 = SEM(ch_out, reduction=reduction)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.03)

        self.attn_conv = nn.Conv2d(ch_out, 1, kernel_size=1)
        self.attn_sigmoid = nn.Sigmoid()

    def forward(self, x, return_attn=False):
        device = x.device
        gk = self.gk.to(device)
        lk = self.lk.to(device)

        DoG = F.conv2d(x, gk, padding='same', groups=self.groups)
        LoG = F.conv2d(DoG, lk, padding='same', groups=self.groups)

        DoG = self.conv1(DoG - x)
        LoG = self.conv2(LoG)
        tot = self.conv3(DoG * LoG)

        tot1 = self.sem1(tot)

        x_converted = self.x_conv(x)
        x1 = self.sem2(x_converted)
        out = self.prelu(x_converted + x1 + tot + tot1)

        attn_map = self.attn_sigmoid(self.attn_conv(out))
        if attn_map.shape[-2:] != x.shape[-2:]:
            attn_map = F.interpolate(attn_map, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return (out, attn_map) if return_attn else out

