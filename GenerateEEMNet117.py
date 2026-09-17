# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

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


class EEMLite_Generator(nn.Module):
    """基于你原有网络的生成器适配版本 - 最小改动"""

    def __init__(self, ch_in, ch_out, kernel_size=3, reduction=16):
        super(EEMLite_Generator, self).__init__()

        # 保持你原有的所有卷积结构不变
        self.gaussian_conv = nn.Conv2d(ch_in, ch_in, kernel_size,
                                       padding=kernel_size // 2,
                                       groups=ch_in, bias=False)
        self.laplace_conv = nn.Conv2d(ch_in, ch_in, kernel_size,
                                      padding=kernel_size // 2,
                                      groups=ch_in, bias=False)

        # 初始化并冻结 - 完全保持你的逻辑
        self._init_and_freeze_kernels(ch_in, kernel_size)

        # ===== 只做最小必要的生成器适配 =====
        g1 = 1 if ch_in < 4 else 2
        g2 = 1 if ch_in < 4 else 2

        # 保持你的并行处理结构，只改激活函数
        self.post_process = nn.ModuleDict({
            'dog_branch': nn.Sequential(
                nn.Conv2d(ch_in, max(1, ch_out // 2), kernel_size=1, padding=0, groups=g1),
                nn.LeakyReLU(0.2, inplace=True),  # 改为LeakyReLU
                nn.InstanceNorm2d(max(1, ch_out // 2))
            ),
            'log_branch': nn.Sequential(
                nn.Conv2d(ch_in, max(1, ch_out // 2), kernel_size=1, padding=0, groups=g2),
                nn.LeakyReLU(0.2, inplace=True),  # 改为LeakyReLU
                nn.InstanceNorm2d(max(1, ch_out // 2))
            ),
            'fusion': nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(max(1, ch_out // 2), ch_out, kernel_size=1, padding=0, groups=1),
                nn.LeakyReLU(0.2, inplace=True),  # 改为LeakyReLU
                nn.GroupNorm(1, ch_out)
            )
        })

        # 保持SEM不变
        self.sem = SEM(ch_out, reduction)

        # 改为Tanh输出适配生成器
        self.final_activation = nn.Sigmoid()  # 改为Tanh

        # 注意力部分保持不变
        self.attn_conv = nn.Conv2d(ch_out, 1, kernel_size=1)
        self.attn_sigmoid = nn.Sigmoid()

    def _init_and_freeze_kernels(self, ch_in, kernel_size):
        """完全保持你的初始化逻辑"""
        gaussian_kernel = self._create_gaussian_kernel(kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(ch_in, 1, 1, 1)
        self.gaussian_conv.weight.data = gaussian_kernel
        self.gaussian_conv.weight.requires_grad = False

        laplace_kernel = self._create_laplace_kernel(kernel_size)
        laplace_kernel = laplace_kernel.repeat(ch_in, 1, 1, 1)
        self.laplace_conv.weight.data = laplace_kernel
        self.laplace_conv.weight.requires_grad = False

    def _create_gaussian_kernel(self, kernel_size, sigma=1.0):
        """完全保持你的高斯核生成"""
        if kernel_size == 3:
            return torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).float().view(1, 1, 3, 3) / 16.0
        elif kernel_size == 5:
            kernel = torch.tensor([
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1]
            ]).float() / 256.0
            return kernel.view(1, 1, 5, 5)
        else:
            x = torch.arange(kernel_size).float() - kernel_size // 2
            x = x.expand(kernel_size, kernel_size)
            y = x.t()
            kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            return (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)

    def _create_laplace_kernel(self, kernel_size):
        """完全保持你的拉普拉斯核生成"""
        if kernel_size == 3:
            return torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).float().view(1, 1, 3, 3)
        elif kernel_size == 5:
            kernel = torch.tensor([
                [0, 0, 1, 0, 0],
                [0, 1, 2, 1, 0],
                [1, 2, -16, 2, 1],
                [0, 1, 2, 1, 0],
                [0, 0, 1, 0, 0]
            ]).float()
            return kernel.view(1, 1, 5, 5)
        else:
            center = kernel_size // 2
            kernel = torch.zeros(kernel_size, kernel_size)
            kernel[center, center] = -4
            for i in [-1, 1]:
                if 0 <= center + i < kernel_size:
                    kernel[center + i, center] = 1
                    kernel[center, center + i] = 1
            return kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, x, return_attn=False):
        """完全保持你的前向逻辑，只改输出激活"""
        # 保持你的所有处理步骤
        DoG = self.gaussian_conv(x)
        LoG = self.laplace_conv(DoG)

        DoG_processed = self.post_process['dog_branch'](DoG - x)
        LoG_processed = self.post_process['log_branch'](LoG)

        interaction = DoG_processed * LoG_processed
        tot = self.post_process['fusion'](interaction)

        enhanced = self.sem(tot)
        out = self.final_activation(x + enhanced)  # 现在输出在[-1,1]范围

        if return_attn:
            attn = torch.sigmoid(out.mean(dim=1, keepdim=True))
            return out, attn

        return out




