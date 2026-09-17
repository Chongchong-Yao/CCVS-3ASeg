# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
import pandas as pd
import torch.nn.functional as F

# ===================== 模型1代码（EEMLite_Generator）=====================
class SEM_Lite(nn.Module):
    def __init__(self, ch_out, reduction=16):
        super(SEM_Lite, self).__init__()
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
    def __init__(self, ch_in, ch_out, kernel_size=3, reduction=16):
        super(EEMLite_Generator, self).__init__()

        self.gaussian_conv = nn.Conv2d(ch_in, ch_in, kernel_size,
                                       padding=kernel_size // 2,
                                       groups=ch_in, bias=False)

        self.laplace_conv = nn.Conv2d(ch_in, ch_in, kernel_size,
                                      padding=kernel_size // 2,
                                      groups=ch_in, bias=False)


        self._init_and_freeze_kernels(ch_in, kernel_size)

        g1 = 1 if ch_in < 4 else 2
        g2 = 1 if ch_in < 4 else 2

        self.post_process = nn.ModuleDict({
            'dog_branch': nn.Sequential(
                nn.Conv2d(ch_in, max(1, ch_out // 2), kernel_size=1, padding=0, groups=g1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(max(1, ch_out // 2))
            ),
            'log_branch': nn.Sequential(
                nn.Conv2d(ch_in, max(1, ch_out // 2), kernel_size=1, padding=0, groups=g2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(max(1, ch_out // 2))
            ),
            'fusion': nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(max(1, ch_out // 2), ch_out, kernel_size=1, padding=0, groups=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.GroupNorm(1, ch_out)
            )
        })

        self.sem = SEM_Lite(ch_out, reduction)

        self.final_activation = nn.Sigmoid()
        self.attn_conv = nn.Conv2d(ch_out, 1, kernel_size=1)
        self.attn_sigmoid = nn.Sigmoid()

    def _init_and_freeze_kernels(self, ch_in, kernel_size):
        gaussian_kernel = self._create_gaussian_kernel(kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(ch_in, 1, 1, 1)
        self.gaussian_conv.weight.data = gaussian_kernel
        self.gaussian_conv.weight.requires_grad = False

        laplace_kernel = self._create_laplace_kernel(kernel_size)
        laplace_kernel = laplace_kernel.repeat(ch_in, 1, 1, 1)
        self.laplace_conv.weight.data = laplace_kernel
        self.laplace_conv.weight.requires_grad = False

    def _create_gaussian_kernel(self, kernel_size, sigma=1.0):
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
        DoG = self.gaussian_conv(x)
        LoG = self.laplace_conv(DoG)
        DoG_processed = self.post_process['dog_branch'](DoG - x)
        LoG_processed = self.post_process['log_branch'](LoG)
        interaction = DoG_processed * LoG_processed
        tot = self.post_process['fusion'](interaction)
        enhanced = self.sem(tot)
        out = self.final_activation(x + enhanced)
        if return_attn:
            attn = torch.sigmoid(out.mean(dim=1, keepdim=True))
            return out, attn
        return out

# ===================== 模型2代码（EEM_Generator）=====================
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
        if ch_in % groups != 0:
            groups = 1
        self.groups = groups

        gk = gaussiankernel(ch_in, ch_in // self.groups, kernel, sigma=kernel - 2, kernelvalue=0.9)
        lk = laplaceiankernel(ch_in, ch_in // self.groups, kernel, kernelvalue=0.9)
        self.register_buffer("gk", gk)
        self.register_buffer("lk", lk)

        g1 = 1 if ch_in < 4 else 2
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

        self.x_conv = nn.Conv2d(ch_in, ch_out, 1)
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

# ===================== 通用统计函数 =====================
def count_model_params(model, model_name):
    """统计模型的可训练/不可训练参数、Buffer"""
    stats = {
        "model_name": model_name,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "non_trainable_params": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "buffer_elements": sum(b.numel() for b in model.buffers()),
        "total_params": sum(p.numel() for p in model.parameters()),  # 仅参数（不含Buffer）
        "total_storage": sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())  # 参数+Buffer
    }
    return stats

# ===================== 模块级明细统计 =====================
def count_module_params(model, model_name):
    """统计模块级可训练参数明细"""
    module_stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            module_stats.append({
                "model_name": model_name,
                "module": name,
                "param_count": param.numel()
            })
    return pd.DataFrame(module_stats)

# ===================== 运行统计与对比 =====================
if __name__ == "__main__":
    # 统一参数配置
    ch_in = 3
    ch_out = 1
    kernel = 5
    reduction = 16
    groups = 1

    # 初始化模型
    model1 = EEMLite_Generator(ch_in=ch_in, ch_out=ch_out, kernel_size=kernel, reduction=reduction)
    model2 = EEM_Generator(ch_in=ch_in, ch_out=ch_out, kernel=kernel, groups=groups, reduction=reduction)

    # 统计整体参数
    stats1 = count_model_params(model1, "EEMLite_Generator")
    stats2 = count_model_params(model2, "EEM_Generator")

    # 转换为DataFrame对比
    df_overall = pd.DataFrame([stats1, stats2])
    print("="*100)
    print("模型整体参数对比（ch_in=3, ch_out=1, kernel=5）")
    print("="*100)
    print(df_overall.to_string(index=False))

    # 统计模块级明细
    df_module1 = count_module_params(model1, "EEMLite_Generator")
    df_module2 = count_module_params(model2, "EEM_Generator")
    df_module = pd.concat([df_module1, df_module2], ignore_index=True)

    print("\n" + "="*100)
    print("模型模块级可训练参数明细")
    print("="*100)
    print(df_module.to_string(index=False))

    # 计算差异
    print("\n" + "="*100)
    print("参数差异对比")
    print("="*100)
    print(f"可训练参数差异：模型1 - 模型2 = {stats1['trainable_params'] - stats2['trainable_params']}（模型1多{stats1['trainable_params'] - stats2['trainable_params']}个）")
    print(f"不可训练参数差异：模型1 - 模型2 = {stats1['non_trainable_params'] - stats2['non_trainable_params']}（模型1多{stats1['non_trainable_params'] - stats2['non_trainable_params']}个）")
    print(f"Buffer元素差异：模型2 - 模型1 = {stats2['buffer_elements'] - stats1['buffer_elements']}（模型2多{stats2['buffer_elements'] - stats1['buffer_elements']}个）")
    print(f"总存储量差异：模型2 - 模型1 = {stats2['total_storage'] - stats1['total_storage']}（模型2多{stats2['total_storage'] - stats1['total_storage']}个元素）")