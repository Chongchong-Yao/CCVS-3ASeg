# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

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
    def __init__(self, ch_in, ch_out, kernel_size=3, reduction=16):
        super(EEMLite_Generator, self).__init__()

        # 保存路径设置
        self.heatmap_save_dir = None
        self.save_heatmaps = False

        # 根据权重文件结构调整模型结构
        self.gaussian_conv = nn.Conv2d(ch_in, ch_in, kernel_size,
                                       padding=kernel_size // 2,
                                       groups=ch_in, bias=False)
        self.laplace_conv = nn.Conv2d(ch_in, ch_in, kernel_size,
                                      padding=kernel_size // 2,
                                      groups=ch_in, bias=False)

        # 初始化并冻结
        self._init_and_freeze_kernels(ch_in, kernel_size)

        # 动态计算中间通道数
        mid_channels = max(1, ch_out // 2)

        self.post_process = nn.ModuleDict({
            'dog_branch': nn.Sequential(
                nn.Conv2d(ch_in, mid_channels, kernel_size=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(mid_channels)
            ),
            'log_branch': nn.Sequential(
                nn.Conv2d(ch_in, mid_channels, kernel_size=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(mid_channels)
            ),
            'fusion': nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(mid_channels, ch_out, kernel_size=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.GroupNorm(1, ch_out)
            )
        })

        self.sem = SEM(ch_out, reduction)
        self.final_activation = nn.Sigmoid()

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

    def set_heatmap_save_path(self, save_dir):
        """设置热力图保存路径"""
        self.heatmap_save_dir = save_dir
        self.save_heatmaps = True
        # 创建各阶段文件夹
        stages = ['input', 'gaussian', 'laplace', 'dog_processed', 'log_processed',
                 'interaction', 'fusion', 'sem_enhanced', 'final_output']
        for stage in stages:
            os.makedirs(os.path.join(save_dir, stage), exist_ok=True)

    def save_heatmap(self, tensor, stage_name, filename):
        """保存特征图热力图"""
        if not self.save_heatmaps or self.heatmap_save_dir is None:
            return

        # 转换为numpy并处理
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取batch中的第一个

        # 对多通道特征图，取平均或选择前几个通道
        if tensor.dim() == 3:
            if tensor.size(0) > 3:  # 多通道，取前3个通道或平均
                if tensor.size(0) >= 3:
                    tensor_vis = tensor[:3]  # 取前3个通道
                else:
                    tensor_vis = tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)
            else:
                tensor_vis = tensor
        else:
            tensor_vis = tensor.unsqueeze(0)

        # 归一化到[0,1]
        tensor_vis = tensor_vis.detach().cpu()
        min_val = tensor_vis.min()
        max_val = tensor_vis.max()
        if max_val > min_val:
            tensor_vis = (tensor_vis - min_val) / (max_val - min_val)

        # 转换为numpy并调整维度
        if tensor_vis.size(0) == 1:  # 单通道，使用热力图配色
            heatmap = tensor_vis.squeeze().numpy()
            heatmap = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            plt.figure(figsize=(8, 6))
            plt.imshow(heatmap_colored)
            plt.title(f'{stage_name} - Heatmap')
            plt.colorbar()
            plt.axis('off')
        else:  # 多通道，直接显示
            tensor_np = tensor_vis.numpy().transpose(1, 2, 0)
            plt.figure(figsize=(8, 6))
            plt.imshow(tensor_np)
            plt.title(f'{stage_name} - Feature Map')
            plt.axis('off')

        # 保存图片
        save_path = os.path.join(self.heatmap_save_dir, stage_name, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0.1)
        plt.close()

    def forward(self, x, return_attn=False, save_prefix=""):
        # 保存输入
        if self.save_heatmaps:
            self.save_heatmap(x, 'input', f'{save_prefix}_input.png')

        # 高斯卷积
        DoG = self.gaussian_conv(x)
        if self.save_heatmaps:
            self.save_heatmap(DoG, 'gaussian', f'{save_prefix}_gaussian.png')

        # 拉普拉斯卷积
        LoG = self.laplace_conv(DoG)
        if self.save_heatmaps:
            self.save_heatmap(LoG, 'laplace', f'{save_prefix}_laplace.png')

        # DoG分支处理
        DoG_processed = self.post_process['dog_branch'](DoG - x)
        if self.save_heatmaps:
            self.save_heatmap(DoG_processed, 'dog_processed', f'{save_prefix}_dog_processed.png')

        # LoG分支处理
        LoG_processed = self.post_process['log_branch'](LoG)
        if self.save_heatmaps:
            self.save_heatmap(LoG_processed, 'log_processed', f'{save_prefix}_log_processed.png')

        # 交互特征
        interaction = DoG_processed * LoG_processed
        if self.save_heatmaps:
            self.save_heatmap(interaction, 'interaction', f'{save_prefix}_interaction.png')

        # 融合
        tot = self.post_process['fusion'](interaction)
        if self.save_heatmaps:
            self.save_heatmap(tot, 'fusion', f'{save_prefix}_fusion.png')

        # SEM增强
        enhanced = self.sem(tot)
        if self.save_heatmaps:
            self.save_heatmap(enhanced, 'sem_enhanced', f'{save_prefix}_sem_enhanced.png')

        # 最终输出
        out = self.final_activation(x + enhanced)
        if self.save_heatmaps:
            self.save_heatmap(out, 'final_output', f'{save_prefix}_final_output.png')

        return out


def load_image(image_path, size=(448, 448)):
    """加载并预处理图像"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor, original_size


def test_model():
    """测试函数"""
    # 参数设置
    image_path = "C:/Users/ycc/Desktop/fsdownload/crack_2/images/CFD_011.jpg"  # 替换为你的测试图片路径
    model_weights = "./checkpoints/best_checkpoint.pth"
    heatmap_save_dir = "heatmap_results"


    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"测试图片不存在: {image_path}")
        return
    if not os.path.exists(model_weights):
        print(f"模型权重不存在: {model_weights}")
        return

    # 创建设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载权重文件检查结构
    try:
        checkpoint = torch.load(model_weights, map_location=device, weights_only=False)
        generator_weights = checkpoint['generator_state_dict']

        # 分析权重结构来确定输入输出通道数
        print("分析权重结构...")

        # 通过权重形状推断通道数
        gaussian_weight_shape = generator_weights['gaussian_conv.weight'].shape
        ch_in = gaussian_weight_shape[0]  # 输入通道数
        print(f"高斯卷积权重形状: {gaussian_weight_shape}")

        # 推断输出通道数
        if 'post_process.fusion.1.weight' in generator_weights:
            fusion_weight_shape = generator_weights['post_process.fusion.1.weight'].shape
            ch_out = fusion_weight_shape[0]  # 输出通道数
        else:
            # 如果找不到融合层权重，使用默认值
            ch_out = 1  # 从错误信息看，权重文件中的输出通道是1

        print(f"推断的通道数: 输入={ch_in}, 输出={ch_out}")

        # 创建与权重匹配的模型
        model = EEMLite_Generator(ch_in=ch_in, ch_out=ch_out, kernel_size=3)

        # 加载权重，忽略不匹配的键
        model.load_state_dict(generator_weights, strict=False)
        print("模型权重加载成功（忽略不匹配的键）")

    except Exception as e:
        print(f"权重加载失败: {e}")
        return

    model.to(device)
    model.eval()

    # 设置热力图保存
    model.set_heatmap_save_path(heatmap_save_dir)

    # 加载测试图像
    input_tensor, original_size = load_image(image_path)

    # 确保输入图像通道数与模型匹配
    # 注意：权重文件中的高斯卷积期望3个输入通道，但我们的模型创建时使用了推断的通道数1
    # 我们需要强制使用3通道输入
    if input_tensor.shape[1] != 3:
        print(f"强制使用3通道输入: {input_tensor.shape[1]} -> 3")
        # 如果输入不是3通道，复制单通道为3通道
        if input_tensor.shape[1] == 1:
            input_tensor = input_tensor.repeat(1, 3, 1, 1)

    input_tensor = input_tensor.to(device)

    print(f"输入图像尺寸: {input_tensor.shape}")
    print(f"模型输入通道: {model.gaussian_conv.weight.shape[1]}")
    print(f"模型输出通道: {model.post_process['fusion'][1].weight.shape[0]}")

    # 推理
    with torch.no_grad():
        output = model(input_tensor, save_prefix="test_sample")

    # 保存最终结果
    output_image = output[0].cpu()

    # 处理输出图像
    if output_image.shape[0] == 1:  # 单通道输出
        output_image = output_image.squeeze().numpy()
        output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
        output_pil = Image.fromarray(output_image, mode='L')
    else:  # 多通道输出
        output_image = output_image.permute(1, 2, 0).numpy()
        output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
        output_pil = Image.fromarray(output_image)

    output_pil.save(os.path.join(heatmap_save_dir, "final_result.jpg"))

    print(f"测试完成！结果保存在: {heatmap_save_dir}")


if __name__ == "__main__":
    test_model()