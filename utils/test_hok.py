import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import cv2
from PIL import Image
import time


class FeatureMapVisualizer:
    """Feature Map Visualization Tool - Inverted Heatmap Version"""

    def __init__(self, save_dir="feature_maps"):
        self.save_dir = save_dir
        self.hook_handles = []
        self.feature_maps = {}
        os.makedirs(save_dir, exist_ok=True)

    def register_hooks(self, model):
        """Register forward hooks for the model"""
        # Clear previous hooks
        self.remove_hooks()

        # Register hooks for key layers
        layers_to_hook = [
            'gaussian_conv', 'laplace_conv',
            'post_process.dog_branch.0', 'post_process.log_branch.0',
            'post_process.fusion.1', 'sem.conv.0', 'sem.conv.1', 'sem.conv.2', 'sem.conv.3',
            'final_activation'
        ]

        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in layers_to_hook):
                handle = module.register_forward_hook(self._hook_fn(name))
                self.hook_handles.append(handle)
                print(f"Registered feature map hook: {name}")

    def _hook_fn(self, layer_name):
        """Hook function - Save input and output feature maps"""

        def hook(module, input, output):
            # Save input and output feature maps
            if input[0] is not None:
                self.feature_maps[f"{layer_name}_input"] = {
                    'data': input[0].detach().cpu(),
                    'shape': input[0].shape
                }
            if output is not None:
                self.feature_maps[f"{layer_name}_output"] = {
                    'data': output.detach().cpu(),
                    'shape': output.shape
                }

        return hook

    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        print("Removed all feature map hooks")

    def visualize_feature_maps(self, epoch=0, batch_idx=0, original_img=None, img_name="unknown"):
        """Visualize feature maps - Inverted heatmap version"""
        if not self.feature_maps:
            print("No feature map data to visualize")
            return

        # Create organized directory structure
        img_base_name = os.path.splitext(os.path.basename(img_name))[0]
        batch_dir = os.path.join(self.save_dir, f"batch_{batch_idx:04d}_{img_base_name}")
        os.makedirs(batch_dir, exist_ok=True)

        print(f"Visualizing feature maps: {img_name} (batch {batch_idx})")

        # Visualize each feature map
        for name, feature_info in self.feature_maps.items():
            self._save_elegant_feature_map(feature_info['data'], name, batch_dir, original_img)

        # Create feature map overview report
        self._create_feature_report(batch_dir)

        print(f"Feature maps saved to: {batch_dir}")
        return batch_dir

    def _save_elegant_feature_map(self, feature_map, layer_name, save_dir, original_img=None):
        """Save elegant feature map heatmap - Inverted color mapping"""
        try:
            # 检查特征图是否为空或无效
            if feature_map is None or feature_map.numel() == 0:
                print(f"Skipping empty feature map: {layer_name}")
                return

            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(f'Feature Map Analysis - {layer_name}\nShape: {feature_map.shape}',
                         fontsize=16, fontweight='bold', y=0.95)

            # 左子图：特征图网格
            self._plot_feature_grid(feature_map, ax1, layer_name)

            # 右子图：反转热图叠加
            self._plot_heatmap_overlay(feature_map, ax2, layer_name, original_img)

            plt.tight_layout()

            # 保存高质量图像
            filename = os.path.join(save_dir, f"{layer_name.replace('.', '_')}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

        except Exception as e:
            print(f"Error saving feature map {layer_name}: {e}")
            import traceback
            traceback.print_exc()

    def _plot_feature_grid(self, feature_map, ax, layer_name):
        """绘制特征图网格"""
        try:
            # 处理特征图数据
            if feature_map.dim() == 4:
                fm = feature_map[0]  # 取第一个batch
                n_channels = min(16, fm.size(0))  # 最多显示16个通道
                fm = fm[:n_channels]

                if fm.size(0) > 0:
                    # 归一化每个通道
                    fm_normalized = []
                    for i in range(fm.size(0)):
                        channel = fm[i]
                        c_min = channel.min()
                        c_max = channel.max()
                        if c_max - c_min > 1e-8:
                            normalized = (channel - c_min) / (c_max - c_min)
                        else:
                            normalized = torch.zeros_like(channel)
                        fm_normalized.append(normalized)

                    fm_normalized = torch.stack(fm_normalized)
                    grid = make_grid(fm_normalized.unsqueeze(1), nrow=4, padding=2, normalize=False)
                    grid_np = grid.numpy().transpose(1, 2, 0)

                    im1 = ax.imshow(grid_np, cmap='viridis', aspect='auto')
                    ax.set_title(f'Feature Map Channels (First {n_channels})', fontsize=12, fontweight='bold')
                    ax.axis('off')
                    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, 'No Feature Map Data', horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title('No Feature Map Data', fontsize=12, fontweight='bold')
                    ax.axis('off')
            else:
                # 处理非4维特征图
                stats_text = self._get_feature_stats(feature_map)
                ax.text(0.1, 0.5, stats_text, fontsize=10, fontfamily='monospace',
                        verticalalignment='center', transform=ax.transAxes)
                ax.set_title('Feature Map Statistics', fontsize=12, fontweight='bold')
                ax.axis('off')

        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting grid:\n{str(e)}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10)
            ax.set_title('Error Plotting Feature Grid', fontsize=12, fontweight='bold')
            ax.axis('off')

    def _plot_heatmap_overlay(self, feature_map, ax, layer_name, original_img):
        """绘制热图叠加"""
        try:
            if original_img is None:
                # 如果没有原图，显示统计信息
                stats_text = self._get_feature_stats(feature_map)
                ax.text(0.1, 0.5, stats_text, fontsize=10, fontfamily='monospace',
                        verticalalignment='center', transform=ax.transAxes)
                ax.set_title('Feature Map Statistics', fontsize=12, fontweight='bold')
                ax.axis('off')
                return

            # 特殊处理SEM模块的特征图
            if 'sem.conv' in layer_name:
                self._plot_sem_heatmap(feature_map, ax, layer_name, original_img)
                return

            # 创建特征图热图叠加 - 反转版本
            if feature_map.dim() == 4:
                overall_feature = feature_map.mean(dim=1, keepdim=True)[0]  # 取第一个batch，平均所有通道
            else:
                # 对于非4维特征图，直接使用
                overall_feature = feature_map[0] if feature_map.dim() > 0 else feature_map

            # 确保我们有有效的特征图数据
            if overall_feature.numel() == 0:
                ax.text(0.5, 0.5, 'Empty Feature Map\nCannot create heatmap',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title('Empty Feature Map', fontsize=12, fontweight='bold')
                ax.axis('off')
                return

            # 转换为numpy数组
            overall_feature_np = overall_feature.squeeze().cpu().numpy()

            # 检查数组是否有效
            if not isinstance(overall_feature_np, np.ndarray) or overall_feature_np.size == 0:
                ax.text(0.5, 0.5, 'Invalid Feature Map Data\nCannot create heatmap',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title('Invalid Feature Map Data', fontsize=12, fontweight='bold')
                ax.axis('off')
                return

            # 确保至少是2D数组
            if overall_feature_np.ndim < 2:
                # 如果是1D或0D，尝试重塑为2D
                if overall_feature_np.ndim == 1:
                    # 如果是1D，尝试找到合适的形状
                    size = int(np.sqrt(overall_feature_np.size))
                    if size * size == overall_feature_np.size:
                        overall_feature_np = overall_feature_np.reshape(size, size)
                    else:
                        # 如果不能重塑为正方形，使用矩形
                        overall_feature_np = overall_feature_np.reshape(1, -1)
                else:
                    # 如果是0D，扩展为1x1
                    overall_feature_np = np.array([[overall_feature_np]])

            # 归一化
            f_min = overall_feature_np.min()
            f_max = overall_feature_np.max()
            if f_max - f_min > 1e-8:
                overall_feature_norm = (overall_feature_np - f_min) / (f_max - f_min)
            else:
                overall_feature_norm = np.zeros_like(overall_feature_np)

            # 反转热图
            overall_feature_norm = 1.0 - overall_feature_norm

            # 调整到原图尺寸
            if isinstance(original_img, Image.Image):
                orig_w, orig_h = original_img.size
            else:
                orig_h, orig_w = original_img.shape[:2]

            # 确保数组是连续的并且类型正确
            overall_feature_norm = np.ascontiguousarray(overall_feature_norm.astype(np.float32))

            try:
                # 调整大小
                heatmap_resized = cv2.resize(overall_feature_norm, (orig_w, orig_h),
                                             interpolation=cv2.INTER_LINEAR)

                # 创建热图
                heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # 转换为RGB并叠加
                if isinstance(original_img, Image.Image):
                    original_np = np.array(original_img.convert("RGB"))
                else:
                    original_np = original_img

                # 确保尺寸匹配
                if original_np.shape[:2] != heatmap_color.shape[:2]:
                    original_np = cv2.resize(original_np, (heatmap_color.shape[1], heatmap_color.shape[0]))

                # 叠加热图和原图
                overlay = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)

                ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                ax.set_title('Inverted Heatmap Overlay (Low=Hot, High=Cold)', fontsize=12, fontweight='bold')
                ax.axis('off')

            except Exception as cv_error:
                ax.text(0.5, 0.5, f'OpenCV Error:\n{str(cv_error)}',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=10)
                ax.set_title('OpenCV Processing Error', fontsize=12, fontweight='bold')
                ax.axis('off')

        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating heatmap:\n{str(e)}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10)
            ax.set_title('Error Creating Heatmap', fontsize=12, fontweight='bold')
            ax.axis('off')

    def _plot_sem_heatmap(self, feature_map, ax, layer_name, original_img):
        """专门处理SEM模块的特征图"""
        try:
            # SEM模块的特征图通常是1x1的通道注意力权重
            # 我们需要特殊处理这些特征图

            # 获取特征图数据
            if feature_map.dim() == 4:
                # 对于4D特征图，取第一个样本
                sem_data = feature_map[0]
            else:
                sem_data = feature_map

            # 转换为numpy
            sem_np = sem_data.cpu().numpy()

            # 获取原图尺寸
            if isinstance(original_img, Image.Image):
                orig_w, orig_h = original_img.size
                original_np = np.array(original_img.convert("RGB"))
            else:
                orig_h, orig_w = original_img.shape[:2]
                original_np = original_img

            # 创建SEM热图
            if sem_np.ndim == 1:
                # 如果是1D，表示通道注意力权重
                # 创建一个与特征图通道数对应的彩色条
                n_channels = sem_np.shape[0]
                channel_colors = plt.cm.viridis(sem_np / (sem_np.max() + 1e-8))

                # 创建通道可视化
                fig_channel, ax_channel = plt.subplots(figsize=(10, 2))
                for i, color in enumerate(channel_colors):
                    ax_channel.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
                ax_channel.set_xlim(0, n_channels)
                ax_channel.set_ylim(0, 1)
                ax_channel.set_title(f'SEM Channel Weights - {layer_name}')
                ax_channel.set_xlabel('Channel Index')
                ax_channel.set_ylabel('Weight')

                # 保存通道可视化
                channel_path = os.path.join(os.path.dirname(ax.get_figure().canvas.manager.get_window_title()),
                                            f"{layer_name.replace('.', '_')}_channels.png")
                plt.savefig(channel_path, bbox_inches='tight')
                plt.close(fig_channel)

                # 在原图上显示SEM信息
                ax.imshow(cv2.cvtColor(original_np, cv2.COLOR_BGR2RGB))
                ax.text(0.5, 0.9, f'SEM Module - {layer_name}',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
                ax.text(0.5, 0.7, f'Channels: {n_channels}',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=10, color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
                ax.text(0.5, 0.5, f'Min: {sem_np.min():.4f}, Max: {sem_np.max():.4f}',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=10, color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
                ax.set_title('SEM Channel Attention Weights', fontsize=12, fontweight='bold')
                ax.axis('off')

            else:
                # 对于其他形状的SEM特征图，尝试正常处理
                self._plot_heatmap_overlay(feature_map, ax, layer_name, original_img)

        except Exception as e:
            ax.text(0.5, 0.5, f'SEM Processing Error:\n{str(e)}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10)
            ax.set_title('SEM Processing Error', fontsize=12, fontweight='bold')
            ax.axis('off')

    def _get_feature_stats(self, feature_map):
        """获取特征图统计信息"""
        try:
            stats = []
            stats.append("=== Feature Map Statistics ===")
            stats.append(f"Shape: {tuple(feature_map.shape)}")
            stats.append(f"Min: {feature_map.min():.6f}")
            stats.append(f"Max: {feature_map.max():.6f}")
            stats.append(f"Mean: {feature_map.mean():.6f}")

            # 安全的std计算
            if feature_map.numel() > 1:
                std_val = feature_map.std().item()
            else:
                std_val = 0.0
            stats.append(f"Std: {std_val:.6f}")

            stats.append(f"Non-zero elements: {(feature_map != 0).sum().item()}")
            stats.append(f"Sparsity: {((feature_map == 0).sum().item() / feature_map.numel()):.2%}")

            return "\n".join(stats)
        except Exception as e:
            return f"Error calculating stats: {e}"

    def _create_feature_report(self, save_dir):
        """创建特征图分析报告"""
        try:
            # 收集统计信息
            stats_data = []
            for name, feature_info in self.feature_maps.items():
                fm = feature_info['data']
                try:
                    # 安全统计计算
                    min_val = fm.min().item()
                    max_val = fm.max().item()
                    mean_val = fm.mean().item()

                    if fm.numel() > 1:
                        std_val = fm.std().item()
                    else:
                        std_val = 0.0

                    stats_data.append({
                        'layer': name,
                        'shape': tuple(fm.shape),
                        'min': min_val,
                        'max': max_val,
                        'mean': mean_val,
                        'std': std_val,
                        'non_zero': (fm != 0).sum().item(),
                        'sparsity': (fm == 0).sum().item() / fm.numel()
                    })
                except Exception as e:
                    print(f"Error processing stats for {name}: {e}")
                    continue

            # HTML模板
            html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Feature Map Visualization Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .feature-item { border: 1px solid #ddd; padding: 10px; border-radius: 5px; background: #f9f9f9; }
        .feature-item img { width: 100%; height: auto; }
        .feature-name { font-weight: bold; margin-bottom: 10px; text-align: center; }
    </style>
</head>
<body>
    <h1>Feature Map Visualization Report</h1>
    <div class="container">
"""

            for stat in stats_data:
                img_filename = f"{stat['layer'].replace('.', '_')}.png"
                html_content += f"""
        <div class="feature-item">
            <div class="feature-name">{stat['layer']}</div>
            <img src="{img_filename}" alt="{stat['layer']}">
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Shape</td><td>{stat['shape']}</td></tr>
                <tr><td>Min</td><td>{stat['min']:.6f}</td></tr>
                <tr><td>Max</td><td>{stat['max']:.6f}</td></tr>
                <tr><td>Mean</td><td>{stat['mean']:.6f}</td></tr>
                <tr><td>Std</td><td>{stat['std']:.6f}</td></tr>
                <tr><td>Non-zero</td><td>{stat['non_zero']:,}</td></tr>
                <tr><td>Sparsity</td><td>{stat['sparsity']:.2%}</td></tr>
            </table>
        </div>
"""

            html_content += """
    </div>
</body>
</html>"""

            # 确保使用UTF-8编码保存
            report_path = os.path.join(save_dir, "feature_analysis_report.html")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"Feature map report saved: {report_path}")

        except Exception as e:
            print(f"Error creating feature map report: {e}")

    def clear_feature_maps(self):
        """清空特征图缓存"""
        self.feature_maps.clear()