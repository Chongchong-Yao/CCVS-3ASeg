import torch
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import sklearn.metrics as metrics
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import time
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 导入模型（确保模型文件路径正确）
from GenerateEEMNet import EEMLite_Generator


def load_model(model_path, device="cuda"):
    """加载训练好的模型并设置为评估模式"""
    model = EEMLite_Generator(ch_in=3, ch_out=1)
    checkpoint = torch.load(model_path, map_location=device)
    print("权重文件中的键名：", checkpoint.keys())
    model.load_state_dict(checkpoint['generator_state_dict'], strict=False)
    model.to(device)
    model.eval()
    print(f"成功加载模型权重: {model_path}")
    return model


def preprocess_image_and_mask(img_path, mask_path, size=(512, 640), ignore_value=None):
    """
    预处理图片和对应的标签掩膜（核心修改：保留原始A通道，用A通道掩膜RGB）
    Args:
        img_path: 图像路径（RGBA格式）
        mask_path: 掩膜路径（可为None）
        size: 预处理后图像尺寸
        ignore_value: 需要忽略的像素值（归一化前）
    Returns:
        img_rgb: 经A通道掩膜后的RGB PIL图像
        tensor: 预处理后的图像张量 [1,3,H,W]（RGB）
        mask_any: 忽略区域掩码 [1,1,H,W]
        mask_binary: 二值化掩膜 [1,1,H,W]（None表示无掩膜）
        original_alpha: 原始A通道（用于最终输出RGBA）
    """
    # 读取RGBA图像，分离RGB和A通道
    img_rgba = Image.open(img_path).convert("RGBA")
    img_rgb_pil = img_rgba.convert("RGB")  # 先转为RGB
    original_alpha = np.array(img_rgba.split()[-1])  # 提取原始A通道（0=透明，255=不透明）

    # 用A通道掩膜RGB：A=0的区域，RGB设为0（黑色）
    img_rgb_np = np.array(img_rgb_pil)
    alpha_mask = (original_alpha > 0)  # A>0的区域保留，A=0的区域掩膜
    img_rgb_masked_np = img_rgb_np * alpha_mask[..., np.newaxis]  # 广播掩膜到RGB三通道
    img_rgb_masked = Image.fromarray(img_rgb_masked_np)  # 掩膜后的RGB图像

    # 图像预处理流程
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    tensor = transform(img_rgb_masked).unsqueeze(0)  # [1,3,H,W]（已用A通道掩膜）

    # 处理忽略值
    if ignore_value is not None:
        ignore_norm = ignore_value / 255.0
        mask = (tensor == ignore_norm)
        tensor = tensor.clone()
        tensor[mask] = 0.0
        mask_any = mask.any(dim=1, keepdim=True)  # [1,1,H,W]
    else:
        mask_any = torch.zeros((1, 1, *tensor.shape[2:]), dtype=torch.bool)

    # 掩膜预处理（如果有外部掩膜文件，优先级低于A通道）
    if mask_path and os.path.exists(mask_path):
        mask_img = Image.open(mask_path).convert("L")  # 转为灰度图
        mask_tensor = transforms.Resize(size)(mask_img)
        mask_tensor = transforms.ToTensor()(mask_tensor).unsqueeze(0)  # [1,1,H,W]
        mask_binary = (mask_tensor > 0).float()  # 二值化：裂缝=1，背景=0
    else:
        mask_binary = None

    return img_rgb_masked, tensor, mask_any, mask_binary, original_alpha


def calculate_classification_metrics(prediction, target, threshold=0.25):
    """
    计算多类别分类指标（背景类=0，裂缝类=1）
    Args:
        prediction: 预测结果（张量或numpy数组）
        target: 真实标签（张量或numpy数组）
        threshold: 二值化阈值
    Returns:
        包含全局指标和类别指标的字典
    """
    # 转为numpy数组并展平
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred_flat = prediction.flatten()
    target_flat = target.flatten()

    # 二值化
    pred_binary = (pred_flat < threshold).astype(np.uint8)
    target_binary = (target_flat > 0).astype(np.uint8)

    # 混淆矩阵
    cm = confusion_matrix(target_binary, pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_pixels = len(pred_flat)

    # 全局指标
    accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0

    # 背景类（0）指标
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    support_0 = tn + fp
    iou_0 = tn / (tn + fn + fp) if (tn + fn + fp) > 0 else 0

    # 裂缝类（1）指标
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    support_1 = tp + fn
    iou_1 = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    # 宏观/加权平均指标
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    weight_0 = support_0 / total_pixels if total_pixels > 0 else 0
    weight_1 = support_1 / total_pixels if total_pixels > 0 else 0
    weighted_precision = precision_0 * weight_0 + precision_1 * weight_1
    weighted_recall = recall_0 * weight_0 + recall_1 * weight_1
    weighted_f1 = f1_0 * weight_0 + f1_1 * weight_1

    # 整体IoU和Dice
    intersection_total = tp + tn
    union_total = tp + fp + fn + tn
    overall_iou = intersection_total / union_total if union_total > 0 else 0
    mean_iou = (iou_0 + iou_1) / 2
    dice = 2 * intersection_total / (2 * intersection_total + fp + fn) if (2 * intersection_total + fp + fn) > 0 else 0

    return {
        'overall': {
            'accuracy': accuracy,
            'overall_iou': overall_iou,
            'mean_iou': mean_iou,
            'dice': dice,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'total_pixels': total_pixels
        },
        'class_metrics': {
            0: {'precision': precision_0, 'recall': recall_0, 'f1_score': f1_0, 'support': support_0, 'iou': iou_0},
            1: {'precision': precision_1, 'recall': recall_1, 'f1_score': f1_1, 'support': support_1, 'iou': iou_1}
        }
    }


def calculate_pr_curve(predictions, targets):
    """
    计算精确率-召回率曲线和AP值
    Args:
        predictions: 预测结果列表
        targets: 真实标签列表
    Returns:
        ap: 平均精度
        precision_vals: 精确率数组
        recall_vals: 召回率数组
    """
    all_preds = []
    all_targets = []

    for pred, target in zip(predictions, targets):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        all_preds.extend(pred.flatten())
        all_targets.extend(target.flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    ap = average_precision_score(all_targets, all_preds)
    precision_vals, recall_vals, _ = precision_recall_curve(all_targets, all_preds)

    return ap, precision_vals, recall_vals


def create_elegant_metrics_plots(metrics_history, speed_metrics, save_dir):
    """
    创建优雅的指标可视化图表
    Args:
        metrics_history: 指标历史记录字典
        speed_metrics: 速度指标字典
        save_dir: 图表保存目录
    Returns:
        图表文件路径列表
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 主要性能指标趋势图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('模型性能指标趋势', fontsize=16, fontweight='bold')

    # 准确率和F1分数
    if metrics_history['overall_accuracy'] and metrics_history['class_1_f1']:
        x_range = range(1, len(metrics_history['overall_accuracy']) + 1)
        axes[0, 0].plot(x_range, metrics_history['overall_accuracy'], 'o-', linewidth=2, markersize=4, label='准确率')
        axes[0, 0].plot(x_range, metrics_history['class_1_f1'], 's-', linewidth=2, markersize=4, label='裂缝F1')
        axes[0, 0].plot(x_range, metrics_history['class_0_f1'], '^-', linewidth=2, markersize=4, label='背景F1')
        axes[0, 0].plot(x_range, metrics_history['macro_f1'], 'v-', linewidth=2, markersize=4, label='宏观F1')
        axes[0, 0].set_xlabel('图像序号')
        axes[0, 0].set_ylabel('分数')
        axes[0, 0].set_title('准确率 & F1分数趋势')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 精确率和召回率
    if metrics_history['class_1_precision'] and metrics_history['class_1_recall']:
        x_range = range(1, len(metrics_history['class_1_precision']) + 1)
        axes[0, 1].plot(x_range, metrics_history['class_1_precision'], 'o-', linewidth=2, markersize=4,
                        label='裂缝精确率')
        axes[0, 1].plot(x_range, metrics_history['class_1_recall'], 's-', linewidth=2, markersize=4, label='裂缝召回率')
        axes[0, 1].plot(x_range, metrics_history['class_0_precision'], '^-', linewidth=2, markersize=4,
                        label='背景精确率')
        axes[0, 1].plot(x_range, metrics_history['class_0_recall'], 'd-', linewidth=2, markersize=4, label='背景召回率')
        axes[0, 1].set_xlabel('图像序号')
        axes[0, 1].set_ylabel('分数')
        axes[0, 1].set_title('精确率 & 召回率趋势')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # IoU和Dice系数
    if metrics_history['class_1_iou'] and metrics_history['overall_dice']:
        x_range = range(1, len(metrics_history['class_1_iou']) + 1)
        axes[1, 0].plot(x_range, metrics_history['class_1_iou'], 'o-', linewidth=2, markersize=4, label='裂缝IoU')
        axes[1, 0].plot(x_range, metrics_history['class_0_iou'], 's-', linewidth=2, markersize=4, label='背景IoU')
        axes[1, 0].plot(x_range, metrics_history['mean_iou'], '^-', linewidth=2, markersize=4, label='平均IoU')
        axes[1, 0].plot(x_range, metrics_history['overall_dice'], 'v-', linewidth=2, markersize=4, label='Dice系数')
        axes[1, 0].set_xlabel('图像序号')
        axes[1, 0].set_ylabel('分数')
        axes[1, 0].set_title('分割指标趋势')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 推理速度趋势
    if speed_metrics['inference_times']:
        x_range = range(1, len(speed_metrics['inference_times']) + 1)
        axes[1, 1].plot(x_range, speed_metrics['inference_times'], '^-', color='purple', linewidth=2, markersize=4,
                        label='推理时间')
        axes[1, 1].set_xlabel('图像序号')
        axes[1, 1].set_ylabel('时间 (秒)')
        axes[1, 1].set_title('推理时间趋势')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    performance_plot_path = os.path.join(save_dir, "elegant_performance_metrics.png")
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. 速度指标汇总图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('模型推理速度分析', fontsize=16, fontweight='bold')

    # 推理时间分布
    if speed_metrics['inference_times']:
        axes[0].hist(speed_metrics['inference_times'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(speed_metrics['inference_times']), color='red', linestyle='--', linewidth=2,
                        label=f'平均时间: {np.mean(speed_metrics["inference_times"]):.4f}s')
        axes[0].set_xlabel('推理时间 (秒)')
        axes[0].set_ylabel('频次')
        axes[0].set_title('推理时间分布')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # FPS分布
    if speed_metrics['fps_list']:
        axes[1].hist(speed_metrics['fps_list'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].axvline(np.mean(speed_metrics['fps_list']), color='red', linestyle='--', linewidth=2,
                        label=f'平均FPS: {np.mean(speed_metrics["fps_list"]):.2f}')
        axes[1].set_xlabel('FPS')
        axes[1].set_ylabel('频次')
        axes[1].set_title('FPS分布')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    speed_plot_path = os.path.join(save_dir, "elegant_speed_analysis.png")
    plt.savefig(speed_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 3. 类别性能对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('类别性能对比', fontsize=16, fontweight='bold')

    # 精确率和召回率对比
    categories = ['背景', '裂缝']
    precision_values = [np.mean(metrics_history['class_0_precision']), np.mean(metrics_history['class_1_precision'])]
    recall_values = [np.mean(metrics_history['class_0_recall']), np.mean(metrics_history['class_1_recall'])]

    x = np.arange(len(categories))
    width = 0.35

    axes[0].bar(x - width / 2, precision_values, width, label='精确率', alpha=0.7)
    axes[0].bar(x + width / 2, recall_values, width, label='召回率', alpha=0.7)
    axes[0].set_xlabel('类别')
    axes[0].set_ylabel('分数')
    axes[0].set_title('各类别精确率和召回率')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1分数和IoU对比
    f1_values = [np.mean(metrics_history['class_0_f1']), np.mean(metrics_history['class_1_f1'])]
    iou_values = [np.mean(metrics_history['class_0_iou']), np.mean(metrics_history['class_1_iou'])]

    axes[1].bar(x - width / 2, f1_values, width, label='F1分数', alpha=0.7)
    axes[1].bar(x + width / 2, iou_values, width, label='IoU', alpha=0.7)
    axes[1].set_xlabel('类别')
    axes[1].set_ylabel('分数')
    axes[1].set_title('各类别F1分数和IoU')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    class_comparison_path = os.path.join(save_dir, "class_performance_comparison.png")
    plt.savefig(class_comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"🎨 优雅指标图表已保存至: {save_dir}")
    return [performance_plot_path, speed_plot_path, class_comparison_path]


def save_pure_heatmap(data, original_img, save_path):
    """
    保存纯热图（不叠加原图），尺寸与原图一致
    Args:
        data: 热图数据（张量或numpy数组）
        original_img: 原始PIL图像
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 数据预处理
    if isinstance(data, torch.Tensor):
        data = data.squeeze().cpu().numpy()

    # 归一化到0-255
    norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    heatmap_uint8 = (norm * 255).astype(np.uint8)

    # 调整尺寸并生成彩色热图
    orig_w, orig_h = original_img.size
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(save_path, heatmap_color)
    print(f"已保存纯热图: {save_path}")


def save_image(attn_tensor, mask_tensor, save_path, original_img=None, original_alpha=None,
               keep_white=True, heat_threshold=0.18, blend_alpha=0.8,
               enhance_strength=3.5, soften=False, soften_ksize=15):
    """
    稳健版：仅将 attn 中 > heat_threshold 的像素对原图对应位置进行黑色加深处理（让区域更黑）
    核心修改：输出RGBA格式，复用原始A通道
    Args:
        attn_tensor: 注意力张量
        mask_tensor: 掩膜张量
        save_path: 保存路径
        original_img: 原始图像（PIL或numpy数组）
        original_alpha: 原始A通道（numpy数组，来自输入RGBA）
        keep_white: 掩膜区域是否显示白色
        heat_threshold: 注意力阈值
        blend_alpha: 混合透明度
        enhance_strength: 加深强度
        soften: 是否软化掩码
        soften_ksize: 软化核大小
    Returns:
        处理后的RGBA图像（numpy数组）
    """
    # 创建保存目录
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # 验证输入图像
    if original_img is None:
        raise ValueError("original_img 不能为空。")
    if isinstance(original_img, Image.Image):
        img_rgb = np.array(original_img.convert("RGB"))
    else:
        img_rgb = np.asarray(original_img)
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:
            img_rgb = img_rgb[..., :3]

    # 验证A通道（确保输出RGBA）
    if original_alpha is None:
        # 若未提供原始A通道，默认全不透明（兼容旧逻辑）
        original_alpha = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8) * 255
    else:
        # 调整A通道尺寸与RGB一致
        original_alpha = cv2.resize(original_alpha, (img_rgb.shape[1], img_rgb.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

    # 转换颜色空间
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # 注意力张量预处理
    if isinstance(attn_tensor, torch.Tensor):
        attn_np = attn_tensor.detach().cpu().numpy()
    else:
        attn_np = np.asarray(attn_tensor)

    # 降维处理
    while attn_np.ndim > 2:
        attn_np = attn_np.mean(axis=0)
    attn_np = attn_np.astype(np.float32)

    # 处理异常值
    if not np.isfinite(attn_np).all():
        attn_np = np.nan_to_num(attn_np, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.nanmin(attn_np)) if attn_np.size else 0.0
    mx = float(np.nanmax(attn_np)) if attn_np.size else 0.0
    if mx - mn < 1e-8:
        attn_norm = np.zeros_like(attn_np, dtype=np.float32)
    else:
        attn_norm = (attn_np - mn) / (mx - mn)

    # 调整尺寸
    attn_resized = cv2.resize(attn_norm.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    # 生成热力掩码（越冷的地方（attn越小），越加深）
    heat_mask = (attn_resized < float(heat_threshold))  # 冷区域掩码
    if soften:
        k = soften_ksize if (soften_ksize % 2 == 1) else (soften_ksize + 1)
        heat_mask_float = cv2.GaussianBlur(heat_mask.astype(np.float32), (k, k), 0)
        heat_alpha = np.clip(heat_mask_float, 0.0, 1.0) * float(blend_alpha)
    else:
        heat_alpha = heat_mask.astype(np.float32) * float(blend_alpha)

    # -------------------------- 核心黑色加深逻辑 --------------------------
    black_template = np.zeros_like(img_bgr, dtype=np.uint8)
    darken_factor = 1.0 + (enhance_strength - 1.0) * heat_alpha  # 冷区域加深系数更大
    darken_factor = np.clip(darken_factor, 1.0, None)

    img_float = img_bgr.astype(np.float32)
    darken_factor_3d = darken_factor[..., np.newaxis]
    darkened_bgr = np.clip(img_float / darken_factor_3d, 0, 255).astype(np.uint8)
    # --------------------------------------------------------------------------

    # 应用加深效果
    overlay = img_bgr.copy()
    mask_idxs = heat_mask
    if mask_idxs.any():
        original_pixels = img_bgr[mask_idxs].astype(np.float32)
        darkened_pixels = darkened_bgr[mask_idxs].astype(np.float32)
        alpha_values = heat_alpha[mask_idxs][:, np.newaxis]
        blended_pixels = (original_pixels * (1.0 - alpha_values) +
                          darkened_pixels * alpha_values).astype(np.uint8)
        overlay[mask_idxs] = blended_pixels

    # 应用掩膜着色
    if mask_tensor is not None:
        if isinstance(mask_tensor, torch.Tensor):
            mask_np = mask_tensor.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask_tensor)

        mask_np = np.squeeze(mask_np)
        while mask_np.ndim > 2:
            mask_np = mask_np.any(axis=0)
        mask_resized = cv2.resize(mask_np.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        if mask_resized.any():
            overlay[mask_resized] = (255, 255, 255) if keep_white else (0, 0, 0)

    # -------------------------- 核心修改：输出RGBA --------------------------
    # 转换回RGB，合并原始A通道
    # overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay_rgba = np.dstack((overlay, original_alpha))  # 合并RGB+原始A通道
    # --------------------------------------------------------------------------

    # 保存RGBA图像（必须为PNG格式，JPG不支持透明）
    overlay_rgba = overlay_rgba.astype(np.uint8)
    cv2.imwrite(save_path, overlay_rgba)
    print(f"已保存RGBA结果图: {save_path}")

    return overlay_rgba


def find_mask_for_image(img_path, mask_dir):
    """
    根据图像文件名找到对应的掩膜文件
    适配命名约定：image: '20160222_080850.jpg' -> mask: '20160222_080850_mask.png'
    Args:
        img_path: 图像路径
        mask_dir: 掩膜目录
    Returns:
        找到的掩膜路径（None表示未找到）
    """
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]

    # 可能的掩膜文件名模式
    possible_mask_names = [
        f"{base_name}_mask.png",
        f"{base_name}_mask.jpg",
        f"{base_name}.png",
        f"{base_name}.jpg",
        base_name + '.png',
        base_name + '.jpg'
    ]

    for mask_name in possible_mask_names:
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            return mask_path

    return None


def calculate_average_metrics(metrics_list):
    """
    计算指标列表的平均值，只处理数值类型的指标
    Args:
        metrics_list: 指标字典列表
    Returns:
        平均指标字典
    """
    if not metrics_list:
        return {}

    # 获取所有数值类型的键
    numeric_keys = []
    for key in metrics_list[0].keys():
        if isinstance(metrics_list[0][key], (int, float, np.number)):
            numeric_keys.append(key)

    # 计算平均值
    avg_metrics = {}
    for key in numeric_keys:
        try:
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
        except (TypeError, ValueError):
            continue

    return avg_metrics


@torch.no_grad()
def test_fgem_with_mask(
        model_path,
        input_path,
        mask_dir=None,
        device="cuda",
        save_dir_input="./run/predict_heatmap/input",
        save_dir_output="./run/predict_heatmap/output",
        save_dir_orig="./run/predict_images/original",
        save_dir_result="./run/predict_images/result",
        save_dir_metrics="./run/metrics",
        eval_threshold=0.25
):
    """
    测试FGEM模型（带掩膜评估）
    核心修改：
    1. 输入RGBA图像，用A通道掩膜RGB后输入模型
    2. 输出RGBA格式结果（复用原始A通道）
    Args:
        model_path: 模型权重路径
        input_path: 输入图像路径（文件或目录，RGBA格式）
        mask_dir: 掩膜目录（None表示不评估）
        device: 运行设备（cuda/cpu）
        save_dir_input: 输入热图保存目录
        save_dir_output: 输出热图保存目录
        save_dir_orig: 原始图像保存目录
        save_dir_result: 结果图像保存目录（RGBA格式）
        save_dir_metrics: 指标保存目录
        eval_threshold: 评估阈值
    """
    # 加载模型
    model = load_model(model_path, device)

    # 初始化指标存储
    all_overall_metrics = []
    all_class_metrics = {0: [], 1: []}
    metrics_history = {
        'overall_accuracy': [], 'overall_iou': [], 'mean_iou': [], 'overall_dice': [],
        'macro_precision': [], 'macro_recall': [], 'macro_f1': [],
        'weighted_precision': [], 'weighted_recall': [], 'weighted_f1': [],
        'class_0_precision': [], 'class_0_recall': [], 'class_0_f1': [], 'class_0_iou': [],
        'class_1_precision': [], 'class_1_recall': [], 'class_1_f1': [], 'class_1_iou': []
    }
    speed_metrics = {
        'inference_times': [], 'fps_list': [], 'preprocess_times': [], 'postprocess_times': []
    }
    all_predictions = []
    all_targets = []

    # 获取图像列表
    if os.path.isdir(input_path):
        img_list = [os.path.join(input_path, f)
                    for f in os.listdir(input_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # 包含常见图像格式
    else:
        img_list = [input_path]

    valid_count = 0
    missing_masks = []

    # 批量处理图像
    for img_path in tqdm(img_list, desc="Testing FGEM"):
        # 查找对应的掩膜文件
        mask_path = find_mask_for_image(img_path, mask_dir) if mask_dir else None
        if mask_dir and mask_path is None:
            missing_masks.append(os.path.basename(img_path))
            print(f"⚠️  未找到对应掩膜文件: {os.path.basename(img_path)}")

        # 预处理计时（核心修改：获取原始A通道）
        preprocess_start = time.perf_counter()
        original_img, tensor, ignore_mask, mask_binary, original_alpha = preprocess_image_and_mask(
            img_path, mask_path, ignore_value=None
        )
        tensor = tensor.to(device)
        ignore_mask = ignore_mask.to(device)
        preprocess_time = time.perf_counter() - preprocess_start
        speed_metrics['preprocess_times'].append(preprocess_time)

        # 推理计时
        inference_start = time.perf_counter()
        out = model(tensor)
        attn_map = out.mean(dim=1, keepdim=True)
        inference_time = time.perf_counter() - inference_start
        speed_metrics['inference_times'].append(inference_time)
        speed_metrics['fps_list'].append(1.0 / inference_time)

        # 后处理计时
        postprocess_start = time.perf_counter()

        # 保存原始图像（保留RGBA格式）
        orig_save_path = os.path.join(save_dir_orig, os.path.basename(img_path))
        os.makedirs(os.path.dirname(orig_save_path), exist_ok=True)
        # 读取原始RGBA图像并保存（确保原始透明度不变）
        original_rgba = Image.open(img_path).convert("RGBA")
        original_rgba.save(orig_save_path)

        # 保存结果图像（核心修改：传入原始A通道，输出RGBA）
        result_save_path = os.path.join(save_dir_result, os.path.splitext(os.path.basename(img_path))[0] + ".png")
        save_image(
            attn_map, ignore_mask, result_save_path,
            original_img=original_img,
            original_alpha=original_alpha,  # 传入原始A通道
            keep_white=True
        )

        # 保存输入热图
        input_gray = tensor.mean(dim=1, keepdim=True)
        input_heatmap_path = os.path.join(
            save_dir_input, f"{os.path.splitext(os.path.basename(img_path))[0]}_input_heatmap.png"
        )
        save_pure_heatmap(input_gray, original_img, input_heatmap_path)

        # 保存输出热图
        output_heatmap_path = os.path.join(
            save_dir_output, f"{os.path.splitext(os.path.basename(img_path))[0]}_output_heatmap.png"
        )
        save_pure_heatmap(attn_map, original_img, output_heatmap_path)

        postprocess_time = time.perf_counter() - postprocess_start
        speed_metrics['postprocess_times'].append(postprocess_time)

        # 计算评估指标（有掩膜时）
        if mask_binary is not None:
            # 调整预测结果尺寸
            pred_resized = torch.nn.functional.interpolate(
                attn_map, size=mask_binary.shape[2:], mode='bilinear', align_corners=False
            )

            # 计算指标
            metrics_result = calculate_classification_metrics(pred_resized, mask_binary, threshold=eval_threshold)

            # 更新指标存储
            all_overall_metrics.append(metrics_result['overall'])
            for class_id in [0, 1]:
                all_class_metrics[class_id].append(metrics_result['class_metrics'][class_id])

            # 更新指标历史
            metrics_history['overall_accuracy'].append(metrics_result['overall']['accuracy'])
            metrics_history['overall_iou'].append(metrics_result['overall']['overall_iou'])
            metrics_history['mean_iou'].append(metrics_result['overall']['mean_iou'])
            metrics_history['overall_dice'].append(metrics_result['overall']['dice'])
            metrics_history['macro_precision'].append(metrics_result['overall']['macro_precision'])
            metrics_history['macro_recall'].append(metrics_result['overall']['macro_recall'])
            metrics_history['macro_f1'].append(metrics_result['overall']['macro_f1'])
            metrics_history['weighted_precision'].append(metrics_result['overall']['weighted_precision'])
            metrics_history['weighted_recall'].append(metrics_result['overall']['weighted_recall'])
            metrics_history['weighted_f1'].append(metrics_result['overall']['weighted_f1'])
            metrics_history['class_0_precision'].append(metrics_result['class_metrics'][0]['precision'])
            metrics_history['class_0_recall'].append(metrics_result['class_metrics'][0]['recall'])
            metrics_history['class_0_f1'].append(metrics_result['class_metrics'][0]['f1_score'])
            metrics_history['class_0_iou'].append(metrics_result['class_metrics'][0]['iou'])
            metrics_history['class_1_precision'].append(metrics_result['class_metrics'][1]['precision'])
            metrics_history['class_1_recall'].append(metrics_result['class_metrics'][1]['recall'])
            metrics_history['class_1_f1'].append(metrics_result['class_metrics'][1]['f1_score'])
            metrics_history['class_1_iou'].append(metrics_result['class_metrics'][1]['iou'])

            # 存储预测和目标值
            all_predictions.append(pred_resized)
            all_targets.append(mask_binary)

            valid_count += 1

            # 打印单图结果
            print(f"{os.path.basename(img_path)} - "
                  f"裂缝F1: {metrics_result['class_metrics'][1]['f1_score']:.4f}, "
                  f"背景F1: {metrics_result['class_metrics'][0]['f1_score']:.4f}, "
                  f"宏观F1: {metrics_result['overall']['macro_f1']:.4f}, "
                  f"推理时间: {inference_time:.4f}s, FPS: {1.0 / inference_time:.2f}")

    # 输出汇总结果
    if valid_count > 0:
        print("\n" + "=" * 80)
        print("模型评估结果汇总（分类视角）")
        print("=" * 80)

        # 计算平均指标
        avg_overall = calculate_average_metrics(all_overall_metrics)
        avg_class_0 = calculate_average_metrics(all_class_metrics[0])
        avg_class_1 = calculate_average_metrics(all_class_metrics[1])

        # 计算速度指标
        avg_speed_metrics = {
            'avg_inference_time': np.mean(speed_metrics['inference_times']),
            'std_inference_time': np.std(speed_metrics['inference_times']),
            'avg_fps': np.mean(speed_metrics['fps_list']),
            'std_fps': np.std(speed_metrics['fps_list']),
            'avg_preprocess_time': np.mean(speed_metrics['preprocess_times']),
            'avg_postprocess_time': np.mean(speed_metrics['postprocess_times']),
            'min_inference_time': np.min(speed_metrics['inference_times']),
            'max_inference_time': np.max(speed_metrics['inference_times']),
            'total_inference_time': np.sum(speed_metrics['inference_times'])
        }

        # 打印全局指标
        print(f"测试图像数量: {valid_count}")
        print(f"\n全局指标:")
        print(f"   准确率 (Accuracy): {avg_overall['accuracy']:.4f}")
        print(f"   整体IoU: {avg_overall['overall_iou']:.4f}")
        print(f"   平均IoU: {avg_overall['mean_iou']:.4f}")
        print(f"   Dice系数: {avg_overall['dice']:.4f}")
        print(f"   宏观精确率: {avg_overall['macro_precision']:.4f}")
        print(f"   宏观召回率: {avg_overall['macro_recall']:.4f}")
        print(f"   宏观F1分数: {avg_overall['macro_f1']:.4f}")
        print(f"   加权精确率: {avg_overall['weighted_precision']:.4f}")
        print(f"   加权召回率: {avg_overall['weighted_recall']:.4f}")
        print(f"   加权F1分数: {avg_overall['weighted_f1']:.4f}")

        # 打印类别指标
        print(f"\n背景类别指标 (类别 0):")
        print(f"   精确率: {avg_class_0['precision']:.4f}")
        print(f"   召回率: {avg_class_0['recall']:.4f}")
        print(f"   F1分数: {avg_class_0['f1_score']:.4f}")
        print(f"   IoU: {avg_class_0['iou']:.4f}")
        print(f"   支持度: {avg_class_0['support']:.0f} 像素")

        print(f"\n裂缝类别指标 (类别 1):")
        print(f"   精确率: {avg_class_1['precision']:.4f}")
        print(f"   召回率: {avg_class_1['recall']:.4f}")
        print(f"   F1分数: {avg_class_1['f1_score']:.4f}")
        print(f"   IoU: {avg_class_1['iou']:.4f}")
        print(f"   支持度: {avg_class_1['support']:.0f} 像素")

        # 打印速度指标
        print(f"\n速度性能指标:")
        print(
            f"   平均推理时间: {avg_speed_metrics['avg_inference_time']:.4f}s ± {avg_speed_metrics['std_inference_time']:.4f}s")
        print(f"   最快推理时间: {avg_speed_metrics['min_inference_time']:.4f}s")
        print(f"   最慢推理时间: {avg_speed_metrics['max_inference_time']:.4f}s")
        print(f"   平均FPS: {avg_speed_metrics['avg_fps']:.2f} ± {avg_speed_metrics['std_fps']:.2f}")
        print(f"   平均预处理时间: {avg_speed_metrics['avg_preprocess_time']:.4f}s")
        print(f"   平均后处理时间: {avg_speed_metrics['avg_postprocess_time']:.4f}s")
        print(f"   总推理时间: {avg_speed_metrics['total_inference_time']:.2f}s")

        # 混淆矩阵统计
        total_tp = sum(m['tp'] for m in all_overall_metrics)
        total_fp = sum(m['fp'] for m in all_overall_metrics)
        total_fn = sum(m['fn'] for m in all_overall_metrics)
        total_tn = sum(m['tn'] for m in all_overall_metrics)
        total_pixels = sum(m['total_pixels'] for m in all_overall_metrics)

        print(f"\n混淆矩阵统计:")
        print(f"   真阳性 (TP): {total_tp}")
        print(f"   假阳性 (FP): {total_fp}")
        print(f"   假阴性 (FN): {total_fn}")
        print(f"   真阴性 (TN): {total_tn}")
        print(f"   总像素数: {total_pixels}")

        # 计算PR曲线和AP
        ap, precision_curve, recall_curve = calculate_pr_curve(all_predictions, all_targets)
        print(f"平均精度 (AP): {ap:.4f}")

        # 保存指标结果
        os.makedirs(save_dir_metrics, exist_ok=True)
        metrics_file = os.path.join(save_dir_metrics, "evaluation_results.txt")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write("模型评估结果（分类视角）\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试图像数量: {valid_count}\n")
            f.write(f"评估阈值: {eval_threshold}\n\n")

            f.write("全局指标:\n")
            for key, value in avg_overall.items():
                f.write(f"{key}: {value:.4f}\n")

            f.write(f"\n背景类别指标:\n")
            for key, value in avg_class_0.items():
                f.write(f"{key}: {value:.4f}\n")

            f.write(f"\n裂缝类别指标:\n")
            for key, value in avg_class_1.items():
                f.write(f"{key}: {value:.4f}\n")

            f.write(f"\n速度指标:\n")
            for key, value in avg_speed_metrics.items():
                f.write(f"{key}: {value:.4f}\n")

            f.write(f"\nAP: {ap:.4f}\n")
            f.write(f"\n混淆矩阵:\n")
            f.write(f"TP: {total_tp}\n")
            f.write(f"FP: {total_fp}\n")
            f.write(f"FN: {total_fn}\n")
            f.write(f"TN: {total_tn}\n")
            f.write(f"总像素数: {total_pixels}\n")

            if missing_masks:
                f.write(f"\n缺失掩膜的文件 ({len(missing_masks)} 个):\n")
                for missing in missing_masks:
                    f.write(f"{missing}\n")

        print(f"指标结果已保存: {metrics_file}")

        # 保存可视化图表
        elegant_plots = create_elegant_metrics_plots(metrics_history, speed_metrics, save_dir_metrics)
        print(f"优雅指标图表已保存: {elegant_plots}")

        # 保存PR曲线
        plt.figure(figsize=(10, 8))
        plt.plot(recall_curve, precision_curve, 'b-', linewidth=2, label=f'PR curve (AP = {ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        pr_curve_path = os.path.join(save_dir_metrics, "pr_curve.png")
        plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PR曲线已保存: {pr_curve_path}")

        # 打印缺失掩膜信息
        if missing_masks:
            print(f"\n有 {len(missing_masks)} 个图像未找到对应掩膜文件")

    else:
        print("未找到有效的掩膜文件，跳过指标计算")


# 配置路径
pre_dir = 'D:/myDataManager/pycharmProject/Crack-Segmentation/road_roi_net/RoadDataset/train/paper3_select/ROI/123___jpg__/frames'

if __name__ == "__main__":
    # 测试配置（默认注释一个，启用另一个）
    # test_fgem_with_mask(
    #     model_path="./checkpoints/best_checkpoint.pth",
    #     input_path=r'D:/myDataManager/pycharmProject/Crack-Segmentation/FGEM/IOT_duikangGenerateNet/images',
    #     mask_dir=r'D:/myDataManager/pycharmProject/Crack-Segmentation/FGEM/IOT_duikangGenerateNet/images',
    #     device="cuda:0" if torch.cuda.is_available() else "cpu",
    #     save_dir_input=pre_dir + "/predict_heatmap/input",
    #     save_dir_output=pre_dir + "/predict_heatmap/output",
    #     save_dir_orig=pre_dir + "/predict_images/original",
    #     save_dir_result=pre_dir + "/predict_images/result",
    #     save_dir_metrics=pre_dir + "/metrics",
    #     eval_threshold=0.25
    # )

    test_fgem_with_mask(
        model_path="./checkpoints/best_checkpoint.pth",
        input_path=r'D:\myDataManager\pycharmProject\Crack-Segmentation\road_roi_net\RoadDataset\train\paper3_select\ROI\123___jpg__\frames',
        mask_dir=r'D:\myDataManager\pycharmProject\Crack-Segmentation\road_roi_net\RoadDataset\train\paper3_select\ROI\123___jpg__\frames',
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        save_dir_input=pre_dir + "/predict_heatmap/input",
        save_dir_output=pre_dir + "/predict_heatmap/output",
        save_dir_orig=pre_dir + "/predict_images/original",
        save_dir_result=pre_dir + "/predict_images/result",
        save_dir_metrics=pre_dir + "/metrics",
        eval_threshold=0.25
    )