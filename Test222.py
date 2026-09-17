import torch
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from FGEM.IOT_duikangGenerateNet.utils.test_hok import FeatureMapVisualizer
from GenerateEEMNet117 import EEMLite_Generator
model_path="./checkpoints/best_checkpoint.pth",
#from GenerateEEMNet_V2 import EEM_Generator


def load_model(model_path, device="cuda"):
    model = EEMLite_Generator(ch_in=3, ch_out=1)
    checkpoint = torch.load(model_path, map_location=device)
    print("权重文件中的键名：", checkpoint.keys())
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.to(device)
    model.eval()
    print(f"成功加载模型权重: {model_path}")
    return model


def preprocess_image_and_mask(img_path, mask_path, size=(512, 640), ignore_value=None):
    """
    预处理图片和对应的标签掩膜
    """
    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # [1,3,H,W]

    if ignore_value is not None:
        ignore_norm = ignore_value / 255.0
        mask = (tensor == ignore_norm)
        tensor = tensor.clone()
        tensor[mask] = 0.0
        mask_any = mask.any(dim=1, keepdim=True)  # [1,1,H,W]
    else:
        mask_any = torch.zeros((1, 1, *tensor.shape[2:]), dtype=torch.bool)

    # 预处理掩膜
    if mask_path and os.path.exists(mask_path):
        mask_img = Image.open(mask_path).convert("L")  # 转为灰度图
        mask_tensor = transforms.Resize(size)(mask_img)
        mask_tensor = transforms.ToTensor()(mask_tensor).unsqueeze(0)  # [1,1,H,W]
        # 二值化掩膜：裂缝区域为1，背景为0
        mask_binary = (mask_tensor > 0).float()
    else:
        mask_binary = None

    return img, tensor, mask_any, mask_binary


def calculate_classification_metrics(prediction, target, threshold=0.25):
    """
    计算多类别分类指标（背景类=0，裂缝类=1）
    """
    # 确保数据是numpy数组
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # 展平数组
    pred_flat = prediction.flatten()
    target_flat = target.flatten()

    # 二值化预测结果
    pred_binary = (pred_flat < threshold).astype(np.uint8)
    target_binary = (target_flat > 0).astype(np.uint8)

    # 计算混淆矩阵
    cm = confusion_matrix(target_binary, pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 计算总体指标
    total_pixels = len(pred_flat)
    accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0

    # 计算每个类别的指标
    metrics_by_class = {}

    # 背景类 (class 0)
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    support_0 = tn + fp  # 真实背景像素数量

    metrics_by_class[0] = {
        'precision': precision_0,
        'recall': recall_0,
        'f1_score': f1_0,
        'support': support_0
    }

    # 裂缝类 (class 1)
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    support_1 = tp + fn  # 真实裂缝像素数量

    metrics_by_class[1] = {
        'precision': precision_1,
        'recall': recall_1,
        'f1_score': f1_1,
        'support': support_1
    }

    # 计算宏观平均指标
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    # 计算加权平均指标
    weight_0 = support_0 / total_pixels
    weight_1 = support_1 / total_pixels
    weighted_precision = precision_0 * weight_0 + precision_1 * weight_1
    weighted_recall = recall_0 * weight_0 + recall_1 * weight_1
    weighted_f1 = f1_0 * weight_0 + f1_1 * weight_1

    # 计算IoU指标
    intersection_0 = tn
    union_0 = tn + fn + fp
    iou_0 = intersection_0 / union_0 if union_0 > 0 else 0

    intersection_1 = tp
    union_1 = tp + fp + fn
    iou_1 = intersection_1 / union_1 if union_1 > 0 else 0

    # 添加IoU到类别指标
    metrics_by_class[0]['iou'] = iou_0
    metrics_by_class[1]['iou'] = iou_1

    # 计算平均IoU
    mean_iou = (iou_0 + iou_1) / 2

    # 计算整体IoU (所有类别的交集/并集)
    intersection_total = tp + tn
    union_total = tp + fp + fn + tn
    overall_iou = intersection_total / union_total if union_total > 0 else 0

    # 计算Dice系数
    dice = 2 * (tp + tn) / (2 * (tp + tn) + fp + fn) if (2 * (tp + tn) + fp + fn) > 0 else 0

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
        'class_metrics': metrics_by_class
    }


def calculate_pr_curve(predictions, targets):
    """
    计算精确率-召回率曲线和AP值
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

    # 计算AP
    ap = average_precision_score(all_targets, all_preds)

    # 计算PR曲线
    precision_vals, recall_vals, _ = precision_recall_curve(all_targets, all_preds)

    return ap, precision_vals, recall_vals


def create_elegant_metrics_plots(metrics_history, speed_metrics, save_dir):
    """
    Create elegant metrics visualization charts - English version
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1. Main performance metrics trend chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Metrics Trend', fontsize=16, fontweight='bold')

    # Accuracy and F1 score
    if metrics_history['overall_accuracy'] and metrics_history['class_1_f1']:
        x_range = range(1, len(metrics_history['overall_accuracy']) + 1)
        axes[0, 0].plot(x_range, metrics_history['overall_accuracy'], 'o-', linewidth=2, markersize=4, label='Accuracy')
        axes[0, 0].plot(x_range, metrics_history['class_1_f1'], 's-', linewidth=2, markersize=4, label='Crack F1')
        axes[0, 0].plot(x_range, metrics_history['class_0_f1'], '^-', linewidth=2, markersize=4, label='Background F1')
        axes[0, 0].plot(x_range, metrics_history['macro_f1'], 'v-', linewidth=2, markersize=4, label='Macro F1')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Accuracy & F1 Score Trend')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Precision and Recall
    if metrics_history['class_1_precision'] and metrics_history['class_1_recall']:
        x_range = range(1, len(metrics_history['class_1_precision']) + 1)
        axes[0, 1].plot(x_range, metrics_history['class_1_precision'], 'o-', linewidth=2, markersize=4,
                        label='Crack Precision')
        axes[0, 1].plot(x_range, metrics_history['class_1_recall'], 's-', linewidth=2, markersize=4, label='Crack Recall')
        axes[0, 1].plot(x_range, metrics_history['class_0_precision'], '^-', linewidth=2, markersize=4,
                        label='Background Precision')
        axes[0, 1].plot(x_range, metrics_history['class_0_recall'], 'd-', linewidth=2, markersize=4, label='Background Recall')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Precision & Recall Trend')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # IoU and Dice coefficient
    if metrics_history['class_1_iou'] and metrics_history['overall_dice']:
        x_range = range(1, len(metrics_history['class_1_iou']) + 1)
        axes[1, 0].plot(x_range, metrics_history['class_1_iou'], 'o-', linewidth=2, markersize=4, label='Crack IoU')
        axes[1, 0].plot(x_range, metrics_history['class_0_iou'], 's-', linewidth=2, markersize=4, label='Background IoU')
        axes[1, 0].plot(x_range, metrics_history['mean_iou'], '^-', linewidth=2, markersize=4, label='Mean IoU')
        axes[1, 0].plot(x_range, metrics_history['overall_dice'], 'v-', linewidth=2, markersize=4, label='Dice Coefficient')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Segmentation Metrics Trend')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Inference speed trend
    if speed_metrics['inference_times']:
        x_range = range(1, len(speed_metrics['inference_times']) + 1)
        axes[1, 1].plot(x_range, speed_metrics['inference_times'], '^-', color='purple',
                        linewidth=2, markersize=4, label='Inference Time')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Inference Time Trend')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    performance_plot_path = os.path.join(save_dir, "elegant_performance_metrics.png")
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. Speed metrics summary chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Inference Speed Analysis', fontsize=16, fontweight='bold')

    # Inference time distribution
    if speed_metrics['inference_times']:
        axes[0].hist(speed_metrics['inference_times'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(speed_metrics['inference_times']), color='red', linestyle='--',
                        linewidth=2, label=f'Average: {np.mean(speed_metrics["inference_times"]):.4f}s')
        axes[0].set_xlabel('Inference Time (seconds)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Inference Time Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # FPS distribution
    if speed_metrics['fps_list']:
        axes[1].hist(speed_metrics['fps_list'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].axvline(np.mean(speed_metrics['fps_list']), color='red', linestyle='--',
                        linewidth=2, label=f'Average: {np.mean(speed_metrics["fps_list"]):.2f}')
        axes[1].set_xlabel('FPS')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('FPS Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    speed_plot_path = os.path.join(save_dir, "elegant_speed_analysis.png")
    plt.savefig(speed_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 3. Class performance comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Class Performance Comparison', fontsize=16, fontweight='bold')

    # Precision and Recall comparison
    categories = ['Background', 'Crack']
    precision_values = [np.mean(metrics_history['class_0_precision']), np.mean(metrics_history['class_1_precision'])]
    recall_values = [np.mean(metrics_history['class_0_recall']), np.mean(metrics_history['class_1_recall'])]

    x = np.arange(len(categories))
    width = 0.35

    axes[0].bar(x - width / 2, precision_values, width, label='Precision', alpha=0.7)
    axes[0].bar(x + width / 2, recall_values, width, label='Recall', alpha=0.7)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Precision and Recall by Class')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 score and IoU comparison
    f1_values = [np.mean(metrics_history['class_0_f1']), np.mean(metrics_history['class_1_f1'])]
    iou_values = [np.mean(metrics_history['class_0_iou']), np.mean(metrics_history['class_1_iou'])]

    axes[1].bar(x - width / 2, f1_values, width, label='F1 Score', alpha=0.7)
    axes[1].bar(x + width / 2, iou_values, width, label='IoU', alpha=0.7)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Score')
    axes[1].set_title('F1 Score and IoU by Class')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    class_comparison_path = os.path.join(save_dir, "class_performance_comparison.png")
    plt.savefig(class_comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Elegant metrics plots saved to: {save_dir}")
    return [performance_plot_path, speed_plot_path, class_comparison_path]


def save_pure_heatmap(data, original_img, save_path):
    """
    保存纯热图（不叠加原图），尺寸与原图一致
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(data, torch.Tensor):
        data = data.squeeze().cpu().numpy()

    norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    heatmap_uint8 = (norm * 255).astype(np.uint8)

    orig_w, orig_h = original_img.size
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(save_path, heatmap_color)
    print(f"已保存纯热图: {save_path}")








def save_image(attn_tensor, mask_tensor, save_path, original_img=None,
               keep_white=True, heat_threshold=0.25, blend_alpha=0.8,
               enhance_strength=3.5, soften=False, soften_ksize=15):
    """
    稳健版：仅将 attn 中 > heat_threshold 的像素对原图对应位置进行黑色加深处理（让区域更黑）
    """
    import os
    import numpy as np
    import cv2
    from PIL import Image
    import torch

    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

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

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    if isinstance(attn_tensor, torch.Tensor):
        attn_np = attn_tensor.detach().cpu().numpy()
    else:
        attn_np = np.asarray(attn_tensor)

    while attn_np.ndim > 2:
        attn_np = attn_np.mean(axis=0)
    attn_np = attn_np.astype(np.float32)

    if not np.isfinite(attn_np).all():
        attn_np = np.nan_to_num(attn_np, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.nanmin(attn_np)) if attn_np.size else 0.0
    mx = float(np.nanmax(attn_np)) if attn_np.size else 0.0
    if mx - mn < 1e-8:
        attn_norm = np.zeros_like(attn_np, dtype=np.float32)
    else:
        attn_norm = (attn_np - mn) / (mx - mn)

    attn_resized = cv2.resize(attn_norm.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    heat_mask = (attn_resized < float(heat_threshold))
    if soften:
        k = soften_ksize if (soften_ksize % 2 == 1) else (soften_ksize + 1)
        heat_mask_float = cv2.GaussianBlur(heat_mask.astype(np.float32), (k, k), 0)
        heat_alpha = np.clip(heat_mask_float, 0.0, 1.0) * float(blend_alpha)
    else:
        heat_alpha = heat_mask.astype(np.float32) * float(blend_alpha)

    # -------------------------- 核心修改：黑色加深逻辑 --------------------------
    # 1. 生成"纯黑色模板"（与原图同尺寸，所有像素为(0,0,0)）
    black_template = np.zeros_like(img_bgr, dtype=np.uint8)
    # 2. 计算黑色加深系数：系数越大，加深效果越强（1.0=不加深，>1.0=更黑）
    # enhance_strength越大、heat_alpha越高，加深系数越大
    darken_factor = 1.0 + (enhance_strength - 1.0) * heat_alpha
    # 确保系数不小于1.0（避免反向变亮）
    darken_factor = np.clip(darken_factor, 1.0, None)

    # 3. 对原图进行黑色加深：像素值 = 原图像素值 / 加深系数（值越小越黑）
    # 先转float避免整数除法失真，计算后clip到0-255范围
    img_float = img_bgr.astype(np.float32)
    # 扩展系数维度以匹配BGR三通道（(H,W) → (H,W,1)）
    darken_factor_3d = darken_factor[..., np.newaxis]
    # 执行加深计算
    darkened_bgr = np.clip(img_float / darken_factor_3d, 0, 255).astype(np.uint8)
    # --------------------------------------------------------------------------

    # 应用黑色加深效果到注意力区域
    overlay = img_bgr.copy()
    mask_idxs = heat_mask
    if mask_idxs.any():
        # 直接使用数组运算实现加权混合（替代cv2.addWeighted）
        # 提取掩码区域的原图和加深图像素
        original_pixels = img_bgr[mask_idxs].astype(np.float32)
        darkened_pixels = darkened_bgr[mask_idxs].astype(np.float32)
        # 获取对应位置的alpha值并扩展维度以匹配像素形状
        alpha_values = heat_alpha[mask_idxs][:, np.newaxis]
        # 计算混合像素值：alpha越大，加深效果越明显
        blended_pixels = (original_pixels * (1.0 - alpha_values) +
                         darkened_pixels * alpha_values).astype(np.uint8)
        # 将混合结果赋值回原图
        overlay[mask_idxs] = blended_pixels

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
            if keep_white:
                overlay[mask_resized] = (255, 255, 255)
            else:
                overlay[mask_resized] = (0, 0, 0)

    overlay = overlay.astype(np.uint8)
    cv2.imwrite(save_path, overlay)
    print(f"已保存: {save_path}")

    return overlay


def find_mask_for_image(img_path, mask_dir):
    """
    根据图像文件名找到对应的掩膜文件
    适配您的命名约定：image: '20160222_080850.jpg' -> mask: '20160222_080850_mask.png'
    """
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]  # 去掉扩展名，得到 '20160222_080850'

    # 可能的掩膜文件名模式
    possible_mask_names = [
        f"{base_name}_mask.png",  # 您的命名约定
        f"{base_name}_mask.jpg",
        f"{base_name}.png",  # 备用方案
        f"{base_name}.jpg",
        base_name + '.png',  # 直接使用相同文件名
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
    """
    if not metrics_list:
        return {}

    # 获取所有数值类型的键
    numeric_keys = []
    for key in metrics_list[0].keys():
        # 检查第一个元素的值是否为数值类型
        if isinstance(metrics_list[0][key], (int, float, np.number)):
            numeric_keys.append(key)

    # 计算平均值
    avg_metrics = {}
    for key in numeric_keys:
        try:
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
        except (TypeError, ValueError):
            # 如果计算失败，跳过该键
            continue

    return avg_metrics


@torch.no_grad()
def test_fgem_with_mask(
        model_path,
        input_path,
        mask_dir=None,  # 新增：掩膜文件目录
        device="cuda",
        save_dir_input="./run/predict_heatmap/input",
        save_dir_output="./run/predict_heatmap/output",
        save_dir_orig="./run/predict_images/original",
        save_dir_result="./run/predict_images/result",
        save_dir_metrics="./run/metrics",  # 新增：指标保存目录
        eval_threshold=0.25,  # 新增：评估阈值
        save_feature_maps=True,  # 新增：控制是否保存特征图
        max_feature_batches=3    # 新增：最多保存几个batch的特征图
):
    model = load_model(model_path, device)
    # ========== 新增：特征图可视化初始化 ==========
    if save_feature_maps:
        feature_visualizer = FeatureMapVisualizer(
            save_dir=os.path.join(pre_dir, "feature_maps")
        )
        feature_visualizer.register_hooks(model)
    # ===========================================


    # 初始化指标存储
    all_overall_metrics = []
    all_class_metrics = {0: [], 1: []}  # 0:背景, 1:裂缝

    metrics_history = {
        'overall_accuracy': [], 'overall_iou': [], 'mean_iou': [], 'overall_dice': [],
        'macro_precision': [], 'macro_recall': [], 'macro_f1': [],
        'weighted_precision': [], 'weighted_recall': [], 'weighted_f1': [],
        'class_0_precision': [], 'class_0_recall': [], 'class_0_f1': [], 'class_0_iou': [],
        'class_1_precision': [], 'class_1_recall': [], 'class_1_f1': [], 'class_1_iou': []
    }

    # 新增：速度指标存储
    speed_metrics = {
        'inference_times': [],
        'fps_list': [],
        'preprocess_times': [],
        'postprocess_times': []
    }

    all_predictions = []
    all_targets = []

    if os.path.isdir(input_path):
        img_list = [os.path.join(input_path, f)
                    for f in os.listdir(input_path)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    else:
        img_list = [input_path]

    valid_count = 0
    missing_masks = []

    for batch_idx, img_path in enumerate(tqdm(img_list, desc="Testing FGEM")):
        # 构建对应的掩膜路径
        if mask_dir:
            mask_path = find_mask_for_image(img_path, mask_dir)
            if mask_path is None:
                missing_masks.append(os.path.basename(img_path))
                print(f"⚠️  未找到对应掩膜文件: {os.path.basename(img_path)}")
        else:
            mask_path = None

        # 预处理计时
        preprocess_start = time.perf_counter()
        original_img, tensor, ignore_mask, mask_binary = preprocess_image_and_mask(
            img_path, mask_path, ignore_value=None
        )
        tensor = tensor.to(device)
        ignore_mask = ignore_mask.to(device)
        preprocess_time = time.perf_counter() - preprocess_start
        speed_metrics['preprocess_times'].append(preprocess_time)

        # ---- 推理（精确计时）----
        inference_start = time.perf_counter()
        out = model(tensor)
        # attn_map = model.attn_sigmoid(model.attn_conv(out))
        # attn_map = attn_map.mean(dim=1, keepdim=True)
        attn_map = out.mean(dim=1, keepdim=True)

        inference_time = time.perf_counter() - inference_start
        speed_metrics['inference_times'].append(inference_time)
        speed_metrics['fps_list'].append(1.0 / inference_time)

        # ========== 新增：特征图可视化 ==========
        if save_feature_maps and batch_idx < max_feature_batches:
            feature_dir = feature_visualizer.visualize_feature_maps(
                epoch=0,
                batch_idx=batch_idx,
                original_img=original_img,
                img_name=os.path.basename(img_path)
            )
            feature_visualizer.clear_feature_maps()
        # =====================================

        # 后处理计时
        postprocess_start = time.perf_counter()

        # 保存原图
        orig_save_path = os.path.join(save_dir_orig, os.path.basename(img_path))
        os.makedirs(os.path.dirname(orig_save_path), exist_ok=True)
        original_img.save(orig_save_path)

        # 保存结果图
        result_save_path = os.path.join(save_dir_result, os.path.basename(img_path))
        save_image(attn_map, ignore_mask, result_save_path, original_img=original_img, keep_white=True)

        # 保存输入热图
        input_gray = tensor.mean(dim=1, keepdim=True)
        save_pure_heatmap(input_gray, original_img,
                          os.path.join(save_dir_input, os.path.basename(img_path).split(".")[0] + "_input_heatmap.png"))

        # 保存输出热图
        save_pure_heatmap(attn_map, original_img,
                          os.path.join(save_dir_output,
                                       os.path.basename(img_path).split(".")[0] + "_output_heatmap.png"))

        postprocess_time = time.perf_counter() - postprocess_start
        speed_metrics['postprocess_times'].append(postprocess_time)

        # 计算评估指标（如果有掩膜）
        if mask_binary is not None:
            # 将预测结果调整到与掩膜相同的尺寸
            pred_resized = torch.nn.functional.interpolate(
                attn_map, size=mask_binary.shape[2:], mode='bilinear', align_corners=False
            )

            # 计算指标（分类视角）
            metrics_result = calculate_classification_metrics(pred_resized, mask_binary, threshold=eval_threshold)

            # 分别存储各类指标
            all_overall_metrics.append(metrics_result['overall'])

            for class_id in [0, 1]:
                all_class_metrics[class_id].append(metrics_result['class_metrics'][class_id])

            # 更新历史记录
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

            # 类别0（背景）
            metrics_history['class_0_precision'].append(metrics_result['class_metrics'][0]['precision'])
            metrics_history['class_0_recall'].append(metrics_result['class_metrics'][0]['recall'])
            metrics_history['class_0_f1'].append(metrics_result['class_metrics'][0]['f1_score'])
            metrics_history['class_0_iou'].append(metrics_result['class_metrics'][0]['iou'])

            # 类别1（裂缝）
            metrics_history['class_1_precision'].append(metrics_result['class_metrics'][1]['precision'])
            metrics_history['class_1_recall'].append(metrics_result['class_metrics'][1]['recall'])
            metrics_history['class_1_f1'].append(metrics_result['class_metrics'][1]['f1_score'])
            metrics_history['class_1_iou'].append(metrics_result['class_metrics'][1]['iou'])

            # 存储预测和目标值用于PR曲线
            all_predictions.append(pred_resized)
            all_targets.append(mask_binary)

            valid_count += 1

            print(f"{os.path.basename(img_path)} - "
                  f"裂缝F1: {metrics_result['class_metrics'][1]['f1_score']:.4f}, "
                  f"背景F1: {metrics_result['class_metrics'][0]['f1_score']:.4f}, "
                  f"宏观F1: {metrics_result['overall']['macro_f1']:.4f}, "
                  f"推理时间: {inference_time:.4f}s, FPS: {1.0 / inference_time:.2f}")

    # ========== 新增：清理特征图可视化器 ==========
    if save_feature_maps:
        feature_visualizer.remove_hooks()
    # ===========================================

    # 计算总体指标
    if valid_count > 0:
        print("/n" + "=" * 80)
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

        # 打印详细指标
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

        print(f"\n速度性能指标:")
        print(
            f"   平均推理时间: {avg_speed_metrics['avg_inference_time']:.4f}s ± {avg_speed_metrics['std_inference_time']:.4f}s")
        print(f"   最快推理时间: {avg_speed_metrics['min_inference_time']:.4f}s")
        print(f"   最慢推理时间: {avg_speed_metrics['max_inference_time']:.4f}s")
        print(f"   平均FPS: {avg_speed_metrics['avg_fps']:.2f} ± {avg_speed_metrics['std_fps']:.2f}")
        print(f"   平均预处理时间: {avg_speed_metrics['avg_preprocess_time']:.4f}s")
        print(f"   平均后处理时间: {avg_speed_metrics['avg_postprocess_time']:.4f}s")
        print(f"   总推理时间: {avg_speed_metrics['total_inference_time']:.2f}s")

        # 计算总体的TP, FP, FN, TN
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

        # 保存指标到文件
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

            # 添加缺失掩膜文件信息
            if missing_masks:
                f.write(f"\n缺失掩膜的文件 ({len(missing_masks)} 个):\n")
                for missing in missing_masks:
                    f.write(f"{missing}\n")

        print(f"指标结果已保存: {metrics_file}")

        # 保存优雅的指标图表
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







pre_dir = './tem_crack_______1117/test'
if __name__ == "__main__":
    test_fgem_with_mask(
        model_path="./checkpoints/best_checkpoint.pth",
        # input_path=r'./tem_crack_2/images',
        input_path=r'D:\myDataManager\pycharmProject\Crack-Segmentation\road_roi_net\RoadDataset\train\paper3_select\ROI\123___jpg\frames\crop_49png',

        mask_dir=r'D:\myDataManager\pycharmProject\Crack-Segmentation\road_roi_net\RoadDataset\train\paper3_select\ROI\123___jpg\frames\crop_49png',  # 指定掩膜文件目录
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        save_dir_input=  pre_dir + "/predict_heatmap/input",
        save_dir_output= pre_dir + "/predict_heatmap/output",
        save_dir_orig=   pre_dir + "/predict_images/original",
        save_dir_result= pre_dir + "/predict_images/result",
        save_dir_metrics=pre_dir + "/metrics",  # 指标保存目录
        eval_threshold=0.25,  # 评估阈值
        save_feature_maps=True,  # 新增：启用特征图保存
        max_feature_batches=2  # 新增：只保存前2个batch的特征图
    )