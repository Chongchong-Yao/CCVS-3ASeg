
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F


def enhance_image_with_mask(rgb_image, mask_tensor, threshold=0.25, enhance_strength=10.0):
    """
    用KKKModule输出的掩码张量增强RGB图像的指定区域
    :param rgb_image: 输入RGB图像 (PIL Image 或 torch.Tensor [C,H,W] 或 [H,W,C])
    :param mask_tensor: KKKModule输出的单通道张量 (shape=[1,1,H,W] 或 [H,W])，值范围0~1
    :param threshold: 阈值，小于该值的区域将被增强
    :param enhance_strength: 增强强度系数
    :return: 增强后的RGB图像 (PIL Image)
    """
    # 1. 统一输入图像格式
    if isinstance(rgb_image, Image.Image):
        img_np = np.array(rgb_image.convert("RGB"))  # [H,W,3]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # [3,H,W], 0~1
    elif isinstance(rgb_image, torch.Tensor):
        img_tensor = rgb_image.clone().detach()
        if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:  # [3,H,W]
            img_tensor = img_tensor.float() / 255.0 if img_tensor.max() > 1 else img_tensor.float()
        elif img_tensor.dim() == 3 and img_tensor.shape[2] == 3:  # [H,W,3]
            img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
    else:
        img_np = np.array(rgb_image)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

    C, H, W = img_tensor.shape

    # 2. 处理掩码张量
    if isinstance(mask_tensor, torch.Tensor):
        mask = mask_tensor.detach().clone()
        # 去除批次和通道维度
        while mask.dim() > 2:
            mask = mask.squeeze(0)
        # [H,W]
    else:
        mask = torch.from_numpy(np.array(mask_tensor)).float()

    # 调整掩码尺寸与图像一致
    if mask.shape != (H, W):
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear')[0, 0]

    # 生成增强掩码 (小于threshold的区域需要增强)
    enhance_mask = (mask < threshold).float()  # [H,W], 0或1

    # 如果没有需要增强的区域，直接返回原图
    if enhance_mask.sum() == 0:
        return Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    # 3. 对增强区域进行局部对比度增强
    enhanced_img = img_tensor.clone()

    # 方法1: 使用局部对比度增强 (拉普拉斯锐化)
    for c in range(C):
        channel = img_tensor[c].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        # 拉普拉斯锐化核 - 增强边缘和对比度
        laplacian_kernel = torch.tensor([[-1, -1, -1],
                                         [-1, 8, -1],
                                         [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 应用拉普拉斯滤波
        laplacian = F.conv2d(channel, laplacian_kernel, padding=1)

        # 锐化增强: 原图 + 增强系数 * 拉普拉斯结果
        sharpened = channel + enhance_strength * 0.1 * laplacian

        # 只对增强区域应用锐化
        channel_enhanced = channel * (1 - enhance_mask) + sharpened.squeeze() * enhance_mask
        enhanced_img[c] = torch.clamp(channel_enhanced, 0, 1)

    # 方法2: 对增强区域进行饱和度增强 (可选)
    # 转换为HSV空间进行饱和度增强
    enhanced_np = (enhanced_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    hsv = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2HSV)

    enhance_mask_np = enhance_mask.numpy().astype(np.uint8)

    # 对增强区域增加饱和度
    saturation_boost = int(30 * enhance_strength)
    hsv[..., 1] = np.where(enhance_mask_np > 0,
                           np.clip(hsv[..., 1] + saturation_boost, 0, 255),
                           hsv[..., 1])

    # 转换回RGB
    enhanced_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return Image.fromarray(enhanced_rgb)


def test_enhancement():
    """测试函数"""
    # 1. 生成测试数据
    # 创建一张有明显特征的测试图像
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)

    # 添加一些测试图案
    cv2.rectangle(test_img, (50, 50), (100, 100), (255, 0, 0), -1)  # 红色方块
    cv2.rectangle(test_img, (150, 50), (200, 100), (0, 255, 0), -1)  # 绿色方块
    cv2.rectangle(test_img, (50, 150), (100, 200), (0, 0, 255), -1)  # 蓝色方块
    cv2.rectangle(test_img, (150, 150), (200, 200), (128, 128, 128), -1)  # 灰色方块

    # 添加一些纹理
    for i in range(256):
        for j in range(256):
            if (i + j) % 20 < 10:
                test_img[i, j] = np.clip(test_img[i, j] + 30, 0, 255)

    test_img_pil = Image.fromarray(test_img)

    # 2. 生成模拟的KKKModule输出掩码
    # 创建一些低置信度区域 (需要增强的区域)
    mask = torch.ones(256, 256) * 0.8  # 大部分区域高置信度

    # 设置一些低置信度区域 (需要增强)
    mask[30:120, 30:120] = 0.1  # 左上角区域需要增强
    mask[180:220, 100:150] = 0.15  # 小区域需要增强
    mask[100:130, 200:240] = 0.2  # 另一个小区域需要增强

    # 添加一些随机噪声模拟真实输出
    mask += torch.randn(256, 256) * 0.05
    mask = torch.clamp(mask, 0, 1)

    mask_tensor = mask.unsqueeze(0).unsqueeze(0)  # [1,1,256,256]

    print(f"掩码统计: 最小值={mask.min():.3f}, 最大值={mask.max():.3f}, 均值={mask.mean():.3f}")
    print(f"需要增强的像素比例: {(mask < 0.25).float().mean():.3f}")

    # 3. 执行增强
    enhanced_img = enhance_image_with_mask(
        rgb_image=test_img_pil,
        mask_tensor=mask_tensor,
        threshold=0.25,
        enhance_strength=2.0
    )

    # 4. 保存结果
    test_img_pil.save("original_test.png")
    enhanced_img.save("enhanced_test.png")

    # 可视化掩码
    mask_vis = (mask.numpy() * 255).astype(np.uint8)
    enhance_mask_vis = ((mask < 0.25).float().numpy() * 255).astype(np.uint8)

    cv2.imwrite("mask_visualization.png", mask_vis)
    cv2.imwrite("enhance_mask_visualization.png", enhance_mask_vis)

    print("测试完成!")
    print("原图保存为: original_test.png")
    print("增强图保存为: enhanced_test.png")
    print("掩码可视化: mask_visualization.png")
    print("增强区域可视化: enhance_mask_visualization.png")


if __name__ == "__main__":
    test_enhancement()