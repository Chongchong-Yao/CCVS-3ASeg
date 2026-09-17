import torch
import torch.nn as nn
import torch.nn.functional as F


class CrackAdversarialLoss(nn.Module):
    """道路裂缝检测的对抗损失函数 - 修复设备问题"""

    def __init__(self, lambda_adv=1.0, lambda_content=10.0, lambda_edge=5.0, device='cuda', threshold=0.25):
        super(CrackAdversarialLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_content = lambda_content
        self.lambda_edge = lambda_edge
        self.device = device
        self.threshold = threshold  # 添加这一行

        # 简单的PatchGAN判别器
        self.discriminator = CrackDiscriminator(in_channels=1)

        # 将判别器移动到设备
        self.discriminator = self.discriminator.to(self.device)

        # 内容损失
        self.l1_loss = nn.L1Loss()

        # 边缘特异性损失
        self.edge_loss = EdgeAwareLoss()

        # 将边缘损失也移动到设备
        self.edge_loss = self.edge_loss.to(self.device)

    def generator_loss(self, pred_edges, real_edges, images):
        """生成器总损失"""
        # 确保输入是单通道
        if pred_edges.shape[1] != 1:
            pred_edges = pred_edges.mean(dim=1, keepdim=True)
        if real_edges.shape[1] != 1:
            real_edges = real_edges.mean(dim=1, keepdim=True)

        # 1. 对抗损失 - 让判别器认为生成的边缘是真实的
        d_fake = self.discriminator(pred_edges)
        adv_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))

        # 2. 内容损失 - 保持边缘的基本准确性
        content_loss = self.l1_loss(pred_edges, real_edges)

        # 3. 边缘感知损失 - 强化裂缝的连续性
        edge_aware_loss = self.edge_loss(pred_edges, real_edges, images)

        # 总损失
        total_loss = (self.lambda_adv * adv_loss +
                      self.lambda_content * content_loss +
                      self.lambda_edge * edge_aware_loss)

        return total_loss, {
            'adv_loss': adv_loss.item(),
            'content_loss': content_loss.item(),
            'edge_loss': edge_aware_loss.item(),
            'total_loss': total_loss.item()
        }

    def discriminator_loss(self, pred_edges, real_edges):
        """判别器损失"""
        # 确保输入是单通道
        if pred_edges.shape[1] != 1:
            pred_edges = pred_edges.mean(dim=1, keepdim=True)
        if real_edges.shape[1] != 1:
            real_edges = real_edges.mean(dim=1, keepdim=True)

        # ==== 只添加这两行 ====
        # 对生成器输出进行二值化
        binary_pred_edges = (pred_edges >= self.threshold).float().detach()

        # 真实样本应该被判别为1
        d_real = self.discriminator(real_edges)
        real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))

        # 生成样本应该被判别为0
        d_fake = self.discriminator(binary_pred_edges.detach())
        fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))

        # 总判别器损失
        disc_loss = (real_loss + fake_loss) / 2

        return disc_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'disc_loss': disc_loss.item()
        }


class CrackDiscriminator(nn.Module):
    """简单的裂缝判别器 - 修复版本"""

    def __init__(self, in_channels=1, base_channels=64):
        super(CrackDiscriminator, self).__init__()

        # 明确指定输入通道数
        self.in_channels = in_channels

        self.net = nn.Sequential(
            # 输入: (B, in_channels, H, W)
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),  # 1/2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),  # 1/4
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),  # 1/8
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=1, padding=1),  # 1/8
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出: (B, 1, H/8, W/8) 的patch真实性判断
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1)
        )

        # 应用谱归一化稳定训练
        self._apply_spectral_norm()

    def _apply_spectral_norm(self):
        """为判别器添加谱归一化"""
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                try:
                    nn.utils.spectral_norm(layer)
                except:
                    # 如果谱归一化失败，跳过
                    pass

    def forward(self, x):
        # 确保输入通道正确
        if x.shape[1] != self.in_channels:
            # 如果通道不匹配，转换为单通道
            x = x.mean(dim=1, keepdim=True)
        return torch.sigmoid(self.net(x))


class EdgeAwareLoss(nn.Module):
    """边缘感知损失 - 专门针对裂缝连续性优化"""

    def __init__(self, alpha=0.8):
        super(EdgeAwareLoss, self).__init__()
        self.alpha = alpha

        # Sobel算子用于梯度计算
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)

        # 初始化Sobel核
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)

        self.sobel_x.weight.data = sobel_x_kernel
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.data = sobel_y_kernel
        self.sobel_y.weight.requires_grad = False

    def gradient_loss(self, pred, target):
        """梯度一致性损失"""
        # 确保单通道输入
        if pred.shape[1] != 1:
            pred = pred.mean(dim=1, keepdim=True)
        if target.shape[1] != 1:
            target = target.mean(dim=1, keepdim=True)

        pred_grad_x = self.sobel_x(pred)
        pred_grad_y = self.sobel_y(pred)
        target_grad_x = self.sobel_x(target)
        target_grad_y = self.sobel_y(target)

        grad_loss = (F.l1_loss(pred_grad_x, target_grad_x) +
                     F.l1_loss(pred_grad_y, target_grad_y)) / 2
        return grad_loss

    def continuity_loss(self, pred):
        """裂缝连续性损失 - 惩罚断裂的裂缝"""
        # 确保单通道输入
        if pred.shape[1] != 1:
            pred = pred.mean(dim=1, keepdim=True)

        # 计算连通组件（简化版本）
        binary_pred = (pred > 0).float()

        # 使用形态学操作估计连通性
        kernel = torch.ones(1, 1, 3, 3).to(pred.device)
        dilated = F.conv2d(binary_pred, kernel, padding=1)
        eroded = F.conv2d(binary_pred, kernel, padding=1)

        # 边界区域
        boundary = (dilated > 0).float() - (eroded > 0).float()

        # 连续性损失：鼓励边界点有邻居
        connectivity = F.conv2d(binary_pred, kernel, padding=1)
        continuity_loss = F.mse_loss(connectivity * boundary, torch.ones_like(connectivity) * boundary)

        return continuity_loss

    def forward(self, pred_edges, real_edges, images):
        """边缘感知总损失"""
        # 梯度一致性
        grad_loss = self.gradient_loss(pred_edges, real_edges)

        # 裂缝连续性
        continuity_loss = self.continuity_loss(pred_edges)

        return self.alpha * grad_loss + (1 - self.alpha) * continuity_loss


# ===== 使用示例 =====
class CrackGAN_Trainer:
    """完整的裂缝GAN训练器 - 修复版本"""

    def __init__(self, generator, lr_g=1e-4, lr_d=2e-4, device='cuda'):
        self.generator = generator
        self.device = device

        # 将生成器移动到设备
        self.generator = self.generator.to(self.device)

        # 初始化损失函数
        self.loss_fn = CrackAdversarialLoss(
            lambda_adv=0.1,      # 对抗损失权重降低
            lambda_content=10.0, # 内容损失权重保持
            lambda_edge=2.0      # 边缘损失权重适中
        )

        # 将判别器也移动到设备
        self.loss_fn.discriminator = self.loss_fn.discriminator.to(self.device)

        # 优化器
        self.optimizer_g = torch.optim.Adam(
            generator.parameters(), lr=lr_g, betas=(0.5, 0.999)
        )
        self.optimizer_d = torch.optim.Adam(
            self.loss_fn.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999)
        )

    def train_step(self, images, real_edges):
        """单步训练"""
        # 生成假边缘
        fake_edges = self.generator(images)

        # 训练判别器
        self.optimizer_d.zero_grad()
        d_loss, d_log = self.loss_fn.discriminator_loss(fake_edges, real_edges)
        d_loss.backward()
        self.optimizer_d.step()

        # 训练生成器
        self.optimizer_g.zero_grad()
        g_loss, g_log = self.loss_fn.generator_loss(fake_edges, real_edges, images)
        g_loss.backward()
        self.optimizer_g.step()

        return {
            'generator': g_log,
            'discriminator': d_log,
            'fake_edges': fake_edges.detach()
        }


# ===== 测试代码 =====
if __name__ == "__main__":
    # 测试损失函数
    from GenerateEEMNet import EEMLite_Generator  # 导入你的生成器

    generator = EEMLite_Generator(3, 1)  # 你的生成器
    trainer = CrackGAN_Trainer(generator)

    # 模拟数据 - 确保real_edges是单通道
    batch_size = 2
    images = torch.randn(batch_size, 3, 512, 640)
    real_edges = torch.randn(batch_size, 1, 512, 640)  # 单通道

    # 测试训练步骤
    try:
        result = trainer.train_step(images, real_edges)
        print("训练步骤完成!")
        print("生成器损失:", result['generator'])
        print("判别器损失:", result['discriminator'])
        print("生成边缘形状:", result['fake_edges'].shape)
    except Exception as e:
        print(f"错误: {e}")
        print("检查通道数匹配...")

        # 调试信息
        fake_edges = generator(images)
        print(f"images形状: {images.shape}")
        print(f"real_edges形状: {real_edges.shape}")
        print(f"fake_edges形状: {fake_edges.shape}")