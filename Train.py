import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from tqdm import tqdm
from GenerateEEMNet_V2 import EEM_Generator  # 你的生成器
from SuperLoss import CrackAdversarialLoss, CrackGAN_Trainer  # 你的损失函数
from DataLoader import DroneVehicleMaskDataset  # 你的数据加载器


class CrackGANTrainer:
    """完整的道路裂缝GAN训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.generator = EEM_Generator(3, 1).to(self.device)
        self.trainer = CrackGAN_Trainer(self.generator,
                                        lr_g=config['lr_g'],
                                        lr_d=config['lr_d'],
                                        device=self.device)

        target_size = (512, 512)  # 平衡细节和计算效率

        # 分别定义图像和mask的变换
        img_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        mask_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        # 数据集和数据加载器
        self.train_dataset = DroneVehicleMaskDataset(
            img_dir=config['train_img_dir'],
            mask_dir=config['train_mask_dir'],
            img_transform=img_transform,
            mask_transform=mask_transform
        )

        self.val_dataset = DroneVehicleMaskDataset(
            img_dir=config['val_img_dir'],
            mask_dir=config['val_mask_dir'],
            img_transform=img_transform,
            mask_transform=mask_transform
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        # 训练记录
        self.writer = SummaryWriter(config['log_dir'])
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)

        print(f"训练设备: {self.device}")
        print(f"训练样本数: {len(self.train_dataset)}")
        print(f"验证样本数: {len(self.val_dataset)}")

    def train_epoch(self):
        """训练一个epoch"""
        self.generator.train()
        self.trainer.loss_fn.discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_adv_loss = 0
        epoch_content_loss = 0
        epoch_edge_loss = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} Training")

        for batch_idx, (images, masks) in enumerate(progress_bar):
            # 移动到设备
            images = images.to(self.device)
            masks = masks.to(self.device)

            # 训练步骤
            result = self.trainer.train_step(images, masks)

            # 累计损失
            epoch_g_loss += result['generator']['total_loss']
            epoch_d_loss += result['discriminator']['disc_loss']
            epoch_adv_loss += result['generator']['adv_loss']
            epoch_content_loss += result['generator']['content_loss']
            epoch_edge_loss += result['generator']['edge_loss']

            # 更新进度条
            progress_bar.set_postfix({
                'G_Loss': f"{result['generator']['total_loss']:.4f}",
                'D_Loss': f"{result['discriminator']['disc_loss']:.4f}",
                'Content': f"{result['generator']['content_loss']:.4f}"
            })

            # 记录到tensorboard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Generator_Total_Loss', result['generator']['total_loss'], global_step)
            self.writer.add_scalar('Train/Discriminator_Loss', result['discriminator']['disc_loss'], global_step)
            self.writer.add_scalar('Train/Adversarial_Loss', result['generator']['adv_loss'], global_step)
            self.writer.add_scalar('Train/Content_Loss', result['generator']['content_loss'], global_step)
            self.writer.add_scalar('Train/Edge_Loss', result['generator']['edge_loss'], global_step)

            # 每100个batch保存一次样本图像
            if batch_idx % 1 == 0:
                self._save_sample_images(images, masks, result['fake_edges'], global_step, 'train')

        # 计算epoch平均损失
        num_batches = len(self.train_loader)
        epoch_g_loss /= num_batches
        epoch_d_loss /= num_batches
        epoch_adv_loss /= num_batches
        epoch_content_loss /= num_batches
        epoch_edge_loss /= num_batches

        return {
            'g_loss': epoch_g_loss,
            'd_loss': epoch_d_loss,
            'adv_loss': epoch_adv_loss,
            'content_loss': epoch_content_loss,
            'edge_loss': epoch_edge_loss
        }

    def validate(self):
        """验证"""
        self.generator.eval()
        self.trainer.loss_fn.discriminator.eval()

        val_losses = {
            'g_loss': 0,
            'd_loss': 0,
            'adv_loss': 0,
            'content_loss': 0,
            'edge_loss': 0
        }

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.val_loader, desc="Validating")):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # 生成假边缘
                fake_edges = self.generator(images)

                # 计算损失
                g_loss, g_log = self.trainer.loss_fn.generator_loss(fake_edges, masks, images)
                d_loss, d_log = self.trainer.loss_fn.discriminator_loss(fake_edges, masks)

                # 累计损失
                val_losses['g_loss'] += g_loss.item()
                val_losses['d_loss'] += d_loss.item()
                val_losses['adv_loss'] += g_log['adv_loss']
                val_losses['content_loss'] += g_log['content_loss']
                val_losses['edge_loss'] += g_log['edge_loss']

                # 保存验证样本
                if batch_idx == 0:
                    self._save_sample_images(images, masks, fake_edges, self.current_epoch, 'val')

        # 计算平均损失
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def _save_sample_images(self, images, real_edges, fake_edges, step, phase):
        """保存样本图像到tensorboard"""
        # 取batch中的前4个样本
        images = images[:4].cpu()
        real_edges = real_edges[:4].cpu()
        fake_edges = fake_edges[:4].cpu()

        # 确保fake_edges是单通道
        if fake_edges.shape[1] != 1:
            fake_edges_single = fake_edges.mean(dim=1, keepdim=True)
        else:
            fake_edges_single = fake_edges

        # 对生成器输出进行二值化（使用相同的阈值0.5）
        binary_fake_edges = (fake_edges_single >= 0.25).float()

        # 反标准化图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)

        # 创建网格 - 添加二值化版本
        self.writer.add_images(f'{phase}/Input_Images', images_denorm, step)
        self.writer.add_images(f'{phase}/Real_Edges', real_edges, step)
        self.writer.add_images(f'{phase}/Fake_Edges', fake_edges, step)
        self.writer.add_images(f'{phase}/Binary_Fake_Edges', binary_fake_edges, step)

        # 保存到文件（可选）
        if step % 500 == 0:
            import matplotlib.pyplot as plt

            # 确定实际可用的样本数量
            num_samples = min(4, images.size(0))

            fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

            # 如果只有一个样本，确保axes是二维的
            if num_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_samples):
                # 输入图像
                axes[i, 0].imshow(images_denorm[i].permute(1, 2, 0))
                axes[i, 0].set_title('Input Image')
                axes[i, 0].axis('off')

                # 真实边缘
                real_edge_display = real_edges[i].squeeze()
                axes[i, 1].imshow(real_edge_display, cmap='gray')
                axes[i, 1].set_title('Real Edges')
                axes[i, 1].axis('off')

                # 生成边缘（连续值）- 确保单通道显示
                fake_edge_display = fake_edges_single[i].squeeze()  # 使用单通道版本
                axes[i, 2].imshow(fake_edge_display, cmap='gray')
                axes[i, 2].set_title('Fake Edges (Continuous)')
                axes[i, 2].axis('off')

                # 生成边缘（二值化）
                binary_display = binary_fake_edges[i].squeeze()
                axes[i, 3].imshow(binary_display, cmap='gray')
                axes[i, 3].set_title('Fake Edges (Binary)')
                axes[i, 3].axis('off')

            plt.tight_layout()
            plt.savefig(f'{self.config["save_dir"]}/samples_epoch_{step}.png', dpi=100, bbox_inches='tight')
            plt.close()

    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.trainer.loss_fn.discriminator.state_dict(),
            'optimizer_g_state_dict': self.trainer.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.trainer.optimizer_d.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # 保存最新检查点
        torch.save(checkpoint, f'{self.config["save_dir"]}/latest_checkpoint.pth')

        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, f'{self.config["save_dir"]}/best_checkpoint.pth')
            print(f"保存最佳模型，验证损失: {self.best_val_loss:.6f}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.trainer.loss_fn.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.trainer.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            self.trainer.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"加载检查点，epoch: {self.current_epoch}, 最佳验证损失: {self.best_val_loss:.6f}")
        else:
            print("未找到检查点，从头开始训练")

    def train(self, resume_checkpoint=None):
        """主训练循环"""
        # 加载检查点（如果提供）
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

        print("开始训练...")

        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            print("-" * 50)

            # 训练
            train_losses = self.train_epoch()

            # 验证
            val_losses = self.validate()

            # 打印损失
            print(f"训练损失 - G: {train_losses['g_loss']:.4f}, D: {train_losses['d_loss']:.4f}")
            print(f"验证损失 - G: {val_losses['g_loss']:.4f}, D: {val_losses['d_loss']:.4f}")

            # 记录到tensorboard
            self.writer.add_scalar('Epoch/Train_Generator_Loss', train_losses['g_loss'], epoch)
            self.writer.add_scalar('Epoch/Train_Discriminator_Loss', train_losses['d_loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Generator_Loss', val_losses['g_loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Discriminator_Loss', val_losses['d_loss'], epoch)

            # 保存检查点
            is_best = val_losses['g_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['g_loss']

            self.save_checkpoint(is_best=is_best)

            # 学习率调度（可选）
            if self.config.get('use_scheduler', False):
                self.trainer.optimizer_g.step()
                self.trainer.optimizer_d.step()

        print("训练完成!")
        self.writer.close()


# ===== 配置和主函数 =====
def get_config():
    """训练配置"""
    return {
        # 数据路径
        'train_img_dir': 'path/to/your/val/images',
        'train_mask_dir': 'path/to/your/val/images',
        'val_img_dir': 'path/to/your/val/images',
        'val_mask_dir': 'path/to/your/val/masks',

        # 训练参数
        'batch_size': 4,
        'num_epochs': 200,
        'lr_g': 1e-4,  # 生成器学习率
        'lr_d': 2e-4,  # 判别器学习率
        'num_workers': 4,

        # 保存和日志
        'save_dir': './checkpoints',
        'log_dir': './logs',

        # 可选
        'use_scheduler': False,
    }


def main():
    """主函数"""
    # 获取配置
    config = get_config()

    # 更新为你的实际数据路径
    config.update({
        'train_img_dir': 'D:/myDataManager/pycharmProject/Crack-Segmentation/FGEM/crack/train/images',
        'train_mask_dir': 'D:/myDataManager/pycharmProject/Crack-Segmentation/FGEM/crack/train/masks',
        'val_img_dir': 'D:/myDataManager/pycharmProject/Crack-Segmentation/FGEM/crack/val/images',
        'val_mask_dir': 'D:/myDataManager/pycharmProject/Crack-Segmentation/FGEM/crack/val/masks',
    })

    # 初始化训练器
    trainer = CrackGANTrainer(config)

    # 开始训练（可选：从检查点恢复）
    resume_checkpoint = './checkpoints/latest_checkpoint.pth'
    trainer.train(resume_checkpoint=resume_checkpoint)


if __name__ == "__main__":
    main()