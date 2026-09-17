import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torchvision.transforms as transforms

class DroneVehicleMaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_transform=None, mask_transform=None):
        """
        img_dir: 原图路径
        mask_dir: 灰度标注 mask 路径
        img_transform: 仅用于图像的变换
        mask_transform: 仅用于mask的变换
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 灰度 mask

        # 分别应用不同的变换
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # 如果没有提供mask_transform，默认转换为Tensor
            mask = transforms.ToTensor()(mask)

        # 确保mask是单通道
        if mask.shape[0] != 1:
            mask = mask.mean(dim=0, keepdim=True)

        return img, mask