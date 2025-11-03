# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/31 下午12:01
# @Site : 
# @file : dataset_no_aug.py
# @Software : PyCharm


import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import json

# 无增强：仅保留基础预处理（与验证集一致，避免数据分布差异）
train_transform_no_aug = transforms.Compose([
    transforms.Resize((128, 128)),  # 直接缩放到目标尺寸（无随机裁剪）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集变换保持不变（确保评估一致性）
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GestureDatasetNoAug(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        label_file = os.path.join(root_dir, f"{split}.txt")
        self.samples = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((os.path.join(root_dir, path), int(label)))
        self.labels = [s[1] for s in self.samples]
        with open(os.path.join(root_dir, "classes.json"), "r", encoding="utf-8") as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"加载图片失败: {img_path}, 错误: {e}")
            image = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self):
        class_counts = np.bincount(self.labels)
        class_counts = np.maximum(class_counts, 1)
        return torch.tensor(len(self.labels) / (len(class_counts) * class_counts), dtype=torch.float32)

    def get_class_names(self):
        return self.class_names