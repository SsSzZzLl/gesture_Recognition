# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:22
# @Site : 
# @file : dataset.py
# @Software : PyCharm



# 在 data/dataset.py 中修改
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import json

# 训练集增强变换
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集/测试集变换
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GestureDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.img_paths, self.labels = self._load_data()
        # 计算类别权重
        self.class_weights = self._compute_class_weights()

    def _load_data(self):
        txt_path = os.path.join(self.data_root, f"{self.split}.txt")
        with open(txt_path, "r") as f:
            lines = f.readlines()
        img_paths = []
        labels = []
        for line in lines:
            img_path, label = line.strip().split()
            img_paths.append(os.path.join(self.data_root, img_path))
            labels.append(int(label))
        return img_paths, labels

    def _compute_class_weights(self):
        class_counts = np.bincount(self.labels)
        return len(self.labels) / (len(class_counts) * class_counts)  # 平衡权重

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_weights(self):
        return torch.tensor(self.class_weights, dtype=torch.float32)

    def get_class_names(self):
        with open(os.path.join(self.data_root, "classes.json"), "r") as f:
            return json.load(f)