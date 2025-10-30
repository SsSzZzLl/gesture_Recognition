# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:22
# @Site : 
# @file : dataset.py
# @Software : PyCharm


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import json


class GestureDataset(Dataset):
    """手势数据集加载类"""

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir  # 指向data/processed
        self.split = split
        self.transform = transform
        self.samples = []

        # 加载类别映射
        with open(os.path.join(root_dir, "classes.json"), "r") as f:
            self.classes = json.load(f)
        self.num_classes = len(self.classes)

        # 加载split文件（train.txt/val.txt）
        split_file = os.path.join(root_dir, f"{split}.txt")
        with open(split_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                img_rel_path, label = line.split()
                img_path = os.path.join(root_dir, img_rel_path)
                if os.path.exists(img_path):
                    self.samples.append((img_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # 强制RGB格式
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def get_class_names(self):
        """返回类别名称列表"""
        return list(self.classes.keys())