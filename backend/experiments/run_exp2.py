# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site : 
# @file : run_exp2.py
# @Software : PyCharm


import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import json

# 配置路径
current_path = os.path.abspath(__file__)
experiments_dir = os.path.dirname(current_path)
backend_dir = os.path.dirname(experiments_dir)
sys.path.append(backend_dir)

from models.attention_cnn import AttentionCNN
from training.trainer import Trainer

# 自定义数据集（同run_exp1）
class GestureDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.img_paths, self.labels = self._load_data()
        with open(os.path.join(data_root, "classes.json"), "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

    def _load_data(self):
        txt_path = os.path.join(self.data_root, f"{self.split}.txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        img_paths = []
        labels = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            img_path, label = line.split()
            img_paths.append(os.path.join(self.data_root, img_path))
            labels.append(int(label))
        return img_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_names(self):
        return self.class_names

def main():
    # 配置路径
    data_root = os.path.join(os.path.dirname(backend_dir), "data", "processed")
    exp_root = os.path.join(experiments_dir, "results")
    os.makedirs(exp_root, exist_ok=True)

    # 设备配置
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理（强化增强）
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = GestureDataset(data_root, split="train", transform=transform)
    val_dataset = GestureDataset(data_root, split="val", transform=transform)
    print(f"✅ 数据集加载成功：训练集{len(train_dataset)}样本，验证集{len(val_dataset)}样本")
    print(f"✅ 类别列表：{train_dataset.get_class_names()}")

    # 初始化模型
    model = AttentionCNN(num_classes=6)
    print(f"✅ 模型初始化成功：注意力CNN（{sum(p.numel() for p in model.parameters()):,} 参数量）")

    # 训练配置（适配复杂模型）
    config = {
        "model_name": "attention_cnn",
        "device": str(device_obj),
        "batch_size": 16,
        "epochs": 60,
        "lr": 0.0003,
        "weight_decay": 1e-5,
        "loss": "focal",  # Focal Loss解决类别不平衡
        "lr_milestones": [25, 45],
        "early_stop_patience": 10,
        "data_root": data_root,
        "exp_root": exp_root,
        "num_workers": 0
    }

    # 开始训练
    trainer = Trainer(model, train_dataset, val_dataset, config)
    print(f"✅ 模型和训练器初始化成功，开始训练（设备：{device_obj}）")
    trainer.run()

if __name__ == "__main__":
    main()