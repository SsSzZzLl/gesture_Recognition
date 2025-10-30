# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午7:55
# @Site : 
# @file : compare_models.py
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
evaluate_dir = os.path.dirname(current_path)
backend_dir = os.path.dirname(evaluate_dir)
sys.path.append(backend_dir)

from visualize import (
    load_logs, plot_loss_curve, plot_acc_curve,
    plot_confusion_matrices, plot_metrics_summary
)
from models.base_cnn import BaseCNN
from models.attention_cnn import AttentionCNN
from models.resnet_mini import ResNetMini
from evaluate.metrics import confusion_matrix

# 自定义数据集（用于加载验证集）
class GestureDataset(Dataset):
    def __init__(self, data_root, split="val", transform=None):
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
        # 加载图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 应用transform
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_names(self):
        return self.class_names

def main():
    # ---------------------- 配置参数（需根据你的实际路径修改！）----------------------
    # 1. 三个模型的实验目录（训练后生成的目录，替换为你的实际路径）
    exp_dirs = {
        "基础CNN": "E:/DoooooooooG/gesture_Recognition/backend/experiments/results/base_cnn_20251030_2106",
        "注意力CNN": "E:/DoooooooooG/gesture_Recognition/backend/experiments/results/attention_cnn_20251030_2113",
        "迷你ResNet": "E:/DoooooooooG/gesture_Recognition/backend/experiments/results/resnet_mini_20251030_2122"
    }
    # 2. 数据根目录（data/processed/）
    data_root = "E:/DoooooooooG/gesture_Recognition/data/processed"
    # 3. 图表保存目录
    save_dir = os.path.join(evaluate_dir, "model_comparison")
    os.makedirs(save_dir, exist_ok=True)
    # 4. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -----------------------------------------------------------------------------

    # 1. 加载验证集
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = GestureDataset(data_root, split="val", transform=transform)
    print(f"验证集加载完成：{len(val_dataset)} 样本")

    # 2. 加载模型日志
    logs = load_logs(exp_dirs)
    print("模型日志加载完成")

    # 3. 生成对比曲线
    plot_loss_curve(logs, os.path.join(save_dir, "损失曲线对比.png"))
    plot_acc_curve(logs, os.path.join(save_dir, "准确率曲线对比.png"))

    # 4. 生成混淆矩阵
    model_dict = {
        "基础CNN": (BaseCNN(num_classes=6), exp_dirs["基础CNN"]),
        "注意力CNN": (AttentionCNN(num_classes=6), exp_dirs["注意力CNN"]),
        "迷你ResNet": (ResNetMini(num_classes=6), exp_dirs["迷你ResNet"])
    }
    plot_confusion_matrices(model_dict, val_dataset, device, save_dir)

    # 5. 生成指标汇总图
    plot_metrics_summary(logs, os.path.join(save_dir, "指标汇总对比.png"))

    print(f"\n所有图表已保存至：{save_dir}")

if __name__ == "__main__":
    main()