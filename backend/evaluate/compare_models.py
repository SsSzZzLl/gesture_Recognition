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

# 配置路径（确保能导入backend下的模块）
current_path = os.path.abspath(__file__)
evaluate_dir = os.path.dirname(current_path)
backend_dir = os.path.dirname(evaluate_dir)
sys.path.append(backend_dir)

# 导入所有需要的函数（现在visualize.py已全部包含）
from evaluate.visualize import (
    load_logs, plot_loss_curve, plot_acc_curve,
    plot_confusion_matrices, plot_metrics_summary
)
from models.base_cnn import BaseCNN
from models.attention_cnn import AttentionCNN
from models.resnet_mini import ResNetMini

# 自定义数据集（用于加载验证集，保持与训练一致）
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
        # 加载图像（cv2默认BGR，转为RGB）
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 应用transform（与训练时一致）
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_names(self):
        return self.class_names

def main():
    # ---------------------- 配置参数（根据实际情况修改！）----------------------
    # 1. 三个模型的实验目录（训练后生成的结果目录）
    exp_dirs = {
        "基础CNN": "E:/DoooooooooG/gesture_Recognition/backend/experiments/results/base_cnn_optimized_v3",
        "注意力CNN": "E:/DoooooooooG/gesture_Recognition/backend/experiments/results/attention_cnn_optimized",
        "迷你ResNet": "E:/DoooooooooG/gesture_Recognition/backend/experiments/results/resnet_mini_optimized"
    }
    # 2. 数据根目录（根目录下的data/processed）
    data_root = "E:/DoooooooooG/gesture_Recognition/data/processed"
    # 3. 图表保存目录
    save_dir = os.path.join(evaluate_dir, "model_comparison")
    os.makedirs(save_dir, exist_ok=True)
    # 4. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    # -----------------------------------------------------------------------------

    # 1. 加载验证集（transform与训练时保持一致）
    transform = transforms.Compose([
        transforms.ToPILImage(),  # cv2加载的是numpy数组，转为PIL便于transform处理
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = GestureDataset(data_root, split="val", transform=transform)
    print(f"验证集加载完成：{len(val_dataset)} 个样本")

    # 2. 加载模型日志（现在参数匹配，不会报错）
    logs = load_logs(exp_dirs)
    print("所有模型日志加载完成")

    # 3. 生成训练/验证曲线对比图
    plot_loss_curve(logs, os.path.join(save_dir, "损失曲线对比.png"))
    plot_acc_curve(logs, os.path.join(save_dir, "准确率曲线对比.png"))
    print("损失/准确率曲线生成完成")

    # 4. 生成每个模型的混淆矩阵
    model_dict = {
        "基础CNN": (BaseCNN(num_classes=len(val_dataset.get_class_names())), exp_dirs["基础CNN"]),
        "注意力CNN": (AttentionCNN(num_classes=len(val_dataset.get_class_names())), exp_dirs["注意力CNN"]),
        "迷你ResNet": (ResNetMini(num_classes=len(val_dataset.get_class_names())), exp_dirs["迷你ResNet"])
    }
    plot_confusion_matrices(model_dict, val_dataset, device, save_dir)
    print("混淆矩阵生成完成")

    # 5. 生成指标汇总对比图
    plot_metrics_summary(logs, os.path.join(save_dir, "指标汇总对比.png"))
    print("指标汇总图生成完成")

    print(f"\n所有对比图表已保存至：{save_dir}")

if __name__ == "__main__":
    main()