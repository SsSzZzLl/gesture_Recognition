# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:57
# @Site : 
# @file : run_exp3.py
# @Software : PyCharm


import os
import sys
import torch
import json
from torch.utils.data import DataLoader

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # backend/experiments
backend_dir = os.path.dirname(current_dir)  # backend
root_dir = os.path.dirname(backend_dir)  # 根目录：gesture_Recognition
sys.path.append(root_dir)

# 导入模块（带 backend. 前缀）
from backend.data.dataset import GestureDataset, train_transform, val_transform
from backend.models.resnet_mini import ResNetMini
from backend.training.trainer import Trainer
from backend.evaluate.visualize import plot_confusion_matrix


def main():
    # 数据路径：根目录/data/processed
    data_root = os.path.join(root_dir, "data", "processed")
    save_root = os.path.join(backend_dir, "experiments", "results", "resnet_mini_optimized")
    os.makedirs(save_root, exist_ok=True)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据集
    train_dataset = GestureDataset(
        data_root,
        split="train",
        transform=train_transform
    )
    val_dataset = GestureDataset(
        data_root,
        split="val",
        transform=val_transform
    )
    class_weights = train_dataset.get_class_weights()
    class_names = train_dataset.get_class_names()

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # 初始化模型
    model = ResNetMini(num_classes=len(class_names)).to(device)

    # 训练配置
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights,
        save_dir=save_root,
        model_name="迷你ResNet",
        epochs=50,
        patience=8
    )

    # 启动训练
    logs, val_preds, val_labels = trainer.run()

    # 保存训练日志
    with open(os.path.join(save_root, "训练日志.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    # 绘制混淆矩阵
    plot_confusion_matrix(
        y_true=val_labels,
        y_pred=val_preds,
        class_names=class_names,
        save_path=os.path.join(save_root, "迷你ResNet_混淆矩阵.png")
    )

    print(f"迷你ResNet实验完成，结果保存至: {save_root}")


if __name__ == "__main__":
    main()