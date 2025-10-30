# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site : 
# @file : trainer.py
# @Software : PyCharm


import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import numpy as np

# 配置路径
current_path = os.path.abspath(__file__)
training_dir = os.path.dirname(current_path)
backend_dir = os.path.dirname(training_dir)
sys.path.append(backend_dir)

from evaluate.metrics import compute_accuracy

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = torch.device(config["device"])
        self.model.to(self.device)

        # 创建实验目录
        self.exp_dir = os.path.join(config["exp_root"],
                                    f"{config['model_name']}_" + datetime.now().strftime("%Y%m%d_%H%M"))
        os.makedirs(self.exp_dir, exist_ok=True)

        # 初始化日志
        self.logs = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "config": config
        }

        # 保存配置
        with open(os.path.join(self.exp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def _get_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"]
        )

    def _get_scheduler(self, optimizer):
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config["lr_milestones"],
            gamma=0.1
        )

    def _get_loss_fn(self):
        if self.config["loss"] == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif self.config["loss"] == "focal":
            return FocalLoss()
        else:
            raise ValueError(f"不支持的损失函数：{self.config['loss']}")

    def train_epoch(self, dataloader, optimizer, loss_fn):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for imgs, labels in tqdm(dataloader, desc="训练"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            outputs = self.model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累计损失和预测结果
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)

            if preds.ndim != 1:
                print(f"警告：preds形状异常 {preds.shape}，强制展平")
                preds = preds.flatten()
            if labels.ndim != 1:
                print(f"警告：labels形状异常 {labels.shape}，强制展平")
                labels = labels.flatten()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        acc = compute_accuracy(all_preds, all_labels)

        # 新增：打印预测和标签的分布（仅在第一个epoch结束后打印）
        if len(self.logs["train_loss"]) == 0:  # 只在第1个epoch结束后打印
            print("\n【关键分布调试】")
            print(f"预测标签范围：{np.min(all_preds)} ~ {np.max(all_preds)}")
            print(f"预测标签计数：{np.bincount(all_preds)}")  # 统计每个预测值的数量
            print(f"真实标签范围：{np.min(all_labels)} ~ {np.max(all_labels)}")
            print(f"真实标签计数：{np.bincount(all_labels)}")  # 统计每个真实标签的数量
            print(f"预测与真实的交集：{np.intersect1d(all_preds, all_labels)}")  # 检查是否有重叠值

        return avg_loss, acc

    def val_epoch(self, dataloader, loss_fn):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="验证"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = loss_fn(outputs, labels)

                total_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        acc = compute_accuracy(all_preds, all_labels)
        return avg_loss, acc

    def run(self):
        # 初始化组件
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True
        )
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)
        loss_fn = self._get_loss_fn()

        best_val_acc = 0.0
        early_stop_counter = 0

        # 开始训练
        print(f"\n===== 开始训练 {self.config['model_name']}（设备：{self.device}）=====")
        for epoch in range(self.config["epochs"]):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, loss_fn)
            val_loss, val_acc = self.val_epoch(val_loader, loss_fn)

            # 记录日志
            self.logs["train_loss"].append(round(train_loss, 4))
            self.logs["train_acc"].append(round(train_acc * 100, 2))
            self.logs["val_loss"].append(round(val_loss, 4))
            self.logs["val_acc"].append(round(val_acc * 100, 2))

            # 打印结果
            print(f"训练：损失={train_loss:.4f}, 准确率={train_acc * 100:.2f}%")
            print(f"验证：损失={val_loss:.4f}, 准确率={val_acc * 100:.2f}%")

            # 学习率调度
            scheduler.step()

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.exp_dir, "best_model.pth"))
                print(f"保存最佳模型（验证准确率：{best_val_acc * 100:.2f}%）")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config["early_stop_patience"]:
                    print(f"早停触发（{self.config['early_stop_patience']}个epoch未提升）")
                    break

        # 保存日志
        log_path = os.path.join(self.exp_dir, "log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2)
        print(f"\n训练结束！日志保存至：{log_path}")

# 辅助：Focal Loss（解决类别不平衡）
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss