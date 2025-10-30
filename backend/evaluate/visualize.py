# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site :
# @file : visualize.py
# @Software : PyCharm


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# 配置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """单个模型的混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_metrics_summary(logs, save_path):
    """指标汇总柱状图"""
    model_names = list(logs.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    final_metrics = {metric: [] for metric in metrics}

    for model_name, log in logs.items():
        final_metrics["accuracy"].append(log["val_acc"][-1] / 100)
        final_metrics["precision"].append(log["val_precision"][-1])
        final_metrics["recall"].append(log["val_recall"][-1])
        final_metrics["f1"].append(log["val_f1"][-1])

    x = np.arange(len(model_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, final_metrics[metric], width, label=metric)

    ax.set_xlabel("模型", fontsize=12)
    ax.set_ylabel("指标值（0-1）", fontsize=12)
    ax.set_title("模型验证集指标汇总", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_logs(exp_dirs):
    """
    适配原始调用：接收 {模型名: 目录} 字典，加载所有模型日志
    :param exp_dirs: 字典格式 {模型名称: 结果目录路径}
    :return: 整合后的日志字典 {模型名: 日志数据}
    """
    all_logs = {}
    for model_name, dir_path in exp_dirs.items():
        log_path = os.path.join(dir_path, "训练日志.json")
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"模型 {model_name} 的日志文件不存在：{log_path}")
        with open(log_path, "r", encoding="utf-8") as f:
            all_logs[model_name] = json.load(f)
    return all_logs


def plot_loss_curve(logs, save_path):
    """绘制所有模型的训练/验证损失曲线对比"""
    plt.figure(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 区分3个模型的颜色
    markers = ["o", "s", "^"]  # 区分3个模型的标记

    for idx, (model_name, log) in enumerate(logs.items()):
        epochs = len(log["train_loss"])
        plt.plot(
            range(1, epochs + 1), log["train_loss"],
            color=colors[idx], marker=markers[idx], markersize=4,
            label=f"{model_name} (训练)"
        )
        plt.plot(
            range(1, epochs + 1), log["val_loss"],
            color=colors[idx], marker=markers[idx], markersize=4,
            linestyle="--", label=f"{model_name} (验证)"
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("损失值", fontsize=12)
    plt.title("模型训练/验证损失曲线对比", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_acc_curve(logs, save_path):
    """绘制所有模型的训练/验证准确率曲线对比"""
    plt.figure(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    for idx, (model_name, log) in enumerate(logs.items()):
        epochs = len(log["train_acc"])
        plt.plot(
            range(1, epochs + 1), [acc / 100 for acc in log["train_acc"]],  # 百分比转小数
            color=colors[idx], marker=markers[idx], markersize=4,
            label=f"{model_name} (训练)"
        )
        plt.plot(
            range(1, epochs + 1), [acc / 100 for acc in log["val_acc"]],
            color=colors[idx], marker=markers[idx], markersize=4,
            linestyle="--", label=f"{model_name} (验证)"
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("准确率（0-1）", fontsize=12)
    plt.title("模型训练/验证准确率曲线对比", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrices(model_dict, val_dataset, device, save_dir):
    """
    加载所有模型的最佳权重，生成混淆矩阵
    :param model_dict: {模型名: (模型类, 结果目录)}
    :param val_dataset: 验证集Dataset
    :param device: 运行设备（cuda/cpu）
    :param save_dir: 混淆矩阵保存目录
    """
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    class_names = val_dataset.get_class_names()

    for model_name, (model, exp_dir) in model_dict.items():
        # 加载模型最佳权重
        weight_path = os.path.join(exp_dir, f"{model_name}_best.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"模型 {model_name} 的最佳权重不存在：{weight_path}")

        # 初始化模型并加载权重
        model = model.to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        # 验证集预测
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # 生成混淆矩阵
        save_path = os.path.join(save_dir, f"{model_name}_混淆矩阵.png")
        plot_confusion_matrix(all_labels, all_preds, class_names, save_path)
        print(f"{model_name} 混淆矩阵已保存至：{save_path}")