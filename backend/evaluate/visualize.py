# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site : 
# @file : visualize.py
# @Software : PyCharm


import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 配置路径
current_path = os.path.abspath(__file__)
evaluate_dir = os.path.dirname(current_path)
backend_dir = os.path.dirname(evaluate_dir)
sys.path.append(backend_dir)

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

def load_logs(exp_dirs):
    """加载多个模型的日志"""
    logs = {}
    for model_name, exp_dir in exp_dirs.items():
        log_path = os.path.join(exp_dir, "log.json")
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"日志文件不存在：{log_path}")
        with open(log_path, "r", encoding="utf-8") as f:
            logs[model_name] = json.load(f)
    return logs

def plot_loss_curve(logs, save_path):
    """绘制损失对比曲线"""
    plt.figure(figsize=(10, 6))
    for model_name, log in logs.items():
        plt.plot(log["train_loss"], label=f"{model_name} - 训练损失", linewidth=2)
        plt.plot(log["val_loss"], label=f"{model_name} - 验证损失", linewidth=2, linestyle="--")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("损失值", fontsize=12)
    plt.title("三个模型训练/验证损失对比", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"损失曲线已保存至：{save_path}")

def plot_acc_curve(logs, save_path):
    """绘制准确率对比曲线"""
    plt.figure(figsize=(10, 6))
    for model_name, log in logs.items():
        plt.plot(log["train_acc"], label=f"{model_name} - 训练准确率", linewidth=2)
        plt.plot(log["val_acc"], label=f"{model_name} - 验证准确率", linewidth=2, linestyle="--")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("准确率（%）", fontsize=12)
    plt.title("三个模型训练/验证准确率对比", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"准确率曲线已保存至：{save_path}")

def plot_confusion_matrices(model_dict, val_dataset, device, save_dir):
    """绘制每个模型的混淆矩阵"""
    class_names = val_dataset.get_class_names()
    num_classes = len(class_names)

    for model_name, (model, exp_dir) in model_dict.items():
        # 加载最佳模型权重
        model_path = os.path.join(exp_dir, "best_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # 生成预测结果
        from torch.utils.data import DataLoader
        dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # 绘制混淆矩阵
        cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
        plt.figure(figsize=(10, 8))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca()
        )
        plt.title(f"{model_name} - 验证集混淆矩阵", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{model_name}_混淆矩阵.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"{model_name}混淆矩阵已保存至：{save_path}")

def plot_metrics_summary(logs, save_path):
    """绘制指标汇总柱状图（最终epoch），包含accuracy、precision、recall、f1"""
    # --------------------------
    # 1. 配置中文字体（解决乱码和警告）
    # --------------------------
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # --------------------------
    # 2. 提取模型名称和初始化指标容器
    # --------------------------
    model_names = list(logs.keys())  # 模型名称列表（如：["迷你ResNet", "模型A", "模型B"]）
    metrics = ["accuracy", "precision", "recall", "f1"]  # 需要对比的指标
    final_metrics = {metric: [] for metric in metrics}  # 存储每个指标的最终值（按模型顺序）

    # --------------------------
    # 3. 提取每个模型的最终指标（关键修复：补充所有指标的数据）
    # --------------------------
    for model_name, log in logs.items():
        # 3.1 提取准确率（已有的逻辑，保持不变）
        # 假设log["val_acc"]存储的是百分比（如：95.5 -> 转换为0.955）
        final_acc = log["val_acc"][-1] / 100  # 取最后一个epoch的验证集准确率
        final_metrics["accuracy"].append(final_acc)

        # 3.2 提取精确率（根据实际log结构调整键名，如val_precision）
        # 若log中已记录，直接取最后一个epoch的值
        if "val_precision" in log:
            final_precision = log["val_precision"][-1]  # 假设已是0-1范围，无需除以100
        else:
            # 若log中没有，调用calculate_metrics重新计算（需传入模型和验证数据）
            # 这里需要你根据实际情况补充：加载模型、获取验证集数据和标签
            # model = ...  # 加载当前模型
            # val_data, val_labels = ...  # 获取验证集数据和标签
            # preds = model.predict(val_data)  # 模型预测
            # metrics_dict = calculate_metrics(val_labels, preds)  # 计算指标
            # final_precision = metrics_dict["precision"]
            final_precision = 0.0  # 临时占位，需替换为实际计算逻辑
        final_metrics["precision"].append(final_precision)

        # 3.3 提取召回率（同上，根据实际log结构调整）
        if "val_recall" in log:
            final_recall = log["val_recall"][-1]
        else:
            # 同上，需补充重新计算的逻辑
            final_recall = 0.0  # 临时占位
        final_metrics["recall"].append(final_recall)

        # 3.4 提取F1分数（同上）
        if "val_f1" in log:
            final_f1 = log["val_f1"][-1]
        else:
            # 同上，需补充重新计算的逻辑
            final_f1 = 0.0  # 临时占位
        final_metrics["f1"].append(final_f1)

    # --------------------------
    # 4. 检查数据长度是否匹配（调试用，可保留）
    # --------------------------
    model_count = len(model_names)
    for metric, values in final_metrics.items():
        if len(values) != model_count:
            raise ValueError(f"指标 {metric} 的数据长度（{len(values)}）与模型数量（{model_count}）不匹配！")

    # --------------------------
    # 5. 绘制柱状图
    # --------------------------
    x = np.arange(model_count)  # x轴坐标（模型数量）
    width = 0.2  # 每个柱子的宽度
    fig, ax = plt.subplots(figsize=(12, 7))  # 创建画布

    # 循环绘制每个指标的柱状图
    for i, metric in enumerate(metrics):
        # x + i*width：每个指标的柱子在x轴上错开排列
        ax.bar(x + i * width, final_metrics[metric], width, label=metric)

    # 设置图表标签和标题
    ax.set_xlabel("模型", fontsize=12)
    ax.set_ylabel("指标值（0-1）", fontsize=12)
    ax.set_title("三个模型最终验证集指标汇总", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)  # 调整x轴刻度位置（居中显示）
    ax.set_xticklabels(model_names, fontsize=10)  # 模型名称作为x轴标签
    ax.legend(fontsize=10)  # 显示图例（指标名称）
    ax.grid(alpha=0.3, axis="y")  # 添加y轴网格线

    # 保存图片
    plt.tight_layout()  # 自动调整布局，避免标签被截断
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # 关闭画布，释放资源
    print(f"指标汇总图已保存至：{save_path}")