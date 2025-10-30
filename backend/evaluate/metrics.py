# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site : 
# @file : metrics.py
# @Software : PyCharm


# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site :
# @file : metrics.py
# @Software : PyCharm

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(preds, labels, num_classes):
    """计算多分类核心指标"""
    # 统一转为numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # 准确率
    accuracy = np.mean(preds == labels)

    # 精确率、召回率、F1（处理无预测的情况）
    precision = precision_score(
        labels, preds, average="macro", labels=range(num_classes), zero_division=0
    )
    recall = recall_score(
        labels, preds, average="macro", labels=range(num_classes), zero_division=0
    )
    f1 = f1_score(
        labels, preds, average="macro", labels=range(num_classes), zero_division=0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

def get_confusion_matrix(preds, labels, num_classes):
    """生成混淆矩阵"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    return cm.astype(int)


def compute_accuracy(preds, labels):
    """单独计算准确率（训练时快速调用）"""
    import numpy as np  # 确保导入numpy

    # 1. 统一数据类型并展平为一维数组（关键修复！）
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy().astype(int).ravel()  # 展平为一维
    else:
        preds = np.array(preds).astype(int).ravel()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().astype(int).ravel()  # 展平为一维
    else:
        labels = np.array(labels).astype(int).ravel()

    # 2. 打印调试信息（验证形状）
    print("\n【准确率计算调试】")
    print(f"预测数组形状：{preds.shape}（应为一维，如 (3784,)）")
    print(f"真实数组形状：{labels.shape}（应为一维，如 (3784,)）")
    print(f"前10个预测：{preds[:10]}")
    print(f"前10个真实：{labels[:10]}")
    print(f"前10个匹配结果：{preds[:10] == labels[:10]}")  # 应为布尔数组

    # 3. 计算匹配样本数和准确率
    match_count = sum(preds == labels)  # 现在可正常迭代
    total = len(preds)
    accuracy = match_count / total if total > 0 else 0.0
    print(f"匹配样本数：{match_count}/{total}")
    print(f"计算出的准确率：{accuracy:.4f}")

    return round(accuracy, 4)