# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site : 
# @file : losses.py
# @Software : PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss（解决类别不平衡）"""
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()

class LabelSmoothingLoss(nn.Module):
    """标签平滑（防止过拟合）"""
    def __init__(self, eps=0.1, reduction="batchmean"):  # 1. 默认值改为batchmean
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        smoothed = one_hot * (1 - self.eps) + torch.ones_like(inputs) * self.eps / n_classes
        # 2. 确保kl_div使用当前类的reduction参数（已默认是batchmean）
        return F.kl_div(F.log_softmax(inputs, dim=1), smoothed, reduction=self.reduction)