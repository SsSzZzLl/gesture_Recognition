# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site :
# @file : attention_cnn.py
# @Software : PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """通道注意力模块"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        x_cat = torch.cat([avg_out, max_out], dim=1)  # 拼接
        x_att = self.conv(x_cat)
        return x * self.sigmoid(x_att)


class AttentionCNN(nn.Module):
    """带通道+空间注意力的CNN"""

    def __init__(self, num_classes=6):
        super().__init__()
        # 第一个卷积块（带通道+空间注意力）
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),  # 通道注意力
            SpatialAttention(),  # 空间注意力
            nn.MaxPool2d(2, 2)
        )

        # 第二个卷积块
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128),
            SpatialAttention(),
            nn.MaxPool2d(2, 2)
        )

        # 第三个卷积块
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SEBlock(256),
            SpatialAttention(),
            nn.MaxPool2d(2, 2)
        )

        # 分类头（带Dropout）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 缓解过拟合
            nn.Linear(256 * 16 * 16, 512),  # 128/2^3=16
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)  # 展平特征
        x = self.classifier(x)
        return x