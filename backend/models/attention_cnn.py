# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site : 
# @file : attention_cnn.py
# @Software : PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(AttentionCNN, self).__init__()
        # 卷积层+注意力
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.ca1 = ChannelAttention(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.ca2 = ChannelAttention(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.ca3 = ChannelAttention(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        # 全连接层
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 第一层：卷积+BN+ReLU+注意力+池化
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.ca1(x)
        x = self.pool(x)
        x = self.dropout(x)

        # 第二层
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.ca2(x)
        x = self.pool(x)
        x = self.dropout(x)

        # 第三层
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.ca3(x)
        x = self.pool(x)
        x = self.dropout(x)

        # 分类
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x