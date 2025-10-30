# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site : 
# @file : resnet_mini.py
# @Software : PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块（ bottleneck 简化版）"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut：当输入输出维度不匹配时，用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class ResNetMini(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetMini, self).__init__()
        self.in_channels = 32
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # 残差块（3组）
        self.layer1 = self._make_layer(ResidualBlock, 32, 2, stride=1)  # 不改变尺寸
        self.layer2 = self._make_layer(ResidualBlock, 64, 2, stride=2)  # 尺寸/2
        self.layer3 = self._make_layer(ResidualBlock, 128, 2, stride=2)  # 尺寸/2
        # 全连接层：修正展平维度（128 * 32 * 32，之前错误为128*16*16）
        self.fc = nn.Linear(128 * 32 * 32, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """创建残差层（包含多个残差块）"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 尺寸：128x128 → 128x128
        out = self.layer2(out)  # 尺寸：128x128 → 64x64
        out = self.layer3(out)  # 尺寸：64x64 → 32x32（修正核心！）
        out = out.view(-1, 128 * 32 * 32)  # 展平：128通道 × 32×32尺寸
        out = self.fc(out)
        return out