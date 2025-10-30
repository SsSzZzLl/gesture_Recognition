# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:14
# @Site : 
# @file : base_cnn.py
# @Software : PyCharm


import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(BaseCNN, self).__init__()
        # 卷积层：提取特征
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        # 全连接层：分类
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),  # 128*16*16 = 32768（128x128→16x16经过3次池化）
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 128 * 16 * 16)  # 展平
        x = self.fc_layers(x)
        return x