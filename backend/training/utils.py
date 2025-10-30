# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site : 
# @file : utils.py
# @Software : PyCharm


import torch
import numpy as np

class EarlyStopping:
    """早停策略（防止过拟合）"""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class LearningRateScheduler:
    """学习率调度器（分段衰减）"""
    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch in self.milestones:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.gamma
            print(f"学习率衰减至：{self.optimizer.param_groups[0]['lr']}")