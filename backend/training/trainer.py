# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:15
# @Site : 
# @file : trainer.py
# @Software : PyCharm


#training/trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, class_weights, save_dir, model_name, epochs=50,
                 patience=8):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.class_weights = class_weights
        self.save_dir = save_dir
        self.model_name = model_name
        self.epochs = epochs
        self.patience = patience
        self.best_val_acc = 0.0
        self.counter = 0
        self.logs = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': []
        }
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics = self._calculate_metrics(all_labels, all_preds)
        return avg_loss, metrics

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = self._calculate_metrics(all_labels, all_preds)
        return avg_loss, metrics, all_preds, all_labels

    def _calculate_metrics(self, y_true, y_pred, average='macro'):
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs} | Model: {self.model_name}")
            train_loss, train_metrics = self._train_one_epoch()
            val_loss, val_metrics, val_preds, val_labels = self._validate()

            self.logs['train_loss'].append(train_loss)
            self.logs['val_loss'].append(val_loss)
            self.logs['train_acc'].append(train_metrics['accuracy'] * 100)
            self.logs['val_acc'].append(val_metrics['accuracy'] * 100)
            self.logs['train_precision'].append(train_metrics['precision'])
            self.logs['val_precision'].append(val_metrics['precision'])
            self.logs['train_recall'].append(train_metrics['recall'])
            self.logs['val_recall'].append(val_metrics['recall'])
            self.logs['train_f1'].append(train_metrics['f1'])
            self.logs['val_f1'].append(val_metrics['f1'])

            print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f}")

            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{self.model_name}_best.pth"))
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1} (最佳验证准确率: {self.best_val_acc:.4f})")
                    break
            self.scheduler.step()
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, f"{self.model_name}_best.pth")))
        return self.logs, val_preds, val_labels