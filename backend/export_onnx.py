# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:10
# @Site : 
# @file : export_onnx.py
# @Software : PyCharm


import torch
import os
import argparse
from models.base_cnn import BaseCNN
from models.attention_cnn import AttentionCNN
from models.resnet_mini import ResNetMini


def export_model(model_name, checkpoint_path, output_path, num_classes=6):
    """将PyTorch模型导出为ONNX格式"""
    # 加载模型结构
    if model_name == "base_cnn":
        model = BaseCNN(num_classes=num_classes)
    elif model_name == "attention_cnn":
        model = AttentionCNN(num_classes=num_classes)
    elif model_name == "resnet_mini":
        model = ResNetMini(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型：{model_name}")

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # 切换到推理模式

    # 导出ONNX
    dummy_input = torch.randn(1, 3, 128, 128)  # 输入示例（batch=1）
    torch.onnx.export(
        model, dummy_input, output_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # 支持动态batch
    )
    print(f"ONNX模型导出成功：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["base_cnn", "attention_cnn", "resnet_mini"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重路径（如experiments/results/.../best_model.pth）")
    parser.add_argument("--output", type=str, required=True,
                        help="ONNX输出路径（如../frontend/models/base_cnn.onnx）")
    args = parser.parse_args()

    export_model(args.model, args.checkpoint, args.output)