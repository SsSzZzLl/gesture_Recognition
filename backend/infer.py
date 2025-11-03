# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/31 上午11:55
# @Site : 
# @file : infer.py
# @Software : PyCharm


import os
import sys
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import json

current_path = os.path.abspath(__file__)
backend_dir = os.path.dirname(current_path)
sys.path.append(backend_dir)

from models.base_cnn import BaseCNN
from models.attention_cnn import AttentionCNN
from models.resnet_mini import ResNetMini


def load_classes(classes_path):
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    return {v: k for k, v in classes.items()}


def preprocess_image(image, target_size=(128, 128)):
    """预处理：与训练/验证集保持一致"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # 增加batch维度


def infer_camera(model, class_map, device):
    """摄像头实时推理"""
    cap = cv2.VideoCapture(0)  # 0=默认摄像头，多个摄像头可换1、2等
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 调整画面宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 调整画面高度

    print("📹 摄像头实时验证已启动，按 'q' 键退出...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break

        # 画面水平翻转（镜像显示，更符合操作习惯）
        frame = cv2.flip(frame, 1)
        # 预处理：BGR→RGB+Resize+归一化
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_image(frame_rgb).to(device)

        # 模型预测
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_label = torch.argmax(outputs, dim=1).item()
            pred_action = class_map[pred_label]
            confidence = torch.softmax(outputs, dim=1)[0][pred_label].item() * 100

        # 在画面上绘制结果（绿色文字=高置信度，红色=低置信度）
        color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
        cv2.putText(
            frame, f"Gesture: {pred_action}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
        )
        cv2.putText(
            frame, f"Confidence: {confidence:.1f}%",
            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
        )

        # 显示画面
        cv2.imshow("Gesture Recognition (Camera)", frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("👋 实时验证已结束")


def infer_image(model, image_path, class_map, device):
    """单图片推理（保留原有功能）"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片：{image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess_image(image_rgb).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_label = torch.argmax(outputs, dim=1).item()
        pred_action = class_map[pred_label]
        confidence = torch.softmax(outputs, dim=1)[0][pred_label].item() * 100

    cv2.putText(
        image, f"Gesture: {pred_action} ({confidence:.1f}%)",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.imshow("Gesture Recognition (Image)", image)
    cv2.waitKey(0)
    cv2.imwrite("infer_image_result.jpg", image)
    print(f"💾 图片推理结果已保存至：infer_image_result.jpg")
    cv2.destroyAllWindows()


def infer_video(model, video_path, class_map, device, save_path="infer_video_result.mp4"):
    """视频推理（保留原有功能）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法读取视频：{video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print(f"🎬 视频推理中，按 'q' 键提前退出...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_image(frame_rgb).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_label = torch.argmax(outputs, dim=1).item()
            pred_action = class_map[pred_label]
            confidence = torch.softmax(outputs, dim=1)[0][pred_label].item() * 100

        color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
        cv2.putText(
            frame, f"Gesture: {pred_action} ({confidence:.1f}%)",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
        )

        cv2.imshow("Gesture Recognition (Video)", frame)
        writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"💾 视频推理结果已保存至：{save_path}")


def main():
    # ---------------------- 配置参数（根据实际情况修改！）----------------------
    model_type = "resnet_mini"  # 可选：base_cnn/attention_cnn/resnet_mini
    model_path = "E:/DoooooooooG/gesture_Recognition/backend/experiments/results/resnet_mini_optimized/迷你ResNet_best.pth"
    classes_path = "E:/DoooooooooG/gesture_Recognition/data/processed/classes.json"
    input_type = "camera"  # 可选：camera（摄像头）/ image（图片）/ video（视频）
    input_path = "test_image.jpg"  # input_type为image/video时填写路径
    # -----------------------------------------------------------------------------

    # 设备配置（自动选择GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备：{device}")

    # 加载类别映射（手势标签→名称）
    try:
        class_map = load_classes(classes_path)
        num_classes = len(class_map)
    except Exception as e:
        print(f"❌ 加载类别文件失败：{e}")
        return

    # 加载模型
    try:
        if model_type == "base_cnn":
            model = BaseCNN(num_classes=num_classes).to(device)
        elif model_type == "attention_cnn":
            model = AttentionCNN(num_classes=num_classes).to(device)
        elif model_type == "resnet_mini":
            model = ResNetMini(num_classes=num_classes).to(device)
        else:
            raise ValueError("模型类型错误！可选：base_cnn/attention_cnn/resnet_mini")

        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 成功加载模型：{model_type}")
    except Exception as e:
        print(f"❌ 加载模型失败：{e}")
        return

    # 选择推理模式
    if input_type == "camera":
        infer_camera(model, class_map, device)
    elif input_type == "image":
        infer_image(model, input_path, class_map, device)
    elif input_type == "video":
        infer_video(model, input_path, class_map, device)
    else:
        print("❌ 输入类型错误！可选：camera/image/video")


if __name__ == "__main__":
    main()