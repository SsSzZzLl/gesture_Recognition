import torch
import os
from models.resnet_mini import ResNetMini  # 假设已实现ResNetMini模型

# 配置参数
MODEL_NAME = "resnet_mini"
CHECKPOINT_PATH = "E:/DoooooooooG/gesture_Recognition/backend/experiments/results/resnet_mini_optimized/迷你ResNet_best.pth"
OUTPUT_PATH = "E:/DoooooooooG/gesture_Recognition/frontend/models/resnet_mini_best.onnx"
NUM_CLASSES = 6


def check_paths():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"权重文件不存在：{CHECKPOINT_PATH}")
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def export_model():
    check_paths()
    # 加载模型
    model = ResNetMini(num_classes=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()  # 切换到评估模式

    # 生成虚拟输入（匹配预处理后的尺寸）
    dummy_input = torch.randn(1, 3, 128, 128)  # [batch, channel, height, width]

    # 导出ONNX模型
    torch.onnx.export(
        model, dummy_input, OUTPUT_PATH,
        opset_version=18,  # 兼容onnxruntime-web的版本
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # 支持动态batch尺寸
            "output": {0: "batch_size"}
        },
        export_params=True,
        external_data=True  # 处理大模型时启用
    )

    # 验证导出结果
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
        print("🎉 ONNX模型（含外部.data文件）导出成功！")
    else:
        raise RuntimeError("❌ 模型导出失败")


if __name__ == "__main__":
    try:
        export_model()
    except Exception as e:
        print(f"导出失败：{str(e)}")