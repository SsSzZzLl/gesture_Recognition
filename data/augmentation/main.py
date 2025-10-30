# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:14
# @Site : 
# @file : main.py
# @Software : PyCharm


"""批量处理8个member的视频，生成增强后的数据"""
import random
from video_processor import VideoProcessor

# 全局配置
CONFIG = {
    "PATH": {
        "raw_dir": "../raw",  # 原始视频目录
        "processed_dir": "../processed",  # 处理后数据目录
        "classes": {  # 6类动作标签映射
            "move": 0,
            "leftclick": 1,
            "rightclick": 2,
            "back": 3,
            "scrollup": 4,
            "scrolldown": 5
        }
    },
    "VIDEO": {
        "frame_interval": 1,  # 每秒提取1帧
        "target_size": (128, 128),  # 帧尺寸
        "val_split": 0.2,  # 验证集比例
        "members": [f"member{i}" for i in range(1, 9)]  # 8个member
    },
    "AUGMENT": {
        "num_frame_aug": 3,  # 每帧生成3个增强样本
        "frame_aug": {
            "rotate_range": (-15, 15),
            "brightness_range": (0.7, 1.3),
            "crop_scale": (0.8, 1.0),
            "blur_kernel": (3, 3)
        },
        "video_aug": {
            "speed_factors": (0.8, 1.2),
            "reverse_prob": 0.5
        }
    }
}

if __name__ == "__main__":
    random.seed(42)  # 固定随机种子
    processor = VideoProcessor(CONFIG)
    processor.run()
    print("数据增强完成！结果保存至 data/processed")