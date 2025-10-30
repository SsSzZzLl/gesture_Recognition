# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:14
# @Site : 
# @file : augmentors.py
# @Software : PyCharm


import cv2
import numpy as np
import random

class FrameAugmentor:
    """帧级增强：处理单帧图像的视觉变换"""
    @staticmethod
    def random_flip(frame):
        """随机水平翻转"""
        return cv2.flip(frame, 1) if random.random() > 0.5 else frame

    @staticmethod
    def random_rotate(frame, angle_range=(-15, 15)):
        """随机旋转（-15°~15°）"""
        h, w = frame.shape[:2]
        angle = random.uniform(*angle_range)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    @staticmethod
    def random_brightness(frame, factor_range=(0.7, 1.3)):
        """随机亮度调整"""
        factor = random.uniform(*factor_range)
        return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    @staticmethod
    def random_crop(frame, scale_range=(0.8, 1.0)):
        """随机裁剪后缩放回原尺寸"""
        h, w = frame.shape[:2]
        scale = random.uniform(*scale_range)
        crop_h, crop_w = int(h * scale), int(w * scale)
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        crop = frame[y:y+crop_h, x:x+crop_w]
        return cv2.resize(crop, (w, h))

    @staticmethod
    def random_blur(frame, kernel_size=(3, 3)):
        """随机高斯模糊"""
        return cv2.GaussianBlur(frame, kernel_size, 0) if random.random() > 0.5 else frame

    @classmethod
    def augment(cls, frame, config):
        """组合增强策略"""
        frame = cls.random_flip(frame)
        frame = cls.random_rotate(frame, config["rotate_range"])
        frame = cls.random_brightness(frame, config["brightness_range"])
        frame = cls.random_crop(frame, config["crop_scale"])
        frame = cls.random_blur(frame, config["blur_kernel"])
        return frame


class VideoAugmentor:
    """视频级增强：处理帧序列的时间变换"""
    @staticmethod
    def speed_change(frames, speed_factors=(0.8, 1.2)):
        """调整播放速度（0.8x~1.2x）"""
        factor = random.uniform(*speed_factors)
        new_len = int(len(frames) * factor)
        indices = np.linspace(0, len(frames)-1, new_len).astype(int)
        return [frames[i] for i in indices]

    @staticmethod
    def time_reverse(frames):
        """时间轴反转（动作倒放）"""
        return frames[::-1]

    @classmethod
    def augment(cls, frames, config):
        """组合视频增强"""
        if random.random() < config["reverse_prob"]:
            frames = cls.time_reverse(frames)
        frames = cls.speed_change(frames, config["speed_factors"])
        return frames