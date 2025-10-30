# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:14
# @Site : 
# @file : video_processor.py
# @Software : PyCharm


import cv2
import os
import json
import random
from tqdm import tqdm
from augmentors import FrameAugmentor, VideoAugmentor


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.raw_dir = config["PATH"]["raw_dir"]
        self.processed_dir = config["PATH"]["processed_dir"]
        self.frames_dir = os.path.join(self.processed_dir, "frames")
        self.classes = config["PATH"]["classes"]
        self.members = config["VIDEO"]["members"]

        # 创建输出目录
        os.makedirs(self.frames_dir, exist_ok=True)
        for action in self.classes:
            os.makedirs(os.path.join(self.frames_dir, action), exist_ok=True)

    def extract_frames(self, video_path):
        """从视频中按时间间隔提取帧"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps * self.config["VIDEO"]["frame_interval"])  # 按秒间隔抽帧
        frames = []

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                # 转RGB+缩放
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, self.config["VIDEO"]["target_size"])
                frames.append(frame_resized)
            count += 1
        cap.release()
        return frames

    def process_member(self, member):
        """处理单个member的所有视频"""
        member_dir = os.path.join(self.raw_dir, member)
        if not os.path.exists(member_dir):
            print(f"警告：{member_dir} 不存在，跳过")
            return []

        samples = []  # 存储（图像路径，标签）
        for action in self.classes:
            # 匹配动作视频（如move_1.mp4）
            video_files = [f for f in os.listdir(member_dir) if f.startswith(f"{action}_") and f.endswith(".mp4")]
            if not video_files:
                print(f"警告：{member} 缺少 {action} 视频")
                continue

            for video_file in video_files:
                video_path = os.path.join(member_dir, video_file)
                video_id = video_file.split(".")[0]  # 提取文件名（不含后缀）

                # 1. 提取原始帧
                raw_frames = self.extract_frames(video_path)
                if not raw_frames:
                    print(f"警告：{video_path} 提取帧失败")
                    continue

                # 2. 视频级增强（生成增强帧序列）
                aug_videos = [raw_frames]  # 原始视频
                aug_videos.append(VideoAugmentor.augment(raw_frames, self.config["AUGMENT"]["video_aug"]))  # 增强视频

                # 3. 处理每段视频的帧（原始+帧级增强）
                for vidx, frames in enumerate(aug_videos):
                    for fidx, frame in enumerate(frames):
                        # 保存原始帧
                        raw_name = f"{member}_{video_id}_v{vidx}_f{fidx}.jpg"
                        raw_path = os.path.join(self.frames_dir, action, raw_name)
                        cv2.imwrite(raw_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        samples.append((raw_path, self.classes[action]))

                        # 生成帧级增强样本
                        for aidx in range(self.config["AUGMENT"]["num_frame_aug"]):
                            aug_frame = FrameAugmentor.augment(frame, self.config["AUGMENT"]["frame_aug"])
                            aug_name = f"{member}_{video_id}_v{vidx}_f{fidx}_aug{aidx}.jpg"
                            aug_path = os.path.join(self.frames_dir, action, aug_name)
                            cv2.imwrite(aug_path, cv2.cvtColor(aug_frame, cv2.COLOR_RGB2BGR))
                            samples.append((aug_path, self.classes[action]))

        return samples

    def split_train_val(self, all_samples):
        """按member划分训练/验证集（避免数据泄露）"""
        # 按member分组
        member_samples = {m: [] for m in self.members}
        for path, label in all_samples:
            for m in self.members:
                if m in path:
                    member_samples[m].append((path, label))
                    break

        # 随机选择验证集member（20%）
        val_size = int(len(self.members) * self.config["VIDEO"]["val_split"])
        val_members = set(random.sample(self.members, val_size))

        train, val = [], []
        for m, samples in member_samples.items():
            if m in val_members:
                val.extend(samples)
            else:
                train.extend(samples)
        return train, val

    def save_split_files(self, train, val):
        """保存训练/验证集标签文件"""

        def write_file(path, samples):
            with open(path, "w", encoding="utf-8") as f:
                for img_path, label in samples:
                    rel_path = os.path.relpath(img_path, self.processed_dir)
                    f.write(f"{rel_path} {label}\n")

        write_file(os.path.join(self.processed_dir, "train.txt"), train)
        write_file(os.path.join(self.processed_dir, "val.txt"), val)
        print(f"训练集：{len(train)} 样本，验证集：{len(val)} 样本")

    def run(self):
        """批量处理所有member"""
        all_samples = []
        for member in tqdm(self.members, desc="处理所有member"):
            all_samples.extend(self.process_member(member))

        # 保存类别映射
        with open(os.path.join(self.processed_dir, "classes.json"), "w") as f:
            json.dump(self.classes, f, indent=2)

        # 划分并保存训练/验证集
        train, val = self.split_train_val(all_samples)
        self.save_split_files(train, val)
        print(f"数据处理完成，总样本数：{len(all_samples)}")
