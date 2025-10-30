# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/10/30 下午3:39
# @Site : 
# @file : test_import.py
# @Software : PyCharm


import os
import sys

# 强制添加backend目录到路径
current_script = os.path.abspath(__file__)  # experiments/run_exp1.py的绝对路径
backend_dir = os.path.dirname(os.path.dirname(current_script))  # 得到backend目录
sys.path.append(backend_dir)

# 打印所有Python路径（关键调试信息）
print("Python路径列表：")
for path in sys.path:
    print(f"- {path}")

# 验证backend目录是否正确
print(f"\nbackend目录是否在路径中：{backend_dir in sys.path}")
print(f"backend目录实际路径：{backend_dir}")
print(f"backend目录是否存在：{os.path.exists(backend_dir)}")

# 尝试逐个导入（定位具体哪个模块失败）
try:
    import models
    print("\n成功导入models模块")
except ImportError as e:
    print(f"\n导入models失败：{e}")

try:
    from models import base_cnn
    print("成功导入models.base_cnn")
except ImportError as e:
    print(f"导入models.base_cnn失败：{e}")

try:
    from models.base_cnn import BaseCNN
    print("成功导入BaseCNN类")
except ImportError as e:
    print(f"导入BaseCNN失败：{e}")

try:
    import data
    print("\n成功导入data模块")
except ImportError as e:
    print(f"\n导入data失败：{e}")

try:
    from data.dataset import GestureDataset
    print("成功导入GestureDataset类")
except ImportError as e:
    print(f"导入GestureDataset失败：{e}")

try:
    import training
    print("\n成功导入training模块")
except ImportError as e:
    print(f"\n导入training失败：{e}")

try:
    from training.trainer import Trainer
    print("成功导入Trainer类")
except ImportError as e:
    print(f"导入Trainer失败：{e}")