"""
config.py — V2 版本配置文件
项目：建筑平面图轮廓 → 功能分区色块
输入域 A：白色线稿（轮廓）在黑底上
输出域 B：功能分区色块在黑底上
"""

import os
import torch

# ─── 路径配置 ────────────────────────────────────────────────────────────────

BASE_DIR = r"D:\Claude_jiajia\v2_floorplan"

DATA_DIR        = os.path.join(BASE_DIR, "data")
TRAIN_DIR       = os.path.join(DATA_DIR, "train")   # 512×256 并排图（左A右B）
VAL_DIR         = os.path.join(DATA_DIR, "val")     # 512×256 并排图
TEST_DIR        = os.path.join(DATA_DIR, "test")    # 256×256 纯输入图

CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PROGRESS_DIR    = os.path.join(RESULTS_DIR, "train_progress")
INFERENCE_DIR   = os.path.join(RESULTS_DIR, "inference")

# ─── 设备 ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 超参数 ──────────────────────────────────────────────────────────────────

IMG_SIZE        = 256       # 训练/推理时的图像尺寸
JITTER_SIZE     = 286       # 数据增强：先放大到此尺寸再随机裁剪
BATCH_SIZE      = 1
EPOCHS          = 480       # V2：数据量较多，训练更多轮次
LR              = 0.0002
BETA1           = 0.5
BETA2           = 0.999
LAMBDA_L1       = 100       # L1 loss 权重

SAVE_FREQ       = 10        # 每 N 个 epoch 保存 checkpoint 和训练进度图
NUM_WORKERS     = 0         # Windows 下建议设为 0

# ─── 输入域 A — 建筑平面图轮廓线稿 ─────────────────────────────────────────
# 白色线条在黑底上，无需特别分类

INPUT_CLASSES = {
    "OUTLINE": (255, 255, 255),  # 白色轮廓线
    "BACKGROUND": (0, 0, 0),    # 黑色背景
}

# ─── 输出域 B — 功能分区色块 ─────────────────────────────────────────────────
# 两个数据集混合训练，共用以下色彩方案（部分颜色两个数据集共享含义不同，但颜色一致）
#
# 数据集 1 — Recycling Factory（回收工厂）：
#   Processing Hall  → 红色 (255, 0, 0)
#   Office           → 绿色 (0, 255, 0)
#   Storage          → 蓝色 (0, 0, 255)
#   Courtyard        → 青色 (0, 255, 255)
#
# 数据集 2 — Experience Center（参观中心）：
#   Exhibition Gallery   → 红色 (255, 0, 0)
#   Interactive Lab      → 青色 (0, 255, 255)
#   Impact Simulation    → 蓝色 (0, 0, 255)
#   Leisure Area         → 紫色 (255, 0, 255)

OUTPUT_CLASSES = {
    "RED":    (255,   0,   0),  # Processing Hall / Exhibition Gallery
    "GREEN":  (  0, 255,   0),  # Office（仅回收工厂）
    "BLUE":   (  0,   0, 255),  # Storage / Impact Simulation
    "CYAN":   (  0, 255, 255),  # Courtyard / Interactive Lab
    "MAGENTA":(255,   0, 255),  # Leisure Area（仅参观中心）
}
