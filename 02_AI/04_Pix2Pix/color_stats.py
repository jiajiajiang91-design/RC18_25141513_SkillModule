"""
color_stats.py — 生成图色彩占比统计

扫描推理输出文件夹中所有 gen_*.png，统计每张图的功能分区色彩占比，
结果写入同一个 color_stats.txt。

用法：
  python color_stats.py --input_dir results/inference/multi
  python color_stats.py --input_dir results/inference/multi --output stats.txt
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INFERENCE_DIR

# ─── 颜色定义（容差范围内归为该类）────────────────────────────────────────────

COLORS = [
    ("Processing Hall (红)", (255,   0,   0)),
    ("Office          (绿)", (  0, 255,   0)),
    ("Storage         (蓝)", (  0,   0, 255)),
    ("Courtyard       (青)", (  0, 255, 255)),
    ("Leisure Area    (紫)", (255,   0, 255)),
    ("Background      (黑)", (  0,   0,   0)),
]

TOLERANCE = 60   # 像素各通道与目标颜色的最大允许偏差


def classify_pixels(img_array: np.ndarray) -> dict:
    """
    将图像每个像素归类到最近的颜色分类。
    返回 {颜色名: 占比} 字典，占比为 0~100 的浮点数。
    """
    h, w, _ = img_array.shape
    total = h * w

    # 对每个像素计算与各颜色的 L∞ 距离，取最近的
    img_f = img_array.astype(np.float32)             # (H, W, 3)
    counts = {name: 0 for name, _ in COLORS}
    unclassified = 0

    # 预计算各颜色中心
    centers = np.array([rgb for _, rgb in COLORS], dtype=np.float32)  # (N, 3)

    # 展平像素 (H*W, 3)
    pixels = img_f.reshape(-1, 3)

    # 计算每像素到每个颜色中心的 L∞ 距离 (H*W, N)
    diff = np.abs(pixels[:, None, :] - centers[None, :, :])   # (H*W, N, 3)
    linf = diff.max(axis=2)                                     # (H*W, N)

    nearest_idx = linf.argmin(axis=1)    # (H*W,)
    nearest_dist = linf.min(axis=1)      # (H*W,)

    for i, (name, _) in enumerate(COLORS):
        mask = (nearest_idx == i) & (nearest_dist <= TOLERANCE)
        counts[name] = int(mask.sum())

    unclassified = int((nearest_dist > TOLERANCE).sum())

    # 转为占比（排除背景后的"功能区"占比 + 背景占比）
    result = {}
    for name, _ in COLORS:
        result[name] = counts[name] / total * 100
    result["未分类"] = unclassified / total * 100
    return result


def format_bar(pct: float, width: int = 20) -> str:
    """生成简单的 ASCII 进度条"""
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def analyze_folder(input_dir: str, output_txt: str):
    input_path = Path(input_dir)

    # 只处理 gen_ 开头的图（纯生成图，不含 compare 对比图）
    images = sorted(input_path.glob("gen_*.png"))
    if not images:
        # 也兼容没有 gen_ 前缀的情况
        images = sorted(
            list(input_path.glob("*.png")) +
            list(input_path.glob("*.jpg"))
        )

    if not images:
        print(f"在 {input_dir} 中未找到图像文件")
        return

    print(f"找到 {len(images)} 张图像，开始统计...")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Space Ratio — 功能分区色彩占比统计\n")
        f.write("=" * 70 + "\n\n")

        for idx, img_path in enumerate(images, 1):
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            stats = classify_pixels(arr)

            # 计算功能区总占比（排除黑色背景和未分类）
            func_keys = [n for n, _ in COLORS if "黑" not in n]
            func_total = sum(stats[n] for n in func_keys)

            header = f"[{idx:03d}/{len(images):03d}] {img_path.name}"
            print(header)
            f.write(header + "\n")
            f.write("-" * 70 + "\n")

            for name, _ in COLORS:
                pct = stats[name]
                bar = format_bar(pct)
                line = f"  ■ {name:<24} {bar}  {pct:6.2f}%"
                print(line)
                f.write(line + "\n")

            # 未分类
            if stats["未分类"] > 0.5:
                line = f"  · {'未分类':<24} {'░'*20}  {stats['未分类']:6.2f}%"
                f.write(line + "\n")

            # 功能区合计
            summary = f"  → 功能区合计（非黑色）: {func_total:.2f}%"
            print(summary)
            f.write(summary + "\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write(f"共统计 {len(images)} 张图像\n")

    print(f"\n统计完成！结果已保存至：{output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="功能分区色彩占比统计")
    parser.add_argument("--input_dir",  default=os.path.join(INFERENCE_DIR, "multi"),
                        help="包含生成图的文件夹（默认 results/inference/multi）")
    parser.add_argument("--output",     default=None,
                        help="输出 txt 文件路径（默认存在 input_dir 同目录）")
    args = parser.parse_args()

    output_txt = args.output or os.path.join(args.input_dir, "color_stats.txt")
    analyze_folder(args.input_dir, output_txt)
