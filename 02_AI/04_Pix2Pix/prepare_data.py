"""
prepare_data.py — V2 数据准备脚本
将两个数据集合并、打乱，分配到 train/val 文件夹

用法：python prepare_data.py
"""

import os
import random
import shutil
from PIL import Image

# ─── 源数据路径 ───────────────────────────────────────────────────────────────

SOURCE_DIRS = [
    r"D:\Claude_jiajia\pix2pix_project\数据训练\回收工厂_测试数据",
    r"D:\Claude_jiajia\pix2pix_project\数据训练\参观中心_测试数据",
]

# ─── 目标路径 ─────────────────────────────────────────────────────────────────

TRAIN_DIR = r"D:\Claude_jiajia\v2_floorplan\data\train"
VAL_DIR   = r"D:\Claude_jiajia\v2_floorplan\data\val"

TRAIN_COUNT = 50
VAL_COUNT   = 10

SEED = 42


def convert_to_rgb_if_needed(src_path: str, dst_path: str):
    """复制图片，如果是 RGBA 则合并 alpha 到黑底转为 RGB"""
    img = Image.open(src_path)
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])  # alpha channel as mask
        background.save(dst_path)
    else:
        img = img.convert("RGB")
        img.save(dst_path)


def main():
    # 收集所有源图片路径
    all_images = []
    for src_dir in SOURCE_DIRS:
        if not os.path.exists(src_dir):
            print(f"[警告] 源文件夹不存在：{src_dir}")
            continue
        pngs = sorted([
            os.path.join(src_dir, f)
            for f in os.listdir(src_dir)
            if f.lower().endswith(".png")
        ])
        print(f"从 {src_dir} 找到 {len(pngs)} 张图片")
        all_images.extend(pngs)

    print(f"\n合计 {len(all_images)} 张图片")

    if len(all_images) < TRAIN_COUNT + VAL_COUNT:
        print(f"[错误] 图片数量不足，需要至少 {TRAIN_COUNT + VAL_COUNT} 张，实际 {len(all_images)} 张")
        return

    # 随机打乱
    random.seed(SEED)
    random.shuffle(all_images)

    # 分配
    train_images = all_images[:TRAIN_COUNT]
    val_images   = all_images[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]

    # 确保目标文件夹存在
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    # 复制训练集
    print(f"\n─── 复制训练集到 {TRAIN_DIR} ───")
    for i, src in enumerate(train_images, 1):
        dst = os.path.join(TRAIN_DIR, f"{i:03d}.png")
        convert_to_rgb_if_needed(src, dst)
        print(f"  [{i:02d}/50] {src}  →  {dst}")

    # 复制验证集
    print(f"\n─── 复制验证集到 {VAL_DIR} ───")
    for i, src in enumerate(val_images, 1):
        dst = os.path.join(VAL_DIR, f"{i:03d}.png")
        convert_to_rgb_if_needed(src, dst)
        print(f"  [{i:02d}/10] {src}  →  {dst}")

    print(f"\n数据准备完成！训练集 {TRAIN_COUNT} 张，验证集 {VAL_COUNT} 张")


if __name__ == "__main__":
    main()
