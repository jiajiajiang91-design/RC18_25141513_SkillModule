"""
batch_voronoi.py — 批量 Voronoi 后处理 (v5 — 固定参考 mask)

所有 epoch 共用同一个建筑 mask（从最优 epoch 360 提取），
确保每张输出的外轮廓形状完全一致。

用法：
  python batch_voronoi.py
  python batch_voronoi.py --ref_mask results/inference/multi/gen_epoch_360_test_001.png
"""

import os
import sys
import re
import argparse
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INFERENCE_DIR, BASE_DIR
from voronoi_postprocess import process, DOT_SPACING, DOT_RADIUS

MULTI_DIR      = os.path.join(INFERENCE_DIR, "multi")
VORONOI_DIR    = os.path.join(BASE_DIR, "results", "voronoi")
DEFAULT_OUTLINE = os.path.join(BASE_DIR, "reference.png")


def main():
    parser = argparse.ArgumentParser(description="批量 Voronoi 后处理 (固定mask)")
    parser.add_argument("--input_dir",   default=MULTI_DIR)
    parser.add_argument("--output_dir",  default=VORONOI_DIR)
    parser.add_argument("--outline",     default=DEFAULT_OUTLINE,
                        help="白色轮廓线图路径，用于提取精确建筑 mask（默认 reference.png）")
    parser.add_argument("--dot_spacing", type=int, default=DOT_SPACING)
    parser.add_argument("--dot_radius",  type=int, default=DOT_RADIUS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    outline = args.outline if os.path.exists(args.outline) else None
    if not outline:
        print(f"WARNING: outline not found at {args.outline}, falling back to gen-image mask")
    else:
        print(f"Using outline mask from: {outline}")

    gen_files = sorted(Path(args.input_dir).glob("gen_epoch_*.png"))
    # 也处理 gen_final
    final = list(Path(args.input_dir).glob("gen_final_*.png"))
    gen_files = gen_files + final

    if not gen_files:
        print(f"No gen_epoch_*.png found in {args.input_dir}")
        return

    print(f"Found {len(gen_files)} images, starting Voronoi processing...")
    print(f"Output: {args.output_dir}\n")

    for idx, gen_path in enumerate(gen_files, 1):
        match = re.search(r"(epoch_\d+|final)", gen_path.name)
        label = match.group(1) if match else gen_path.stem

        voronoi_out = os.path.join(args.output_dir, f"voronoi_{label}.png")
        compare_out = os.path.join(args.output_dir, f"compare_{label}.png")

        print(f"[{idx:03d}/{len(gen_files):03d}] {gen_path.name}")

        vor_arr = process(
            str(gen_path), None, voronoi_out,
            dot_spacing=args.dot_spacing,
            dot_radius=args.dot_radius,
            outline_path=outline,
        )

        gen_pil = Image.open(str(gen_path)).convert("RGB").resize((256, 256))
        vor_pil = Image.fromarray(vor_arr)
        canvas = Image.new("RGB", (512, 256))
        canvas.paste(gen_pil, (0, 0))
        canvas.paste(vor_pil, (256, 0))
        canvas.save(compare_out)

    print(f"\nDone! {len(gen_files)} images processed → {args.output_dir}")


if __name__ == "__main__":
    main()
