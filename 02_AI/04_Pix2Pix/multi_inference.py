"""
multi_inference.py — 批量多样性推理脚本

两种生成多样结果的方式：
  1. 多 checkpoint：用不同训练阶段（epoch 100/200/300）的权重分别推理
  2. Dropout 采样：推理时保留 dropout，同一张图多次采样得到不同结果

用法示例：
  # 方式1：所有可用 checkpoint 各跑一遍
  python multi_inference.py --input_dir data/test --mode checkpoints

  # 方式2：只用最新 checkpoint，每张图采样 5 次
  python multi_inference.py --input_dir data/test --mode dropout --samples 5

  # 方式3：两种都跑（推荐，结果最多）
  python multi_inference.py --input_dir data/test --mode both --samples 5

  # 指定特定 epoch 的 checkpoint
  python multi_inference.py --input_dir data/test --mode checkpoints --epochs 100 200 300
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(r"D:\Claude_jiajia\pix2pix_project")

from model import Generator
from config import DEVICE, CHECKPOINT_DIR, INFERENCE_DIR, IMG_SIZE


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def preprocess(img_path: str) -> torch.Tensor:
    """读取图片 → [-1,1] Tensor，shape (1,3,H,W)"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """单张生成图 Tensor → PIL Image"""
    arr = tensor[0].detach().cpu().float()
    arr = (arr * 0.5 + 0.5).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_comparison(input_tensor, gen_tensor, save_path: str):
    """保存输入+生成的横向拼接对比图"""
    input_pil = tensor_to_pil(input_tensor)
    gen_pil   = tensor_to_pil(gen_tensor)
    w, h = input_pil.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(input_pil, (0, 0))
    canvas.paste(gen_pil,   (w, 0))
    canvas.save(save_path)


def load_generator(checkpoint_path: str, dropout_mode: bool = False) -> Generator:
    """
    加载 Generator。
    dropout_mode=True 时保持 train 模式（dropout 激活），用于随机采样。
    """
    G = Generator().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = ckpt.get("G", ckpt)
    G.load_state_dict(state_dict)
    if dropout_mode:
        G.train()   # 保留 dropout，每次 forward 结果不同
    else:
        G.eval()
    return G


def find_all_checkpoints(ckpt_dir: str, target_epochs: list = None) -> list:
    """
    扫描 checkpoint 文件夹，返回 (epoch_label, path) 列表。
    target_epochs: 指定 epoch 列表，None 表示返回所有。
    """
    if not os.path.exists(ckpt_dir):
        return []

    results = []

    # 按 epoch 编号的 checkpoint
    for f in sorted(os.listdir(ckpt_dir)):
        if f.startswith("ckpt_epoch_") and f.endswith(".pth"):
            epoch_num = int(f.replace("ckpt_epoch_", "").replace(".pth", ""))
            if target_epochs is None or epoch_num in target_epochs:
                results.append((f"epoch_{epoch_num:03d}", os.path.join(ckpt_dir, f)))

    # final checkpoint（如果存在且未被 epoch 覆盖）
    final = os.path.join(ckpt_dir, "ckpt_final.pth")
    if os.path.exists(final) and (target_epochs is None):
        results.append(("final", final))

    return results


# ─── 推理模式 ─────────────────────────────────────────────────────────────────

def compute_metrics(gen_tensor: torch.Tensor) -> float:
    """
    计算生成图的质量指标：有色像素占比（非黑色像素比例）。
    值越高说明模型生成了更多功能分区色块，越低说明输出偏黑/欠拟合。
    """
    arr = gen_tensor[0].detach().cpu().float()
    arr = (arr * 0.5 + 0.5).clamp(0, 1)          # [0,1], shape (3,H,W)
    # 像素最大通道值 > 0.15 视为有色
    colored = (arr.max(dim=0).values > 0.15).float()
    return colored.mean().item()


def run_checkpoint_mode(input_paths: list, output_root: str, target_epochs: list = None):
    """用每个 checkpoint 分别推理，图片平铺在 output_root，并输出 metrics.txt"""
    checkpoints = find_all_checkpoints(CHECKPOINT_DIR, target_epochs)
    if not checkpoints:
        print("未找到任何 checkpoint，请先训练或检查路径。")
        return

    print(f"\n[Checkpoint 模式] 找到 {len(checkpoints)} 个 checkpoint")

    os.makedirs(output_root, exist_ok=True)

    # 收集所有 (label, step, value) 用于计算 smoothed / relative
    records = []

    for label, ckpt_path in checkpoints:
        # 从 label 提取 step（epoch 数字）
        try:
            step = int(label.replace("epoch_", "").replace("final", "9999"))
        except ValueError:
            step = 0

        print(f"\n── {label}: {ckpt_path}")
        G = load_generator(ckpt_path, dropout_mode=False)

        with torch.no_grad():
            for img_path in input_paths:
                stem = img_path.stem
                inp = preprocess(str(img_path)).to(DEVICE)
                gen = G(inp)

                tensor_to_pil(gen).save(os.path.join(output_root, f"gen_{label}_{stem}.png"))
                save_comparison(inp, gen, os.path.join(output_root, f"compare_{label}_{stem}.png"))

                value = compute_metrics(gen)
                records.append((f"gen_{label}_{stem}.png", step, value))
                print(f"  {stem} → gen_{label}_{stem}.png  value={value:.4f}")

    # ── 计算 smoothed（EMA, factor=0.6，与 TensorBoard 一致）和 relative ──
    txt_path = os.path.join(output_root, "metrics.txt")
    smoothed = None
    first_value = None

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{'filename':<40} {'step':>6} {'value':>8} {'smoothed':>10} {'relative':>10}\n")
        f.write("-" * 80 + "\n")
        for filename, step, value in records:
            # EMA
            if smoothed is None:
                smoothed = value
            else:
                smoothed = 0.6 * smoothed + 0.4 * value
            # relative（相对第一张的变化比例）
            if first_value is None:
                first_value = value
            relative = (value - first_value) / (first_value + 1e-8)

            line = f"{filename:<40} {step:>6} {value:>8.4f} {smoothed:>10.4f} {relative:>+10.4f}"
            f.write(line + "\n")
            print(line)

    print(f"\nCheckpoint 模式完成，结果在：{output_root}")
    print(f"指标文件：{txt_path}")


def run_dropout_mode(input_paths: list, output_root: str,
                     n_samples: int = 5, checkpoint_path: str = None):
    """同一 checkpoint + dropout 采样多次，结果存到 dropout_samples/ 子文件夹"""
    # 找 checkpoint
    if checkpoint_path is None:
        checkpoints = find_all_checkpoints(CHECKPOINT_DIR)
        if not checkpoints:
            print("未找到任何 checkpoint。")
            return
        label, checkpoint_path = checkpoints[-1]   # 用最新的
    else:
        label = Path(checkpoint_path).stem

    print(f"\n[Dropout 采样模式] 使用 {label}，每张图采样 {n_samples} 次")
    G = load_generator(checkpoint_path, dropout_mode=True)

    save_dir = os.path.join(output_root, f"dropout_{label}")
    os.makedirs(save_dir, exist_ok=True)

    for img_path in input_paths:
        stem = img_path.stem
        inp = preprocess(str(img_path)).to(DEVICE)

        samples = []
        with torch.no_grad():
            for i in range(n_samples):
                gen = G(inp)
                out_path = os.path.join(save_dir, f"gen_{stem}_s{i+1:02d}.png")
                tensor_to_pil(gen).save(out_path)
                samples.append(gen)
                print(f"  {stem} 采样 {i+1}/{n_samples} → gen_{stem}_s{i+1:02d}.png")

        # 拼接所有采样为横排总览图
        pils = [tensor_to_pil(s) for s in samples]
        w, h = pils[0].size
        canvas = Image.new("RGB", (w * len(pils), h))
        for idx, p in enumerate(pils):
            canvas.paste(p, (idx * w, 0))
        canvas.save(os.path.join(save_dir, f"overview_{stem}.png"))
        print(f"  → overview_{stem}.png（{n_samples} 张拼排）")

    print(f"\nDropout 模式完成，结果在：{save_dir}")


# ─── 主入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pix2Pix V2 多样性推理")
    parser.add_argument("--input_dir",  required=True,
                        help="输入图像文件夹（256×256 轮廓线稿）")
    parser.add_argument("--output_dir", default=None,
                        help="结果根目录，默认 results/inference/multi/")
    parser.add_argument("--mode",       default="both",
                        choices=["checkpoints", "dropout", "both"],
                        help="推理模式（默认 both）")
    parser.add_argument("--epochs",     nargs="+", type=int, default=None,
                        help="指定 checkpoint epoch，例如 --epochs 100 200 300")
    parser.add_argument("--samples",    type=int, default=5,
                        help="Dropout 采样次数（默认 5）")
    parser.add_argument("--checkpoint", default=None,
                        help="Dropout 模式指定 checkpoint 路径")
    args = parser.parse_args()

    # 收集输入图片
    input_dir = Path(args.input_dir)
    input_paths = sorted(
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.bmp"))
    )
    if not input_paths:
        raise FileNotFoundError(f"在 {args.input_dir} 中未找到任何图像文件")
    print(f"找到 {len(input_paths)} 张输入图像")

    output_root = args.output_dir or os.path.join(INFERENCE_DIR, "multi")
    os.makedirs(output_root, exist_ok=True)

    if args.mode in ("checkpoints", "both"):
        run_checkpoint_mode(input_paths, output_root, target_epochs=args.epochs)

    if args.mode in ("dropout", "both"):
        run_dropout_mode(input_paths, output_root,
                         n_samples=args.samples,
                         checkpoint_path=args.checkpoint)

    print(f"\n全部完成！所有结果在：{output_root}")
