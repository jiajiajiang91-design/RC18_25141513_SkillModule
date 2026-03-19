"""
inference.py — V2 建筑平面图功能分区推理脚本
用法：
    python inference.py --input_dir <输入文件夹> [--checkpoint <权重文件>]

参数：
    --input_dir   包含输入图像的文件夹路径（256×256 轮廓线稿）
    --checkpoint  模型权重文件路径（.pth），默认自动加载最新 checkpoint
    --output_dir  结果保存路径，默认 results/inference/
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# 先把 v2 目录插到 sys.path 最前面，确保 utils.py 的
# `from config import ...` 找到的是 v2 的 config，而非原版 config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 再追加原版项目路径，以便 import model / utils
sys.path.append(r"D:\Claude_jiajia\pix2pix_project")

from model import Generator
from utils import save_inference_result
from config import DEVICE, CHECKPOINT_DIR, INFERENCE_DIR, IMG_SIZE


def load_generator(checkpoint_path: str) -> Generator:
    """加载 Generator 权重"""
    G = Generator().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    # 兼容直接存 state_dict 和存完整 dict 两种格式
    state_dict = ckpt.get("G", ckpt)
    G.load_state_dict(state_dict)
    G.eval()
    print(f"模型已加载：{checkpoint_path}")
    return G


def preprocess(img_path: str) -> torch.Tensor:
    """读取单张图片 → 归一化到 [-1,1] 的 Tensor，shape (1,3,H,W)"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def run_inference(input_dir: str, checkpoint_path: str = None, output_dir: str = None):
    """
    批量处理输入文件夹中的所有图片。

    Args:
        input_dir:       包含输入图像的文件夹
        checkpoint_path: .pth 文件路径，None 时自动寻找最新 checkpoint
        output_dir:      结果保存路径
    """
    # 自动寻找最新 checkpoint
    if checkpoint_path is None:
        checkpoint_path = _find_latest_checkpoint(CHECKPOINT_DIR)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"在 {CHECKPOINT_DIR} 中未找到任何 checkpoint，"
                "请先运行 train.py 或手动指定 --checkpoint 参数。"
            )

    if output_dir is None:
        output_dir = INFERENCE_DIR

    os.makedirs(output_dir, exist_ok=True)

    G = load_generator(checkpoint_path)

    # 获取所有图片路径
    input_paths = sorted(
        list(Path(input_dir).glob("*.png")) +
        list(Path(input_dir).glob("*.jpg")) +
        list(Path(input_dir).glob("*.bmp"))
    )
    if not input_paths:
        raise FileNotFoundError(f"在 {input_dir} 中未找到任何图像文件")

    print(f"共找到 {len(input_paths)} 张图像，开始推理...")

    with torch.no_grad():
        for i, img_path in enumerate(input_paths, 1):
            input_tensor = preprocess(str(img_path)).to(DEVICE)
            generated = G(input_tensor)

            # 保存：result_<原文件名>（输入+生成对比图）
            stem = img_path.stem
            save_inference_result(
                input_tensor, generated,
                save_path=output_dir,
                filename=f"result_{stem}.png",
            )

            # 也单独保存生成图
            gen_np = generated[0].detach().cpu().float()
            gen_np = (gen_np * 0.5 + 0.5).clamp(0, 1)
            gen_pil = Image.fromarray(
                (gen_np.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            gen_pil.save(os.path.join(output_dir, f"generated_{stem}.png"))

            print(f"[{i}/{len(input_paths)}] {img_path.name} → result_{stem}.png")

    print(f"\n推理完成！结果已保存至：{output_dir}")


def _find_latest_checkpoint(ckpt_dir: str):
    if not os.path.exists(ckpt_dir):
        return None
    # 优先使用 final checkpoint
    final = os.path.join(ckpt_dir, "ckpt_final.pth")
    if os.path.exists(final):
        return final
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pth")]
    )
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pix2Pix V2 建筑平面图推理")
    parser.add_argument("--input_dir",  required=True, help="输入图像文件夹路径")
    parser.add_argument("--checkpoint", default=None,  help="模型权重 .pth 文件路径")
    parser.add_argument("--output_dir", default=None,  help="结果保存路径")
    args = parser.parse_args()

    run_inference(
        input_dir=args.input_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
