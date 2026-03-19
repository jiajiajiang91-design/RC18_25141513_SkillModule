"""
train.py — V2 建筑平面图功能分区训练循环
用法：python train.py

复用上层项目的 model.py 和 dataset.py，使用本目录的 config.py
"""

import os
import sys
import time

import torch
import torch.nn as nn

# 先把 v2 目录插到 sys.path 最前面，确保 dataset.py / utils.py 的
# `from config import ...` 找到的是 v2 的 config，而非原版 config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 再追加原版项目路径，以便 import model / dataset / utils
sys.path.append(r"D:\Claude_jiajia\pix2pix_project")

from model import Generator, Discriminator
from dataset import get_dataloader
from utils import save_progress_image, print_loss

from config import (
    DEVICE, EPOCHS, LR, BETA1, BETA2, LAMBDA_L1,
    CHECKPOINT_DIR, SAVE_FREQ, TRAIN_DIR, VAL_DIR,
)


def main():
    print(f"使用设备：{DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ─── 数据集 ────────────────────────────────────────────────────────────
    train_loader = get_dataloader("train")
    val_loader   = get_dataloader("val")
    print(f"训练集：{len(train_loader.dataset)} 张图像")
    print(f"验证集：{len(val_loader.dataset)} 张图像")

    # ─── 模型 ──────────────────────────────────────────────────────────────
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    # ─── 损失函数 ──────────────────────────────────────────────────────────
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1  = nn.L1Loss()

    # ─── 优化器 ────────────────────────────────────────────────────────────
    optimizer_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    # ─── Learning Rate Scheduler ────────────────────────────────────────────
    def lr_lambda(epoch):
        decay_start = EPOCHS // 2
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / float(EPOCHS - decay_start)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

    # ─── 断点续训 ──────────────────────────────────────────────────────────
    start_epoch = 1
    latest_ckpt = _find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        optimizer_G.load_state_dict(ckpt["optimizer_G"])
        optimizer_D.load_state_dict(ckpt["optimizer_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"已从 checkpoint 恢复，继续从 epoch {start_epoch} 训练")

    # ─── 训练循环 ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        G.train()
        D.train()

        for step, (real_A, real_B) in enumerate(train_loader, 1):
            real_A = real_A.to(DEVICE)
            real_B = real_B.to(DEVICE)

            # ── 训练 Discriminator ────────────────────────────────────────
            optimizer_D.zero_grad()

            fake_B = G(real_A)
            pred_real = D(real_A, real_B)
            pred_fake = D(real_A, fake_B.detach())

            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            loss_D.backward()
            optimizer_D.step()

            # ── 训练 Generator ────────────────────────────────────────────
            optimizer_G.zero_grad()

            fake_B = G(real_A)
            pred_fake = D(real_A, fake_B)

            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1  = criterion_L1(fake_B, real_B) * LAMBDA_L1
            loss_G = loss_G_GAN + loss_G_L1

            loss_G.backward()
            optimizer_G.step()

            # 每 100 步打印一次
            if step % 100 == 0 or step == len(train_loader):
                print_loss(
                    epoch, EPOCHS, step, len(train_loader),
                    loss_D.item(), loss_G.item(),
                    loss_G_GAN.item(), loss_G_L1.item(),
                )

        scheduler_G.step()
        scheduler_D.step()

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{EPOCHS} 完成，耗时 {elapsed:.1f}s")

        # ── 每 SAVE_FREQ 个 epoch 保存 checkpoint 和训练进度图 ────────────
        if epoch % SAVE_FREQ == 0:
            _save_checkpoint(G, D, optimizer_G, optimizer_D, epoch)

            G.eval()
            with torch.no_grad():
                for val_A, val_B in val_loader:
                    val_A, val_B = val_A.to(DEVICE), val_B.to(DEVICE)
                    fake_B = G(val_A)
                    save_path = save_progress_image(val_A, fake_B, val_B, epoch)
                    print(f"进度图已保存：{save_path}")
                    break  # 只取第一批
            G.train()

    # 训练结束保存最终 checkpoint
    _save_checkpoint(G, D, optimizer_G, optimizer_D, EPOCHS, final=True)
    print("训练完成！")


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _save_checkpoint(G, D, opt_G, opt_D, epoch: int, final: bool = False):
    tag = "final" if final else f"epoch_{epoch:04d}"
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{tag}.pth")
    torch.save({
        "epoch":       epoch,
        "G":           G.state_dict(),
        "D":           D.state_dict(),
        "optimizer_G": opt_G.state_dict(),
        "optimizer_D": opt_D.state_dict(),
    }, path)
    print(f"Checkpoint 已保存：{path}")


def _find_latest_checkpoint(ckpt_dir: str):
    """查找最新的（非 final）checkpoint 用于断点续训"""
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pth")]
    )
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else None


if __name__ == "__main__":
    main()
