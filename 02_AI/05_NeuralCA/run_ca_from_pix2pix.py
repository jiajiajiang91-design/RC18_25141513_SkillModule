# -*- coding: utf-8 -*-
"""
run_ca_from_pix2pix.py
================================================================================
加载 pix2pix_to_ca.py 生成的初始状态矩阵，用已训练好的 Neural CA 跑演化。

支持多分辨率推理（通过 --resolution 参数控制）：
    96px  : 原始训练尺寸（最稳定，扩散速度最接近训练表现）
    160px : 中等尺寸（平衡精度和扩散速度）
    288px : 全分辨率（最精细但扩散可能碎片化——也是一种有趣的涌现）

用法：
    # 单分辨率
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --resolution 96
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --resolution 160
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --resolution 288

    # 默认：跑全部三个分辨率并输出对比图
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --all_resolutions

    # 自定义步数和 checkpoint 路径
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --resolution 96 \
        --steps 300 \
        --checkpoint "D:/Claude_jiajia/Documents/260224-1_generate_neural_ca/checkpoints/final"
================================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import PIL.Image
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# 在无显示环境中使用非交互后端
if not os.environ.get('DISPLAY') and sys.platform != 'win32':
    matplotlib.use('Agg')

# ── TensorFlow & GPU 设置 ─────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'   # 抑制 TF Info 输出，保留警告和错误
import tensorflow as tf

# 将 Neural CA 模型定义所在目录加入 sys.path
CA_MODEL_DIR = r"D:\Claude_jiajia\Documents"
if CA_MODEL_DIR not in sys.path:
    sys.path.insert(0, CA_MODEL_DIR)

# 从已有的 decay_neural_ca.py 中导入 CAModel 和可视化工具
# 这样可以保证推理时使用与训练完全一致的模型架构
try:
    from decay_neural_ca import CAModel, to_rgb, CHANNEL_N, TARGET_PADDING
    print(f"  [OK] 成功导入 CAModel 自 {CA_MODEL_DIR}/decay_neural_ca.py")
except ImportError as e:
    print(f"  [错误] 无法导入 CAModel: {e}")
    print(f"  请确认 decay_neural_ca.py 在以下路径: {CA_MODEL_DIR}")
    sys.exit(1)


# =============================================================================
# 默认路径
# =============================================================================

# 训练好的 CA 权重（优先使用更长训练时间的版本）
DEFAULT_CHECKPOINT = r"D:\Claude_jiajia\Documents\260224-1_generate_neural_ca\checkpoints\final"

# 演化步数
DEFAULT_STEPS = 80  # matches training iter_n (64-96); beyond ~100 causes over-evolution

# 帧保存间隔（每 N 步保存一帧）
FRAME_SAVE_INTERVAL = 10

# 三个对比分辨率（单位：像素，含 padding）
COMPARE_RESOLUTIONS = [96, 160, 288]

# 综合对比图中展示的步骤时间点
SUMMARY_STEPS = [0, 20, 40, 60, 80]


# =============================================================================
# GPU 信息打印
# =============================================================================

def print_gpu_info():
    """打印 GPU 检测结果，确认 RTX 5070 Ti 被正确识别。"""
    print("\n  TensorFlow 版本:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  检测到 {len(gpus)} 个 GPU：")
        for g in gpus:
            print(f"    {g.name}")
            # 启用显存按需分配（避免一次性占满 16GB GDDR7）
            try:
                tf.config.experimental.set_memory_growth(g, True)
                print(f"    显存按需分配: 已启用")
            except Exception as ex:
                print(f"    显存配置警告: {ex}")
    else:
        print("  未检测到 GPU，将使用 CPU（速度较慢）")
    print()


# =============================================================================
# 图像处理工具
# =============================================================================

def load_initial_state(npy_path: str) -> np.ndarray:
    """
    加载 pix2pix_to_ca.py 生成的初始状态 .npy 文件。

    返回：float32 数组，shape (1, H, W, 16)
    """
    state = np.load(npy_path).astype(np.float32)
    if state.ndim == 3:
        # 如果没有 batch 维度，添加它
        state = state[None]
    assert state.ndim == 4 and state.shape[-1] == 16, \
        f"期望 shape (1, H, W, 16)，得到 {state.shape}"
    return state


def resize_ca_state(state: np.ndarray, target_size: int) -> np.ndarray:
    """
    将 CA 初始状态 resize 到目标尺寸（含 padding）。

    使用最近邻插值（PIL.Image.NEAREST）保持色块边界清晰，
    避免双线性插值在功能区边界产生混合值。

    参数：
        state       : shape (1, H, W, 16)，float32
        target_size : 目标边长（正方形）

    返回：
        shape (1, target_size, target_size, 16)，float32
    """
    _, H, W, C = state.shape

    if H == target_size and W == target_size:
        return state

    resized_channels = []
    for c in range(C):
        # 提取单通道：(H, W)，转为 uint8（最近邻插值前需要整数图像，0-255范围）
        # 注意：float32 值域 [0, 1]，需要先缩放再插值，再还原
        channel = state[0, :, :, c]  # (H, W)

        # 将浮点值映射到 0-255 范围进行插值（避免精度丢失）
        channel_uint16 = (np.clip(channel, 0, 1) * 65535).astype(np.uint16)
        pil_img = PIL.Image.fromarray(channel_uint16, mode='I;16')
        pil_resized = pil_img.resize((target_size, target_size), PIL.Image.NEAREST)
        channel_resized = np.array(pil_resized, dtype=np.float32) / 65535.0
        resized_channels.append(channel_resized)

    # 重新堆叠：(C, H, W) → (1, H, W, C)
    result = np.stack(resized_channels, axis=-1)[None]  # (1, target_size, target_size, C)
    return result.astype(np.float32)


def ensure_padding(state: np.ndarray, padding_size: int = 16) -> np.ndarray:
    """
    确保 CA 状态四周有足够的 alpha=0 padding（dead zone）。
    如果边界已经是 alpha=0 则跳过。

    参数：
        state        : shape (1, H, W, 16)
        padding_size : 需要的 padding 宽度

    返回：
        可能 padding 后的 state，shape (1, H+2p, W+2p, 16) 或原始 shape
    """
    _, H, W, _ = state.shape
    p = padding_size

    # 检查边界是否已经是 dead zone（alpha 通道 ch3 < 0.05）
    border_alpha = np.concatenate([
        state[0, :p, :, 3].flatten(),    # 顶边
        state[0, -p:, :, 3].flatten(),   # 底边
        state[0, :, :p, 3].flatten(),    # 左边
        state[0, :, -p:, 3].flatten(),   # 右边
    ])

    if border_alpha.max() < 0.05:
        # 边界已经是 dead zone，不需要再加 padding
        return state

    # 加 padding
    padded = np.zeros((1, H + 2*p, W + 2*p, 16), dtype=np.float32)
    padded[0, p:p+H, p:p+W, :] = state[0]
    print(f"  [提示] 已添加 {p}px padding: {H}x{W} → {H+2*p}x{W+2*p}")
    return padded


def state_to_rgb(state: np.ndarray) -> np.ndarray:
    """
    将 CA 状态矩阵 (1, H, W, 16) 转换为可显示的 RGB 图像 (H, W, 3)，uint8。

    使用与训练代码相同的 to_rgb() 逻辑：
        rgb = 1.0 - alpha + rgb_channel  （白色背景合成）
    """
    # to_rgb 期望 TensorFlow tensor，传入 numpy 也可以（自动转换）
    rgb = to_rgb(tf.constant(state)).numpy()[0]  # (H, W, 3)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


# =============================================================================
# CA 推理（单分辨率）
# =============================================================================

def run_ca_evolution(ca_model: CAModel,
                      initial_state: np.ndarray,
                      resolution: int,
                      n_steps: int,
                      output_dir: str,
                      save_interval: int = FRAME_SAVE_INTERVAL,
                      save_hidden: bool = False) -> dict:
    """
    在指定分辨率下跑 CA 演化，保存帧序列和摘要图。

    参数：
        ca_model      : 加载好权重的 CAModel 实例
        initial_state : 原始初始状态 (1, H, W, 16)
        resolution    : 目标分辨率（含 padding 的边长）
        n_steps       : 演化步数
        output_dir    : 帧保存目录
        save_interval : 每隔多少步保存一帧

    返回：
        dict，包含 key_frames（步骤 → RGB数组）和时间统计
    """
    res_dir = os.path.join(output_dir, "frames", f"res_{resolution}")
    os.makedirs(res_dir, exist_ok=True)
    if save_hidden:
        hidden_dir = os.path.join(output_dir, "hidden_states", f"res_{resolution}")
        os.makedirs(hidden_dir, exist_ok=True)

    print(f"\n{'─'*50}")
    print(f"  分辨率: {resolution}x{resolution}  |  步数: {n_steps}")
    print(f"{'─'*50}")

    # ── resize 到目标分辨率 ────────────────────────────────────────────────────
    state = resize_ca_state(initial_state, resolution)
    state = ensure_padding(state, padding_size=16)

    # resize 后实际尺寸（可能因 ensure_padding 而略大）
    _, H, W, _ = state.shape
    print(f"  实际运行尺寸: {H}x{W}")

    # 转为 TensorFlow tensor（利用 GPU 加速）
    x = tf.Variable(state)

    key_frames = {}       # step → RGB uint8 (H, W, 3)
    save_steps = set(range(0, n_steps + 1, save_interval))
    # 确保 SUMMARY_STEPS 中的关键帧一定被保存
    for s in SUMMARY_STEPS:
        if s <= n_steps:
            save_steps.add(s)

    t_start = time.time()

    for step in range(n_steps + 1):
        # 保存帧
        if step in save_steps:
            state_np = x.numpy()
            rgb = state_to_rgb(state_np)
            frame_path = os.path.join(res_dir, f"frame_{step:04d}.png")
            PIL.Image.fromarray(rgb).save(frame_path)

            # 记录关键帧用于摘要
            if step in SUMMARY_STEPS:
                key_frames[step] = rgb

            # 保存完整 16 通道状态（用于隐状态分析）
            if save_hidden and step in SUMMARY_STEPS:
                npy_path = os.path.join(hidden_dir, f"state_t{step:04d}.npy")
                np.save(npy_path, state_np)
                print(f"\n  [hidden] Saved {Path(npy_path).name}")

        # CA 演化（跳过最后一步的更新，只保存）
        if step < n_steps:
            x.assign(ca_model(x))

        # 进度打印（每 50 步一次）
        if step % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (step + 1) * (n_steps - step) if step > 0 else 0
            print(f"\r  step {step:4d}/{n_steps}  |  "
                  f"elapsed: {elapsed:.1f}s  |  ETA: {eta:.1f}s  ", end="")

    t_total = time.time() - t_start
    print(f"\n  完成！耗时 {t_total:.1f}s  ({t_total/n_steps*1000:.1f}ms/step)")

    # ── 保存摘要图（2行4列，8帧并排）────────────────────────────────────────────
    summary_path = os.path.join(output_dir, f"summary_res_{resolution}.png")
    _save_resolution_summary(key_frames, resolution, n_steps, summary_path)
    print(f"  摘要图: {Path(summary_path).name}")

    return {
        "key_frames"  : key_frames,
        "resolution"  : resolution,
        "actual_size" : (H, W),
        "total_time"  : t_total,
        "steps"       : n_steps,
    }


def _save_resolution_summary(key_frames: dict,
                               resolution: int,
                               n_steps: int,
                               save_path: str):
    """
    保存 2行4列的摘要图，展示 8 个关键时间点的衰变状态。
    如果关键帧不足 8 个，用最后一帧补齐。
    """
    # 收集 8 帧（均匀采样）
    all_steps = sorted(key_frames.keys())
    if len(all_steps) >= 8:
        indices = np.linspace(0, len(all_steps) - 1, 8, dtype=int)
        display_steps = [all_steps[i] for i in indices]
    else:
        display_steps = all_steps
        # 用最后一帧补齐到 8 帧
        while len(display_steps) < 8:
            display_steps.append(display_steps[-1])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (ax, step) in enumerate(zip(axes.flat, display_steps)):
        frame = key_frames.get(step, list(key_frames.values())[-1])
        ax.imshow(frame)
        ax.set_title(f't = {step}', fontsize=10)
        ax.axis('off')

    plt.suptitle(
        f'Neural CA Evolution  |  Resolution: {resolution}px  |  {n_steps} steps\n'
        f'Programming Decay — Simulation as Inquiry',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# 多分辨率对比
# =============================================================================

def run_all_resolutions(ca_model: CAModel,
                         initial_state: np.ndarray,
                         resolutions: list,
                         n_steps: int,
                         output_dir: str,
                         save_hidden: bool = False) -> list:
    """
    依次在多个分辨率下跑 CA 演化，最后生成对比图。

    返回：每个分辨率的结果 dict 列表
    """
    results = []
    for res in resolutions:
        result = run_ca_evolution(
            ca_model      = ca_model,
            initial_state = initial_state,
            resolution    = res,
            n_steps       = n_steps,
            output_dir    = output_dir,
            save_hidden   = save_hidden,
        )
        results.append(result)

    # 生成多分辨率对比图
    comparison_path = os.path.join(output_dir, "resolution_comparison.png")
    _save_resolution_comparison(results, SUMMARY_STEPS, n_steps, comparison_path)
    print(f"\n  多分辨率对比图: {comparison_path}")

    return results


def _save_resolution_comparison(results: list,
                                  display_steps: list,
                                  n_steps: int,
                                  save_path: str):
    """
    保存多分辨率对比图：
    - 行：选定的时间步（step 0, 100, 300, 500）
    - 列：不同分辨率（96, 160, 288）

    布局：n_steps_shown 行 × n_resolutions 列
    """
    # 只展示存在的时间步
    valid_steps = [s for s in display_steps if s <= n_steps]

    n_rows = len(valid_steps)
    n_cols = len(results)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 4 * n_rows))
    # 确保 axes 是二维数组
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for col_idx, result in enumerate(results):
        res = result["resolution"]
        key_frames = result["key_frames"]
        t_total = result["total_time"]

        # 列标题（第一行上方）
        axes[0, col_idx].set_title(
            f'{res}×{res}px\n({t_total:.1f}s)',
            fontsize=11, pad=8
        )

        for row_idx, step in enumerate(valid_steps):
            ax = axes[row_idx, col_idx]
            # 找最近的已保存帧
            if step in key_frames:
                frame = key_frames[step]
            else:
                # 找最近的帧
                closest = min(key_frames.keys(), key=lambda k: abs(k - step))
                frame = key_frames[closest]

            ax.imshow(frame)
            ax.axis('off')

            # 行标签（最左列）
            if col_idx == 0:
                ax.set_ylabel(f't = {step}', fontsize=10, rotation=0,
                               labelpad=40, va='center')

    plt.suptitle(
        'Multi-Resolution Comparison  |  Same initial state, emergent patterns at different scales\n'
        'Left: 96px (training size, most stable)  |  Centre: 160px (balanced)  |  Right: 288px (fine-grained, may fragment)',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="用已训练的 Neural CA 对 Pix2Pix 生成的初始状态进行衰变演化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  原始训练尺寸（最稳定）:
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --resolution 96

  中等尺寸:
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --resolution 160

  全分辨率（精细但可能碎片化）:
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --resolution 288

  跑全部三个分辨率并生成对比图:
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --all_resolutions

  自定义步数:
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --all_resolutions --steps 300

  指定 checkpoint 路径:
    python run_ca_from_pix2pix.py --input ca_initial_state.npy --resolution 96 \\
        --checkpoint "D:/Claude_jiajia/Documents/260224-3_generate_neural_ca/checkpoints/final"
        """
    )

    parser.add_argument(
        '--input', type=str, required=True,
        help='.npy 文件路径（由 pix2pix_to_ca.py 生成）'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='输出目录（默认：.npy 文件同目录下的 outputs/ 子文件夹）'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
        help=f'Neural CA 权重路径（不含 .index/.data 扩展名）\n默认: {DEFAULT_CHECKPOINT}'
    )

    # 分辨率参数（二选一）
    res_group = parser.add_mutually_exclusive_group()
    res_group.add_argument(
        '--resolution', type=int, default=None,
        choices=[64, 80, 96, 128, 160, 192, 224, 256, 288, 320],
        help='目标分辨率（像素，正方形边长，含 padding）。默认: 96（训练尺寸）'
    )
    res_group.add_argument(
        '--all_resolutions', action='store_true',
        help='跑全部三个分辨率（96/160/288）并生成对比图'
    )

    parser.add_argument(
        '--steps', type=int, default=DEFAULT_STEPS,
        help=f'演化步数（默认: {DEFAULT_STEPS}，RTX 5070 Ti 下约 1-3 分钟）'
    )
    parser.add_argument(
        '--save_interval', type=int, default=FRAME_SAVE_INTERVAL,
        help=f'帧保存间隔（默认: 每 {FRAME_SAVE_INTERVAL} 步保存一帧）'
    )
    parser.add_argument(
        '--save_hidden', action='store_true',
        help='Save full 16-channel .npy state at key frames (for hidden state analysis). '
             'Files saved as state_t0000.npy etc. in output_dir/hidden_states/.'
    )

    args = parser.parse_args()

    # ── 初始化 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Neural CA 衰变演化  —  Programming Decay")
    print("  基于 Mordvintsev et al. (2020) Growing Neural CA")
    print("=" * 60)

    print_gpu_info()

    # ── 输出目录 ──────────────────────────────────────────────────────────────
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)), "outputs")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"  输出目录: {output_dir}")

    # ── 加载 CA 初始状态 ──────────────────────────────────────────────────────
    print(f"\n  加载初始状态: {args.input}")
    if not os.path.isfile(args.input):
        print(f"  [错误] 找不到文件: {args.input}")
        sys.exit(1)
    initial_state = load_initial_state(args.input)
    print(f"  shape: {initial_state.shape}, dtype: {initial_state.dtype}")
    print(f"  alpha 通道活跃元胞数: {(initial_state[0, :, :, 3] > 0.5).sum():,}")

    # ── 加载 CA 模型权重 ──────────────────────────────────────────────────────
    print(f"\n  加载 CA 权重: {args.checkpoint}")
    ckpt_index = args.checkpoint + ".index"
    if not os.path.isfile(ckpt_index):
        print(f"  [错误] 找不到 checkpoint 文件: {ckpt_index}")
        print(f"  可用的 checkpoint 目录: D:\\Claude_jiajia\\Documents\\260224-1_generate_neural_ca\\checkpoints\\")
        sys.exit(1)

    ca = CAModel(channel_n=CHANNEL_N)
    ca.load_weights(args.checkpoint)
    print(f"  [OK] 权重加载成功")
    print(f"  模型参数数量: {sum(np.prod(w.shape) for w in ca.weights):,}")

    # ── 运行演化 ───────────────────────────────────────────────────────────────
    if args.all_resolutions:
        print(f"\n  模式: 全分辨率对比（96 / 160 / 288）")
        print(f"  演化步数: {args.steps}")
        results = run_all_resolutions(
            ca_model      = ca,
            initial_state = initial_state,
            resolutions   = COMPARE_RESOLUTIONS,
            n_steps       = args.steps,
            output_dir    = output_dir,
            save_hidden   = args.save_hidden,
        )

        # 打印时间对比
        print(f"\n{'─'*50}")
        print(f"  分辨率对比结果：")
        for r in results:
            print(f"    {r['resolution']}px  实际尺寸: {r['actual_size'][0]}×{r['actual_size'][1]}"
                  f"  耗时: {r['total_time']:.1f}s")
        print(f"{'─'*50}")

    else:
        # 单分辨率（默认 96）
        target_res = args.resolution if args.resolution is not None else 96
        print(f"\n  模式: 单分辨率 {target_res}px")
        print(f"  演化步数: {args.steps}")
        result = run_ca_evolution(
            ca_model      = ca,
            initial_state = initial_state,
            resolution    = target_res,
            n_steps       = args.steps,
            output_dir    = output_dir,
            save_interval = args.save_interval,
            save_hidden   = args.save_hidden,
        )

    # ── 完成总结 ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  演化完成！")
    print(f"  输出目录结构:")
    print(f"    {output_dir}/")
    print(f"      frames/res_96/     — 每帧 PNG（如果跑了 96px）")
    print(f"      frames/res_160/    — 每帧 PNG（如果跑了 160px）")
    print(f"      frames/res_288/    — 每帧 PNG（如果跑了 288px）")
    print(f"      summary_res_96.png  — 8帧摘要图")
    print(f"      resolution_comparison.png  — 多分辨率对比（--all_resolutions）")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
