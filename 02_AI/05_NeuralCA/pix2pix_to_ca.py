# -*- coding: utf-8 -*-
"""
pix2pix_to_ca.py
================================================================================
桥接脚本：将 Pix2Pix 生成的功能分区色块图转换为 Neural CA 的初始状态矩阵 T₀

设计逻辑：
    Pix2Pix 输出图中不同颜色代表不同的建筑功能区（回收工厂 / 体验中心）。
    这些功能区对应不同的物理/生物属性（混凝土密度、有机物含量、真菌活性等）。
    通过设置 ch0-3 的初始值，间接控制 Neural CA 中各区域的衰变速率：
        - "难降解区域"（红色/混凝土）→ RGB 设为远离训练目标的浅灰色 → CA 覆盖慢
        - "真菌接种区"（蓝色）→ RGB 接近训练目标 + ch3-15 全为 1.0 → 扩散起点
        - "建筑外部"（黑色背景）→ alpha=0 → CA 不入侵

    注意：ch4-15 是模型学到的内部隐状态，没有人为定义的含义，
    第一步迭代时模型权重就会覆盖这些值。因此控制衰变行为的正确方式
    是通过 ch0-3 的初始值，而不是手动设置 ch4-15 的"衰变速率"。

用法：
    python pix2pix_to_ca.py --input "path/to/generated_image.png"
    python pix2pix_to_ca.py --input "path/to/image.png" --output_dir "path/to/output/"
    python pix2pix_to_ca.py --input_dir "path/to/folder/" --output_dir "path/to/output/"
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import PIL.Image
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# 在无显示环境中使用非交互后端
if not os.environ.get('DISPLAY') and sys.platform != 'win32':
    matplotlib.use('Agg')


# =============================================================================
# 颜色定义与映射
# =============================================================================

# ── Pix2Pix v2 输出色（检测用） ────────────────────────────────────────────────
# 这些是 Pix2Pix 模型输出图中的颜色，用欧几里得距离匹配
DEFINED_COLORS = {
    # 名称           :  (R,   G,   B)
    "red"    : (255,   0,   0),   # 回收工厂 Processing Hall / 体验中心 Exhibition Gallery
    "green"  : (  0, 255,   0),   # 回收工厂 Office
    "blue"   : (  0,   0, 255),   # 回收工厂 Storage / 体验中心 Impact Simulation
    "cyan"   : (  0, 255, 255),   # 回收工厂 Courtyard / 体验中心 Interactive Lab
    "magenta": (255,   0, 255),   # 体验中心 Leisure Area
    "black"  : (  0,   0,   0),   # 建筑外部背景
}

# ── NCA 目标图策略色（归一化 float32，直接用于 CA 初始 RGB） ─────────────────────
# 来自 nca_target_256.png（Dagenham Dock 四色策略图）
# 精确色值来自 PROMPT_nca_strategy_map.md
_DECAY     = (0.886, 0.294, 0.290)   # #E24B4A — Zone C 工业核心，高混凝土，保留衰变
_REUSE     = (0.216, 0.541, 0.866)   # #378ADD — Zone B 仓库改造，材料再用
_DISMANTLE = (0.937, 0.624, 0.153)   # #EF9F27 — Zone D Ford 工厂，拆解回收
_REWILD    = (0.388, 0.600, 0.133)   # #639922 — Zone E+F 河岸湿地，生态修复

# ── Pix2Pix 色 → NCA 策略初始状态 ─────────────────────────────────────────────
#
# 映射逻辑（NCA 已训练在四色策略图上，需把 Pix2Pix 功能区对应到策略色）：
#
#   red   → DECAY    : Processing Hall = 重工业/高混凝土密度 = 保留衰变区
#   green → REWILD   : Office = 有机/低密度 = 靠近河岸的生态修复区
#   blue  → REUSE    : Storage = 材料仓储 = 建筑改造再用区（IS_SEED：作为扩散激活点）
#   cyan  → REWILD   : Courtyard = 开放绿地/互动空间 = 同样归入生态修复区
#   magenta → DISMANTLE : Leisure Area = 低密度/可拆解 = 拆解回收区
#   black → background : alpha=0，CA 不入侵
#
# ca_rgb 使用精确的策略色，使 CA 从正确的颜色语义出发演化。
# All active zones use small random ch4-15 values (Path B: no seed distinction).
COLOR_PROPERTIES = {
    "red"    : {
        "strategy"      : "DECAY",
        "description"   : "DECAY — industrial core / high concrete density",
        "ca_rgb"        : _DECAY,
        "ca_alpha"      : 1.0,
    },
    "green"  : {
        "strategy"      : "REWILD",
        "description"   : "REWILD — organic / green corridor",
        "ca_rgb"        : _REWILD,
        "ca_alpha"      : 1.0,
    },
    "blue"   : {
        "strategy"      : "REUSE",
        "description"   : "REUSE — warehouse / material reuse",
        "ca_rgb"        : _REUSE,
        "ca_alpha"      : 1.0,
    },
    "cyan"   : {
        "strategy"      : "REWILD",
        "description"   : "REWILD — open courtyard / interactive space",
        "ca_rgb"        : _REWILD,
        "ca_alpha"      : 1.0,
    },
    "magenta": {
        "strategy"      : "DISMANTLE",
        "description"   : "DISMANTLE — low density / dismantleable",
        "ca_rgb"        : _DISMANTLE,
        "ca_alpha"      : 1.0,
    },
    "black"  : {
        "strategy"      : "BACKGROUND",
        "description"   : "Building exterior — excluded from CA evolution",
        "ca_rgb"        : (0.0, 0.0, 0.0),
        "ca_alpha"      : 0.0,
    },
}

# ── 可视化颜色（分类图用策略色，直接可用于论文） ─────────────────────────────────
# 与 NCA 目标图的颜色体系一致，让分类图 = Dagenham 四色策略图
VIS_COLORS = {
    "red"    : (226,  75,  74),   # DECAY    #E24B4A
    "green"  : ( 99, 153,  34),   # REWILD   #639922
    "blue"   : ( 55, 138, 221),   # REUSE    #378ADD
    "cyan"   : ( 99, 153,  34),   # REWILD   #639922（同绿，开放空间归入生态修复）
    "magenta": (239, 159,  39),   # DISMANTLE #EF9F27
    "black"  : (  0,   0,   0),   # 背景
}

# 颜色分类时，距离超过此阈值的像素归为背景
CLASSIFICATION_THRESHOLD = 100


# =============================================================================
# 工具函数
# =============================================================================

def load_target_rgb(target_image_path: str) -> tuple:
    """
    （保留备用）从 NCA 目标图像提取平均 RGB 值。

    注意：COLOR_PROPERTIES 里各区域的 ca_rgb 现在已直接硬编码为 Dagenham
    四色策略色（_DECAY / _REUSE / _DISMANTLE / _REWILD），不再依赖此函数
    动态填充颜色。此函数保留供调试或特殊用途。
    """
    if os.path.isfile(target_image_path):
        try:
            img = PIL.Image.open(target_image_path).convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            mask = (arr[..., 0] + arr[..., 1] + arr[..., 2]) > 0.1
            if mask.sum() > 0:
                mean_rgb = arr[mask].mean(axis=0)
                r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
                print(f"  目标图均值 RGB: ({r:.3f}, {g:.3f}, {b:.3f})")
                return (r, g, b)
        except Exception as e:
            print(f"  [警告] 无法读取目标图像: {e}")
    return (0.30, 0.20, 0.15)


def classify_pixels_euclidean(img_array: np.ndarray) -> np.ndarray:
    """
    Fallback: classify each pixel by nearest predefined colour (Euclidean distance).
    Use classify_pixels_kmeans() instead — this is kept for --no_kmeans mode.
    """
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)

    H, W = img_array.shape[:2]
    label_map = np.full((H, W), "black", dtype=object)

    color_names = list(DEFINED_COLORS.keys())
    color_matrix = np.array([DEFINED_COLORS[c] for c in color_names], dtype=np.float32)
    pixels = img_array.reshape(-1, 3).astype(np.float32)
    diff = pixels[:, None, :] - color_matrix[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    min_idx = distances.argmin(axis=1)
    min_dist = distances.min(axis=1)

    valid = min_dist <= CLASSIFICATION_THRESHOLD
    flat_labels = np.where(valid, [color_names[i] for i in min_idx],
                           "black").astype(object)
    return flat_labels.reshape(H, W)


def quantize_colors_kmeans(img_array: np.ndarray,
                            n_clusters: int = 5,
                            black_threshold: int = 30):
    """
    K-Means colour quantization for Pix2Pix output (spatial quantization).

    Background pixels (all RGB < black_threshold) are excluded before clustering
    so that small-area strategy colours are not displaced by the dominant black.

    Args:
        img_array       : (H, W, 3) uint8
        n_clusters      : number of strategy colours to find (excludes background)
        black_threshold : pixels with all channels below this are background

    Returns:
        quantized  : (H, W, 3) uint8 — every pixel replaced by its cluster centroid
        kmeans_labels : (H, W) int   — cluster id per pixel (-1 = background)
        centroids  : (K, 3) float    — cluster centroid colours
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn required for K-Means. Run: pip install scikit-learn")

    H, W = img_array.shape[:2]
    pixels = img_array.reshape(-1, 3).astype(np.float32)

    is_bg_flat = np.all(img_array < black_threshold, axis=-1).reshape(-1)
    fg_pixels = pixels[~is_bg_flat]
    n_fg = fg_pixels.shape[0]

    print(f"  Background pixels : {is_bg_flat.sum():,} ({is_bg_flat.mean()*100:.1f}%)")
    print(f"  Foreground pixels : {n_fg:,} ({(~is_bg_flat).mean()*100:.1f}%)")

    if n_fg == 0:
        print("  [Warning] No foreground pixels — returning original image")
        return img_array, np.full((H, W), -1, dtype=int), np.array([])

    actual_k = min(n_clusters, n_fg)
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=20, max_iter=300)
    fg_labels = kmeans.fit_predict(fg_pixels)
    centroids = kmeans.cluster_centers_

    print(f"  K-Means converged: {actual_k} clusters")
    for i, c in enumerate(centroids):
        n_c = int((fg_labels == i).sum())
        print(f"    Cluster {i}: RGB({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}) "
              f"— {n_c:,} px ({n_c/n_fg*100:.1f}%)")

    full_labels = np.full(H * W, -1, dtype=int)
    full_labels[~is_bg_flat] = fg_labels

    quantized_flat = np.zeros((H * W, 3), dtype=np.uint8)
    for i in range(actual_k):
        mask = full_labels == i
        quantized_flat[mask] = centroids[i].clip(0, 255).astype(np.uint8)
    quantized = quantized_flat.reshape(H, W, 3)

    return quantized, full_labels.reshape(H, W), centroids


def classify_pixels_from_centroids(kmeans_labels: np.ndarray,
                                    centroids: np.ndarray) -> np.ndarray:
    """
    Map each K-Means cluster centroid to the nearest predefined strategy colour.

    Only K distance calculations needed (vs H×W for per-pixel Euclidean).
    Cluster label → strategy colour name → fills label_map.
    """
    H, W = kmeans_labels.shape
    label_map = np.full((H, W), "black", dtype=object)

    color_names = list(DEFINED_COLORS.keys())
    color_matrix = np.array([DEFINED_COLORS[c] for c in color_names], dtype=np.float32)

    centroid_to_color = {}
    for i, centroid in enumerate(centroids):
        distances = np.sqrt(((centroid - color_matrix) ** 2).sum(axis=1))
        min_idx = int(distances.argmin())
        min_dist = float(distances[min_idx])
        centroid_to_color[i] = color_names[min_idx] if min_dist <= CLASSIFICATION_THRESHOLD else "black"
        c = centroid
        print(f"    Cluster {i} RGB({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}) "
              f"-> {centroid_to_color[i]}  (dist={min_dist:.1f})")

    for cluster_id, color_name in centroid_to_color.items():
        label_map[kmeans_labels == cluster_id] = color_name

    return label_map


def build_ca_initial_state(label_map: np.ndarray,
                            add_padding: bool = True,
                            padding_size: int = 16,
                            target_inner_size: int = 64,
                            noise_level: float = 0.15,
                            dropout_rate: float = 0.08) -> np.ndarray:
    """
    Build 16-channel CA initial state from Pix2Pix zone labels.

    Operation order matches training exactly (decay_neural_ca.py):
        1. Assign strategy colours at Pix2Pix resolution (ch0-3)
        2. Resize to 64×64 with NEAREST (same as load_decay_image)
        3. Add ch4-15 small random values at 64×64 scale
        4. Add RGB noise at 64×64 scale (same pixels/noise ratio as training)
        5. Apply dropout at 64×64 scale
        6. Pad to 96×96 (same as TARGET_PADDING=16)

    Args:
        label_map        : (H, W) string array of colour labels
        add_padding      : add 16-px dead-zone border
        padding_size     : border width (should match TARGET_PADDING = 16)
        target_inner_size: resize target before padding (match TARGET_SIZE = 64)
        noise_level      : RGB noise amplitude — match training default (0.15)
        dropout_rate     : fraction of active pixels erased — match training (0.08)

    Returns:
        float32 array, shape (1, target_inner_size+2*p, target_inner_size+2*p, 16)
    """
    H, W = label_map.shape
    state = np.zeros((H, W, 16), dtype=np.float32)

    for color_name, props in COLOR_PROPERTIES.items():
        mask = (label_map == color_name)
        if not mask.any():
            continue
        r, g, b = props["ca_rgb"]
        state[mask, 0] = r
        state[mask, 1] = g
        state[mask, 2] = b
        state[mask, 3] = props["ca_alpha"]
        # ch4-15 will be filled after resize

    # ── Step A: resize ch0-3 to target_inner_size with NEAREST ───────────────
    S = target_inner_size
    if H != S or W != S:
        resized = np.zeros((S, S, 4), dtype=np.float32)
        for c in range(4):
            ch_uint8 = (np.clip(state[:, :, c], 0, 1) * 255).astype(np.uint8)
            pil_ch = PIL.Image.fromarray(ch_uint8, mode='L')
            pil_ch = pil_ch.resize((S, S), PIL.Image.NEAREST)
            resized[:, :, c] = np.array(pil_ch, dtype=np.float32) / 255.0
        state_s = np.zeros((S, S, 16), dtype=np.float32)
        state_s[:, :, :4] = resized
        state = state_s

    # ── Step B: ch4-15 small random values at 64×64 scale ────────────────────
    alive = state[:, :, 3] > 0.5
    n_alive = int(alive.sum())
    if n_alive > 0:
        state[alive, 4:] = np.random.uniform(0, 0.1,
            size=(n_alive, 12)).astype(np.float32)

    # ── Step C: RGB noise at 64×64 scale ─────────────────────────────────────
    if noise_level > 0:
        sh = state.shape[:2]
        rgb_noise = np.random.uniform(-noise_level, noise_level,
                                       (*sh, 3)).astype(np.float32)
        state[:, :, :3] += rgb_noise
        state[:, :, :3] = np.clip(state[:, :, :3], 0, 1)
        state[~alive, :3] = 0.0

    # ── Step D: dropout at 64×64 scale ───────────────────────────────────────
    if dropout_rate > 0:
        drop = (np.random.random(state.shape[:2]) < dropout_rate) & alive
        state[drop, :4] = 0.0

    # ── Step E: pad to 96×96 ─────────────────────────────────────────────────
    if add_padding:
        p = padding_size
        sH, sW = state.shape[:2]
        padded = np.zeros((sH + 2 * p, sW + 2 * p, 16), dtype=np.float32)
        padded[p:p+sH, p:p+sW, :] = state
        state = padded

    return state[None]


def generate_classified_image(label_map: np.ndarray) -> np.ndarray:
    """
    生成清洁化的分类可视化图（用精确颜色替换模糊像素）。

    返回：uint8 RGB 数组，shape (H, W, 3)
    """
    H, W = label_map.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    for color_name, rgb in VIS_COLORS.items():
        mask = (label_map == color_name)
        vis[mask] = rgb
    return vis


def generate_zone_boundary_overlay(classified_img: np.ndarray,
                                    label_map: np.ndarray) -> np.ndarray:
    """
    Highlight zone boundaries on the classified image with a white outline.

    A pixel is a boundary pixel if any of its 4 neighbours belongs to a
    different zone (including background).  Useful for visualising where the
    NCA will have the most work to do (boundary stabilisation).

    Returns: uint8 RGB array, shape (H, W, 3)
    """
    overlay = classified_img.copy()
    H, W = label_map.shape

    for y in range(H):
        for x in range(W):
            zone = label_map[y, x]
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if label_map[ny, nx] != zone:
                        overlay[y, x] = (255, 255, 255)
                        break

    return overlay


def compute_zone_statistics(label_map: np.ndarray) -> dict:
    """
    Count pixels per zone and compute area fractions.

    Returns: dict with per-colour statistics + '_meta' entry.
    """
    H, W = label_map.shape
    total_pixels = H * W
    interior_pixels = int((label_map != "black").sum())

    stats = {}
    for color_name in COLOR_PROPERTIES:
        mask = (label_map == color_name)
        count = int(mask.sum())
        pct_total = count / total_pixels * 100
        pct_interior = count / interior_pixels * 100 if interior_pixels > 0 else 0
        stats[color_name] = {
            "pixel_count"  : count,
            "pct_total"    : pct_total,
            "pct_interior" : pct_interior,
            "strategy"     : COLOR_PROPERTIES[color_name]["strategy"],
            "description"  : COLOR_PROPERTIES[color_name]["description"],
        }

    stats["_meta"] = {
        "total_pixels"    : total_pixels,
        "interior_pixels" : interior_pixels,
        "image_size"      : f"{W}x{H}",
    }
    return stats


# =============================================================================
# 主转换函数
# =============================================================================

def auto_crop_pix2pix(img_array: np.ndarray, crop: str) -> np.ndarray:
    """
    处理 Pix2Pix 并排输出图（input | output 拼接在一起）。

    Pix2Pix 的默认推理脚本通常将 "输入轮廓 | 生成结果" 左右拼接保存为一张图。
    如果图像宽度是高度的两倍，自动识别为并排格式。

    参数：
        img_array : (H, W, 3)，原始读取的图像
        crop      : "auto"    → 如果 W==2H 则取右半（generated），否则用全图
                    "right"   → 强制取右半（generated result）
                    "left"    → 强制取左半（input outline）
                    "full"    → 不裁剪，使用完整图像

    返回：
        裁剪后的 img_array，(H, W', 3)
    """
    H, W = img_array.shape[:2]

    if crop == "full":
        return img_array

    if crop == "left":
        print(f"  [裁剪] 取左半（输入轮廓）: {W//2}x{H}")
        return img_array[:, :W//2, :]

    if crop == "right":
        print(f"  [裁剪] 取右半（生成结果）: {W//2}x{H}")
        return img_array[:, W//2:, :]

    # crop == "auto"
    if W == 2 * H:
        print(f"  [自动裁剪] 检测到并排格式（{W}x{H}），取右半（生成结果）: {W//2}x{H}")
        return img_array[:, W//2:, :]
    else:
        print(f"  [自动裁剪] 非标准并排格式（{W}x{H}），使用完整图像")
        return img_array


def convert_single_image(input_path: str,
                          output_dir: str,
                          decay_image_path: str,
                          add_padding: bool = True,
                          padding_size: int = 16,
                          crop: str = "auto",
                          use_kmeans: bool = True,
                          n_clusters: int = 5,
                          black_threshold: int = 30,
                          target_inner_size: int = 64,
                          noise_level: float = 0.15,
                          dropout_rate: float = 0.08) -> str:
    """
    Convert a single Pix2Pix output image to Neural CA initial state.

    Args:
        input_path        : path to Pix2Pix generated image (PNG)
        output_dir        : output directory
        decay_image_path  : (legacy) not used
        add_padding       : add 16px dead-zone border
        padding_size      : border width (match TARGET_PADDING=16)
        crop              : side-by-side crop mode ("auto"/"right"/"left"/"full")
        use_kmeans        : use K-Means quantization before colour classification
        n_clusters        : K-Means cluster count (strategy colours, excl. background)
        black_threshold   : pixels with all RGB below this treated as background
        target_inner_size : resize target before padding (match TARGET_SIZE=64)
        noise_level       : RGB noise added at 64px scale (match training, default 0.15)
        dropout_rate      : pixel dropout at 64px scale (match training, default 0.08)

    Returns:
        path to saved .npy file
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(input_path).stem

    print(f"\n{'='*60}")
    print(f"  Processing: {Path(input_path).name}")
    print(f"{'='*60}")

    # ── Step 1: load image ─────────────────────────────────────────────────────
    print("[1/7] Loading Pix2Pix output image...")
    img = PIL.Image.open(input_path).convert('RGB')
    img_array = np.array(img, dtype=np.uint8)
    img_array = auto_crop_pix2pix(img_array, crop)
    H, W = img_array.shape[:2]
    print(f"  Image size (after crop): {W}x{H}")

    # ── Step 2: colour quantization ────────────────────────────────────────────
    quantized_img = None
    if use_kmeans:
        print(f"[2/7] K-Means colour quantization (K={n_clusters}, bg_threshold={black_threshold})...")
        quantized_img, kmeans_labels, centroids = quantize_colors_kmeans(
            img_array, n_clusters=n_clusters, black_threshold=black_threshold)
        quantized_path = os.path.join(output_dir, f"{stem}_kmeans_quantized.png")
        PIL.Image.fromarray(quantized_img).save(quantized_path)
        print(f"  Saved: {Path(quantized_path).name}")

        print("[3/7] Mapping cluster centroids to strategy colours...")
        label_map = classify_pixels_from_centroids(kmeans_labels, centroids)
    else:
        print("[2/7] Colour classification (Euclidean distance, K-Means disabled)...")
        label_map = classify_pixels_euclidean(img_array)
        print("[3/7] (K-Means skipped)")

    n_unique = len(np.unique(label_map))
    print(f"  Detected {n_unique} zone types")

    # ── Step 4: generate classification visualisation ──────────────────────────
    print("[4/7] Generating zone classification image...")
    classified_img = generate_classified_image(label_map)
    classified_path = os.path.join(output_dir, f"{stem}_classified_zones.png")
    PIL.Image.fromarray(classified_img).save(classified_path)
    print(f"  Saved: {Path(classified_path).name}")

    zone_overlay = generate_zone_boundary_overlay(classified_img, label_map)
    zone_overlay_path = os.path.join(output_dir, f"{stem}_zone_boundary_overlay.png")
    PIL.Image.fromarray(zone_overlay).save(zone_overlay_path)
    print(f"  Saved: {Path(zone_overlay_path).name}")

    # ── Step 5: strategy colour mapping info ──────────────────────────────────
    print("[5/7] Strategy colour mapping (Dagenham 4-colour):")
    print(f"  DECAY     -> RGB{tuple(int(v*255) for v in _DECAY)}")
    print(f"  REWILD    -> RGB{tuple(int(v*255) for v in _REWILD)}")
    print(f"  REUSE     -> RGB{tuple(int(v*255) for v in _REUSE)}")
    print(f"  DISMANTLE -> RGB{tuple(int(v*255) for v in _DISMANTLE)}")

    # ── Step 6: build 16-channel CA initial state ─────────────────────────────
    print("[6/7] Building 16-channel CA initial state matrix...")
    ca_state = build_ca_initial_state(label_map,
                                       add_padding=add_padding,
                                       padding_size=padding_size,
                                       target_inner_size=target_inner_size,
                                       noise_level=noise_level,
                                       dropout_rate=dropout_rate)
    print(f"  CA state shape: {ca_state.shape}")
    print(f"  dtype: {ca_state.dtype},  range: [{ca_state.min():.3f}, {ca_state.max():.3f}]")

    # ── Step 7: save outputs ───────────────────────────────────────────────────
    print("[7/7] Saving output files...")

    npy_path = os.path.join(output_dir, f"{stem}_ca_initial_state.npy")
    np.save(npy_path, ca_state)
    size_mb = os.path.getsize(npy_path) / 1024 / 1024
    print(f"  Saved [npy]    : {Path(npy_path).name}  ({size_mb:.2f} MB)")

    rgb_preview = ca_state[0, :, :, :3]
    rgb_uint8 = (np.clip(rgb_preview, 0, 1) * 255).astype(np.uint8)
    preview_path = os.path.join(output_dir, f"{stem}_ca_initial_state_preview.png")
    PIL.Image.fromarray(rgb_uint8).save(preview_path)
    print(f"  Saved [preview]: {Path(preview_path).name}")

    stats = compute_zone_statistics(label_map)
    stats_path = os.path.join(output_dir, f"{stem}_zone_statistics.txt")
    _write_statistics(stats, stats_path, input_path)
    print(f"  Saved [stats]  : {Path(stats_path).name}")

    _save_summary_figure(img_array, classified_img, rgb_uint8, zone_overlay,
                          stats, output_dir, stem, quantized_img=quantized_img)
    print(f"  Saved [summary]: {stem}_summary.png")

    _print_statistics(stats, ca_state)

    return npy_path


def _write_statistics(stats: dict, output_path: str, input_path: str):
    """Write zone statistics to a text file."""
    meta = stats["_meta"]
    lines = [
        "=" * 60,
        "  Pix2Pix -> Neural CA  Zone Statistics",
        f"  Input: {input_path}",
        f"  Image size: {meta['image_size']}",
        f"  Total pixels: {meta['total_pixels']:,}",
        f"  Interior pixels: {meta['interior_pixels']:,}",
        "=" * 60,
        "",
        f"{'Zone':<22} {'Pixels':>10} {'% Total':>9} {'% Interior':>11}",
        "-" * 58,
    ]

    total_active = 0
    for color_name, s in stats.items():
        if color_name in ("_meta", "black"):
            continue
        strategy = s.get("strategy", color_name.upper())
        lines.append(
            f"{strategy:<22} "
            f"{s['pixel_count']:>10,} "
            f"{s['pct_total']:>8.1f}% "
            f"{s['pct_interior']:>10.1f}%"
        )
        total_active += s["pixel_count"]

    bg = stats.get("black", {})
    lines.append(
        f"{'BACKGROUND':<22} "
        f"{bg.get('pixel_count', 0):>10,} "
        f"{bg.get('pct_total', 0):>8.1f}%"
    )
    lines += [
        "",
        "-" * 58,
        f"Total active cells (interior): {total_active:,}",
        "=" * 60,
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _print_statistics(stats: dict, ca_state: np.ndarray):
    """Print zone statistics to console."""
    print(f"\n{'─'*50}")
    print(f"  Zone area statistics (interior only)")
    print(f"{'─'*50}")

    total_active = 0
    for color_name, s in stats.items():
        if color_name in ("_meta", "black"):
            continue
        strategy = s.get("strategy", color_name.upper())
        print(f"  {strategy:<12} {s['pct_interior']:>5.1f}%  ({s['pixel_count']:>6,} px)")
        total_active += s["pixel_count"]

    bg = stats.get("black", {})
    print(f"  {'BACKGROUND':<12} {bg.get('pct_total', 0):>5.1f}%  ({bg.get('pixel_count', 0):>6,} px)")
    print(f"{'─'*50}")
    print(f"  Total active cells : {total_active:,}")
    print(f"  CA state shape     : {ca_state.shape}")
    print(f"{'─'*50}")


def _save_summary_figure(original: np.ndarray,
                           classified: np.ndarray,
                           ca_preview: np.ndarray,
                           zone_overlay: np.ndarray,
                           stats: dict,
                           output_dir: str,
                           stem: str,
                           quantized_img: np.ndarray = None):
    """
    Save summary figure.
    Without K-Means: 2x2 grid.
    With K-Means: 2x3 grid (adds quantized image as middle column).
    """
    if quantized_img is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Pix2Pix Output (original)', fontsize=11)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(quantized_img)
        axes[0, 1].set_title('K-Means Quantized (colour clusters)', fontsize=11)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(classified)
        axes[0, 2].set_title('Strategy Zone Classification', fontsize=11)
        axes[0, 2].axis('off')

        axes[1, 0].imshow(ca_preview)
        axes[1, 0].set_title('CA Initial State RGB Preview (ch0-2)', fontsize=11)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(zone_overlay)
        axes[1, 1].set_title('Zone Boundaries (white = NCA evolution boundary)', fontsize=11)
        axes[1, 1].axis('off')

        # Legend panel
        axes[1, 2].axis('off')
        meta = stats["_meta"]
        interior = meta["interior_pixels"]
        legend_lines = []
        for color_name, s in stats.items():
            if color_name in ("_meta", "black"):
                continue
            pct = s["pixel_count"] / interior * 100 if interior > 0 else 0
            legend_lines.append(f"{s['strategy']}: {pct:.1f}%")
        legend_lines.append(f"BACKGROUND: {stats['black']['pct_total']:.1f}%")
        axes[1, 2].text(0.1, 0.5, "\n".join(legend_lines),
                        transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='center',
                        fontfamily='monospace')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Pix2Pix Output (original)', fontsize=11)
        axes[0, 0].axis('off')
        axes[0, 1].imshow(classified)
        axes[0, 1].set_title('Strategy Zone Classification', fontsize=11)
        axes[0, 1].axis('off')
        axes[1, 0].imshow(ca_preview)
        axes[1, 0].set_title('CA Initial State RGB Preview (ch0-2)', fontsize=11)
        axes[1, 0].axis('off')
        axes[1, 1].imshow(zone_overlay)
        axes[1, 1].set_title('Zone Boundaries', fontsize=11)
        axes[1, 1].axis('off')

        meta = stats["_meta"]
        interior = meta["interior_pixels"]
        legend_items = []
        for color_name, s in stats.items():
            if color_name in ("_meta", "black"):
                continue
            pct = s["pixel_count"] / interior * 100 if interior > 0 else 0
            legend_items.append(f"{s.get('strategy', color_name)}: {pct:.1f}%")

    plt.suptitle('Pix2Pix -> Neural CA  Bridge Conversion', fontsize=12)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, f"{stem}_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="将 Pix2Pix 功能分区图转换为 Neural CA 初始状态矩阵",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  单张图像：
    python pix2pix_to_ca.py --input "D:/output/generated.png"
    python pix2pix_to_ca.py --input "generated.png" --output_dir "D:/ca_inputs/"

  批量处理目录：
    python pix2pix_to_ca.py --input_dir "D:/pix2pix/results/" --output_dir "D:/ca_inputs/"

  不加 padding（如果 CA 模型已经处理 padding）：
    python pix2pix_to_ca.py --input "generated.png" --no_padding

  Pix2Pix 输出是"输入|结果"并排图时，指定裁剪方式：
    python pix2pix_to_ca.py --input "result.png" --crop right
    python pix2pix_to_ca.py --input "result.png" --crop auto   # 默认：自动检测
        """
    )

    # 输入方式（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', type=str,
        help='单张 Pix2Pix 输出图的路径（PNG）'
    )
    input_group.add_argument(
        '--input_dir', type=str,
        help='包含多张 Pix2Pix 输出图的目录（批量处理）'
    )

    parser.add_argument(
        '--output_dir', type=str,
        default=None,
        help='输出目录（默认：输入图同目录下的 ca_outputs/ 子文件夹）'
    )
    parser.add_argument(
        '--decay_image', type=str,
        default=r"D:\Claude_jiajia\Documents\decay_image.png",
        help='(Legacy) CA training target image path — no longer used for colour extraction'
    )
    parser.add_argument(
        '--no_padding', action='store_true',
        help='Skip 16px dead-zone padding (default: add padding)'
    )
    parser.add_argument(
        '--padding_size', type=int, default=16,
        help='padding 宽度（默认：16px，与训练时 TARGET_PADDING 一致）'
    )
    parser.add_argument(
        '--crop', type=str, default='auto',
        choices=['auto', 'right', 'left', 'full'],
        help=(
            '处理 Pix2Pix 并排输出（input|output 拼接图）的裁剪方式：\n'
            '  auto  → 自动检测：宽度=2*高度时取右半，否则用全图（默认）\n'
            '  right → 强制取右半（生成结果）\n'
            '  left  → 强制取左半（输入轮廓）\n'
            '  full  → 不裁剪，使用完整图像'
        )
    )
    parser.add_argument(
        '--n_clusters', type=int, default=5,
        help='K-Means cluster count (= number of strategy colours, excl. background). '
             'Default 5 (red/green/blue/cyan/magenta). Background auto-excluded.'
    )
    parser.add_argument(
        '--black_threshold', type=int, default=30,
        help='Background detection threshold: pixels with all RGB < this value are background. '
             'Default 30. Increase to 50 if small strategy areas are being lost.'
    )
    parser.add_argument(
        '--no_kmeans', action='store_true',
        help='Skip K-Means quantization; use per-pixel Euclidean distance matching (not recommended).'
    )
    parser.add_argument(
        '--target_inner_size', type=int, default=64,
        help='Resize to this size before padding (must match TARGET_SIZE=64 in training). Default 64.'
    )
    parser.add_argument(
        '--noise_level', type=float, default=0.15,
        help='RGB noise added at 64px scale — must match training default (default 0.15).'
    )
    parser.add_argument(
        '--dropout_rate', type=float, default=0.08,
        help='Pixel dropout at 64px scale — must match training default (default 0.08).'
    )

    args = parser.parse_args()

    add_padding = not args.no_padding

    # ── 单张模式 ─────────────────────────────────────────────────────────────────
    if args.input:
        if not os.path.isfile(args.input):
            print(f"[错误] 找不到输入文件: {args.input}")
            sys.exit(1)

        # 默认输出目录
        if args.output_dir is None:
            output_dir = os.path.join(os.path.dirname(args.input), "ca_outputs")
        else:
            output_dir = args.output_dir

        npy_path = convert_single_image(
            input_path      = args.input,
            output_dir      = output_dir,
            decay_image_path= args.decay_image,
            add_padding     = add_padding,
            padding_size    = args.padding_size,
            crop            = args.crop,
            use_kmeans        = not args.no_kmeans,
            n_clusters        = args.n_clusters,
            black_threshold   = args.black_threshold,
            target_inner_size = args.target_inner_size,
            noise_level       = args.noise_level,
            dropout_rate      = args.dropout_rate,
        )
        print(f"\n  Output .npy: {npy_path}")
        print(f"  Pass to run_ca_from_pix2pix.py --input \"{npy_path}\"")

    # ── 批量模式 ─────────────────────────────────────────────────────────────────
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"[错误] 找不到输入目录: {args.input_dir}")
            sys.exit(1)

        # 收集所有 PNG 文件
        png_files = sorted(Path(args.input_dir).glob("*.png"))
        if not png_files:
            print(f"[错误] 目录中没有 PNG 文件: {args.input_dir}")
            sys.exit(1)

        if args.output_dir is None:
            output_dir = os.path.join(args.input_dir, "ca_outputs")
        else:
            output_dir = args.output_dir

        print(f"  Found {len(png_files)} images, processing...")
        npy_paths = []
        for i, png_path in enumerate(png_files, 1):
            print(f"\n[{i}/{len(png_files)}]", end="")
            npy_path = convert_single_image(
                input_path      = str(png_path),
                output_dir      = output_dir,
                decay_image_path= args.decay_image,
                add_padding     = add_padding,
                padding_size    = args.padding_size,
                crop            = args.crop,
                use_kmeans      = not args.no_kmeans,
                n_clusters      = args.n_clusters,
                black_threshold = args.black_threshold,
            )
            npy_paths.append(npy_path)

        print(f"\n{'='*60}")
        print(f"  Batch complete! Converted {len(npy_paths)} images.")
        print(f"  Output dir: {output_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
