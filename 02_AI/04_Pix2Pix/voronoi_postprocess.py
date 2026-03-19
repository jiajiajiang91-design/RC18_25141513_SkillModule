"""
voronoi_postprocess.py — Pix2Pix → Voronoi (v5 — 固定参考 mask)

核心改动：所有 epoch 共用同一个建筑 mask（从最优 epoch 提取），
确保每张输出的建筑外轮廓形状完全一致。

用法：
  # 单张
  python voronoi_postprocess.py --input gen_360.png --ref_mask gen_360.png --output out.png

  # batch_voronoi.py 会自动传 --ref_mask
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import cv2
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay, cKDTree

FUNCTIONAL_COLORS = {
    "Processing/Exhibition": (255,   0,   0),
    "Storage/Simulation":    (  0,   0, 255),
    "Office/Lab":            (  0, 255,   0),
    "Courtyard/Leisure":     (  0, 255, 255),
    "Leisure/Magenta":       (255,   0, 255),
}
FUNC_COLOR_LIST = list(FUNCTIONAL_COLORS.values())
FUNC_COLOR_ARR  = np.array(FUNC_COLOR_LIST, dtype=np.float32)

DOT_SPACING   = 6
DOT_RADIUS    = 1
COLOR_THRESH  = 20
MIN_COMP_PX   = 80
TARGET_SEEDS  = 25


def make_mask_from_outline(outline_path, size=(256, 256)):
    """
    从白色轮廓线图（黑底白线）提取建筑内部填充 mask。
    步骤：二值化 → 膨胀闭合缝隙 → flood fill 外部 → 取反得内部
    返回 uint8 数组，255=建筑内部，0=外部
    """
    img = Image.open(outline_path).convert("L").resize(size, Image.NEAREST)
    arr = np.array(img)
    binary = (arr > 80).astype(np.uint8) * 255

    # 膨胀白线，闭合不完全封闭的缝隙
    kernel  = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # 从四角 flood fill 标记外部为 128
    h, w  = dilated.shape
    flood = dilated.copy()
    for corner in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff, corner, 128)

    # 内部 = 既不是白线(255)也不是外部(128)
    interior = ((flood != 255) & (flood != 128)).astype(np.uint8) * 255

    # 轻微膨胀，将轮廓线内侧也纳入
    interior = cv2.dilate(interior, np.ones((3, 3), np.uint8), iterations=1)
    return interior


def make_mask_from_gen(gen_arr):
    """从生成图提取实心建筑 mask：闭合→最大连通→填空洞"""
    gray = gen_arr.max(axis=2).astype(np.uint8)
    raw = (gray > COLOR_THRESH).astype(np.uint8) * 255
    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=3)
    n_comp, comp_map, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n_comp > 1:
        largest_id = 1 + np.argmax(stats[1:, 4])
        mask = ((comp_map == largest_id).astype(np.uint8)) * 255
    else:
        mask = closed
    flood = mask.copy()
    h, w = flood.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 128)
    holes = ((flood != 255) & (flood != 128)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, holes)
    return mask


def nearest_func_color(center):
    dists = np.linalg.norm(FUNC_COLOR_ARR - center, axis=1)
    return FUNC_COLOR_LIST[int(dists.argmin())]


def extract_seeds(gen_arr, mask, target_n=TARGET_SEEDS):
    H, W = gen_arr.shape[:2]
    gray_max = gen_arr.max(axis=2)
    non_black = (gray_max > COLOR_THRESH) & (mask > 0)
    nb_pixels = gen_arr[non_black]
    if len(nb_pixels) < 20:
        return _default_seeds(H, W, mask)
    n_clusters = min(4, max(2, len(nb_pixels) // 50))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(nb_pixels)
    ys, xs = np.where(non_black)
    all_seeds = []
    for cid in range(n_clusters):
        sel = labels == cid
        if sel.sum() < MIN_COMP_PX:
            continue
        center = km.cluster_centers_[cid]
        color = nearest_func_color(center)
        cluster_img = np.zeros((H, W), dtype=np.uint8)
        cluster_img[ys[sel], xs[sel]] = 255
        cluster_img = cv2.morphologyEx(cluster_img, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)
        n_comp, comp_map = cv2.connectedComponents(cluster_img)
        for comp_id in range(1, n_comp):
            comp_mask = comp_map == comp_id
            area = comp_mask.sum()
            if area < MIN_COMP_PX:
                continue
            cy, cx = np.where(comp_mask)
            all_seeds.append((area, int(cy.mean()), int(cx.mean()), color))
    if len(all_seeds) < 2:
        return _default_seeds(H, W, mask)
    all_seeds.sort(key=lambda s: s[0], reverse=True)
    return [(s[1], s[2], s[3]) for s in all_seeds[:target_n]]


def _default_seeds(H, W, mask):
    seeds = []
    for y in range(40, H - 40, 60):
        for x in range(40, W - 40, 60):
            if mask[y, x] > 0:
                seeds.append((y, x, FUNC_COLOR_LIST[len(seeds) % len(FUNC_COLOR_LIST)]))
    if len(seeds) < 2:
        seeds = [(64, 64, (255, 0, 0)), (192, 192, (0, 0, 255))]
    return seeds


def _build_dot_mask(h, w, spacing, radius):
    dot_mask = np.zeros((h, w), dtype=bool)
    ky, kx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    circle = ky ** 2 + kx ** 2 <= radius ** 2
    for cy in range(radius, h - radius, spacing):
        for cx in range(radius, w - radius, spacing):
            dot_mask[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1] |= circle
    return dot_mask


def process(input_path, ref_mask_path, output_path, dot_spacing=DOT_SPACING, dot_radius=DOT_RADIUS,
            outline_path=None):
    """
    Args:
        input_path:    当前 epoch 的生成图（用于提取颜色/种子点）
        ref_mask_path: 参考 gen 图路径（fallback，当 outline_path 未提供时使用）
        outline_path:  白色轮廓线图路径（优先级最高，用于提取精确建筑 mask）
    """
    H, W = 256, 256
    gen_arr = np.array(Image.open(input_path).convert("RGB").resize((W, H), Image.NEAREST))

    # outline_path 优先；其次 ref_mask_path；最后从当前图自身提取
    if outline_path and os.path.exists(outline_path):
        mask = make_mask_from_outline(outline_path, size=(W, H))
    elif ref_mask_path and os.path.exists(ref_mask_path):
        ref_arr = np.array(Image.open(ref_mask_path).convert("RGB").resize((W, H), Image.NEAREST))
        mask = make_mask_from_gen(ref_arr)
    else:
        mask = make_mask_from_gen(gen_arr)

    mask_bool = mask > 0

    # 种子点从当前图提取（颜色分布每个 epoch 不同）
    seeds = extract_seeds(gen_arr, mask)
    seed_yx = np.array([(s[0], s[1]) for s in seeds], dtype=np.float32)
    seed_colors = [s[2] for s in seeds]

    # Voronoi 区域分配
    tree = cKDTree(seed_yx)
    yy, xx = np.mgrid[0:H, 0:W]
    _, near_idx = tree.query(np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32))
    cell_map = near_idx.reshape(H, W)

    output = np.zeros((H, W, 3), dtype=np.uint8)

    # 1) 彩色点阵 — 覆盖整个 mask 区域
    dot_mask = _build_dot_mask(H, W, dot_spacing, dot_radius)
    for i, color in enumerate(seed_colors):
        output[dot_mask & mask_bool & (cell_map == i)] = color

    # 2) 红色区域 Delaunay 连线 — 每个连通岛独立处理
    RED = (255, 0, 0)

    # 2a) 构建红色区域整体 mask
    red_cell_mask = np.zeros((H, W), dtype=np.uint8)
    for i, color in enumerate(seed_colors):
        if color == RED:
            red_cell_mask[cell_map == i] = 255
    red_cell_mask[~mask_bool] = 0

    # 2b) 找各独立连通红色岛
    n_islands, island_map = cv2.connectedComponents(red_cell_mask)

    # 2c) 每个岛单独做 Delaunay（随机采样，产生不规则三角形）
    PX_PER_POINT = 60   # 每隔多少像素面积放一个随机点（越小点越密）
    MIN_AREA     = 30   # 忽略面积小于此值的岛

    rng = np.random.default_rng(42)   # 固定种子，结果可复现

    pil_out = Image.fromarray(output)
    draw = ImageDraw.Draw(pil_out)

    for island_id in range(1, n_islands):
        island_bin = (island_map == island_id).astype(np.uint8)
        area = island_bin.sum()
        if area < MIN_AREA:
            continue

        ys, xs = np.where(island_bin)

        # 随机采样岛内的点（数量由面积决定，至少 3 个）
        n_pts = max(3, area // PX_PER_POINT)
        idx   = rng.choice(len(ys), size=min(n_pts, len(ys)), replace=False)
        rand_pts = [(int(ys[i]), int(xs[i])) for i in idx]

        # 补充边界轮廓点，让边缘三角形贴合岛的形状
        contours_isl, _ = cv2.findContours(island_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_isl:
            for p in c[::max(1, len(c) // 12)]:
                rand_pts.append((int(p[0][1]), int(p[0][0])))

        # 去重
        all_pts = list({(int(y), int(x)) for y, x in rand_pts})

        if len(all_pts) < 3:
            if len(all_pts) == 2:
                p1, p2 = all_pts
                draw.line([(p1[1], p1[0]), (p2[1], p2[0])], fill=(255, 255, 255), width=1)
            continue

        try:
            pts = np.array(all_pts, dtype=np.float32)
            tri = Delaunay(pts)
            # 随机保留 50% 的三角形（点数不变，只减少绘制数量）
            keep = rng.random(len(tri.simplices)) < 0.5
            for simplex, draw_it in zip(tri.simplices, keep):
                if not draw_it:
                    continue
                for j in range(3):
                    p1 = pts[simplex[j]]
                    p2 = pts[simplex[(j + 1) % 3]]
                    # 线段中点也必须在岛内，防止画到岛外
                    my, mx = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
                    if 0 <= my < H and 0 <= mx < W and island_bin[my, mx]:
                        draw.line(
                            [(int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0]))],
                            fill=(255, 255, 255), width=1
                        )
        except Exception:
            pass

    # 3) 建筑外轮廓细线
    contours_list, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_list:
        pts_list = [(int(p[0][0]), int(p[0][1])) for p in contour]
        if len(pts_list) > 2:
            pts_list.append(pts_list[0])
            draw.line(pts_list, fill=(255, 255, 255), width=1)

    output = np.array(pil_out)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(output).save(output_path)
    print(f"  saved: {output_path}  (seeds: {len(seeds)})")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--ref_mask", default=None, help="参考图路径（固定mask来源，通常是最优epoch的gen图）")
    p.add_argument("--output", required=True)
    p.add_argument("--dot_spacing", type=int, default=DOT_SPACING)
    p.add_argument("--dot_radius", type=int, default=DOT_RADIUS)
    a = p.parse_args()
    process(a.input, a.ref_mask, a.output, a.dot_spacing, a.dot_radius)
