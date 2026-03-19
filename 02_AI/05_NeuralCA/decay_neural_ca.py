# -*- coding: utf-8 -*-
"""
decay_neural_ca.py
===============================================================================
Neural Cellular Automata for Architectural Decay Simulation

Based on:  Mordvintsev et al. (2020) "Growing Neural Cellular Automata"
           https://distill.pub/2020/growing-ca
Theory:    Gilpin (2019) "Cellular Automata as Convolutional Neural Networks"

Thesis context — Simulation as Inquiry into Building Decay
───────────────────────────────────────────────────────────
 Original paper  → CA learns to GROW an emoji from a single seed pixel
 This adaptation → CA learns to SPREAD decay from crack initiation sites

 Why Neural CA for architectural decay?
  - Rules are LEARNED from real decay images, not hand-crafted
    (directly answers the "rules problem" of traditional CA — Holland / Batty)
  - Local neighbourhood rules → complex spatial patterns (emergence)
  - Multiple runs from different seeds = Simulation as Inquiry,
    exploring possible decay trajectories, not predicting one outcome
  - CA+CNN mathematical equivalence (Gilpin 2019) provides
    the theoretical justification for this approach

Channel semantics (16 channels per cell):
  ch 0-2   RGB decay texture      (visible state — the decay pattern)
  ch 3     alpha / living mask    (which cells are "active")
  ch 4-15  hidden decay state     (neighbourhood memory, latent dynamics)

What is UNCHANGED from the original paper (and why):
  CAModel architecture, SamplePool, training loop
  → These form the theoretical core; modifying them would break the
    direct link to Mordvintsev 2020 and Gilpin 2019.

What is ADAPTED for architectural decay:
  1. load_decay_image()   — local crack/decay photo replaces emoji URL
  2. make_decay_seed()    — multiple crack nucleation points
  3. Visualisation        — decay-appropriate output, matplotlib
  4. Default experiment   — "Persistent" (decay does not self-repair)
  5. Removed              — Colab magic, TF.js/WebGL, moviepy dependency

Usage (Anaconda):
  conda env create -f environment.yml
  conda activate neural_ca
  python decay_neural_ca.py
  python decay_neural_ca.py --quick     # fast test run (~5 min)
===============================================================================
"""

import os
import sys
import io
import time
import argparse
import PIL.Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tqdm

# ── TensorFlow setup ─────────────────────────────────────────────────────────
# Suppress TF info messages (keep warnings and errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

# Use non-interactive backend when no display is available (e.g. SSH / server)
if not os.environ.get('DISPLAY') and sys.platform != 'win32':
    matplotlib.use('Agg')


# =============================================================================
# PARAMETERS — change these to match your project
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
TARGET_IMAGE_PATH = r"D:\Claude_jiajia\pix2pix_to_ca\nca_target_256.png"
OUTPUT_DIR        = r"D:\Claude_jiajia\Documents\generate_neural_ca"

# ── CA architecture (unchanged from paper) ────────────────────────────────────
CHANNEL_N      = 16    # state channels per cell
TARGET_PADDING = 16    # border padding around target image
TARGET_SIZE    = 64    # target is resized to 64x64 (paper used 40)
BATCH_SIZE     = 8
POOL_SIZE      = 1024
CELL_FIRE_RATE = 0.5   # stochastic: 50% of cells update each step

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_STEPS = 16000    # full training (GPU: ~1-2 hr, CPU: ~6-10 hr)
LR_INIT     = 2e-3     # initial learning rate (drops 10x at step 2000)

# ── Decay-specific ────────────────────────────────────────────────────────────
N_SEED_POINTS = 3      # crack nucleation sites (1 = same as paper)

# Experiment type:
#   "Growing"      best for studying how decay spreads from seeds
#   "Persistent"   best for studying what stable pattern the CA maintains
#   "Regenerating" NOT suitable for decay (self-repair ≠ real decay)
EXPERIMENT_TYPE  = "Persistent"
EXPERIMENT_MAP   = {"Growing": 0, "Persistent": 1, "Regenerating": 2}
EXPERIMENT_N     = EXPERIMENT_MAP[EXPERIMENT_TYPE]
USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
DAMAGE_N         = [0, 0, 3][EXPERIMENT_N]

# ── Output subdirectories ─────────────────────────────────────────────────────
DIR_INPUT       = os.path.join(OUTPUT_DIR, "input")
DIR_CHECKPOINTS = os.path.join(OUTPUT_DIR, "checkpoints")
DIR_TRAINING    = os.path.join(OUTPUT_DIR, "training")
DIR_RESULTS     = os.path.join(OUTPUT_DIR, "results")


# =============================================================================
# UTILITY FUNCTIONS  (unchanged from original paper)
# =============================================================================

def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
            fmt = 'jpeg'
        with open(f, 'wb') as fh:
            np2pil(a).save(fh, fmt, quality=95)
    else:
        np2pil(a).save(f, fmt, quality=95)


def tile2d(a, w=None):
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w - len(a)) % w
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), 'constant')
    h = len(a) // w
    a = a.reshape([h, w] + list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
    return a


def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


# ── RGBA / RGB helpers (unchanged from paper) ────────────────────────────────

def to_rgba(x):
    return x[..., :4]

def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
    """Convert pre-multiplied RGBA to displayable RGB (white background)."""
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb

def get_living_mask(x):
    """A cell is alive if any cell in its 3x3 neighbourhood has alpha > 0.1."""
    alpha = x[:, :, :, 3:4]
    return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1


# =============================================================================
# DECAY IMAGE LOADING & SEED  (adapted for architectural decay)
# =============================================================================

def load_decay_image(path, size=TARGET_SIZE):
    """
    Load a local building decay / crack image as the CA training target.

    The image represents the spatial decay PATTERN the CA must learn to
    reproduce — e.g. a crack network, rust distribution, spalling texture.

    IMPORTANT: If the image has a transparent background (RGBA PNG), the
    transparency is PRESERVED.  This is ideal because:
      - alpha > 0  → crack/decay pixels (alive cells)
      - alpha = 0  → background (dead cells)
    This matches exactly how the original paper handles emoji targets.

    Non-square images are padded to square with transparent pixels before
    resizing, to avoid distorting the crack pattern.

    Returns: float32 array, shape (size, size, 4)
    """
    img = PIL.Image.open(path)
    print(f"  Original size: {img.size}, mode: {img.mode}")

    # Convert to RGBA — keep transparency if present
    img = img.convert('RGBA')

    # Pad non-square images to square (fill with transparent black)
    # This prevents distortion when resizing to size×size
    w_img, h_img = img.size
    if w_img != h_img:
        side = max(w_img, h_img)
        padded = PIL.Image.new('RGBA', (side, side), (0, 0, 0, 0))
        offset_x = (side - w_img) // 2
        offset_y = (side - h_img) // 2
        padded.paste(img, (offset_x, offset_y))
        img = padded
        print(f"  Padded to square: {side}x{side} (transparent fill)")

    img = img.resize((size, size), PIL.Image.NEAREST)  # NEAREST preserves sharp zone boundaries
    img = np.float32(img) / 255.0

    # Pre-multiply RGB by alpha (paper convention)
    # Where alpha=0 (transparent), RGB becomes 0 → dead cell
    # Where alpha=1 (crack), RGB retains crack colour
    img[..., :3] *= img[..., 3:]
    return img


def make_synthetic_crack_target(size=TARGET_SIZE):
    """Fallback: synthetic crack pattern for testing when no image is available."""
    rng = np.random.default_rng(42)
    img = np.ones((size, size, 3), np.float32) * 0.85
    # Horizontal crack
    cy = size // 2
    for x in range(size):
        y = cy + int(4 * np.sin(x * 0.15))
        y = np.clip(y, 0, size - 1)
        for dy in range(-1, 2):
            yy = np.clip(y + dy, 0, size - 1)
            img[yy, x] = [0.08, 0.06, 0.04]
    # Random speckle texture
    img += rng.uniform(-0.06, 0.06, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    alpha = np.ones((*img.shape[:2], 1), np.float32)
    return np.concatenate([img, alpha], axis=-1)


def make_decay_seed(h, w, n_seeds=N_SEED_POINTS, n_batch=1):
    """
    Create initial CA state with crack initiation points.

    Unlike the original (single centre pixel), architectural decay can
    begin at multiple NUCLEATION SITES — stress concentrations,
    pre-existing micro-cracks, or material discontinuities.

    Returns: float32 array, shape (n_batch, h, w, CHANNEL_N)
    """
    x = np.zeros([n_batch, h, w, CHANNEL_N], np.float32)
    margin = TARGET_PADDING + 4   # keep seeds within the actual image area

    if n_seeds == 1:
        x[:, h // 2, w // 2, 3:] = 1.0
    else:
        for _ in range(n_seeds):
            r = np.random.randint(margin, h - margin)
            c = np.random.randint(margin, w - margin)
            x[:, r, c, 3:] = 1.0

    return x


def make_road_seed_from_target(target_padded, n_batch=1):
    """
    Create CA seed state where road/infrastructure pixels are the nucleation sites.

    In the Dagenham strategy map:
      dark gray (~#2C2C2A) = Remediation Walk road skeleton
      These pixels represent the 'spine' from which strategies spread outward.

    Detection: pixels where alpha > 0.5 (inside image) AND all RGB channels
    are dark (< 0.30) AND approximately neutral grey (max-min < 0.15).
    This robustly captures the road colour without hardcoding an exact value.

    Returns: float32 array, shape (n_batch, H, W, CHANNEL_N)
    """
    if hasattr(target_padded, 'numpy'):
        arr = target_padded.numpy()
    else:
        arr = np.asarray(target_padded)

    H, W = arr.shape[:2]
    x = np.zeros([n_batch, H, W, CHANNEL_N], np.float32)

    rgb   = arr[:, :, :3]            # (H, W, 3)
    alpha = arr[:, :, 3] if arr.shape[2] > 3 else np.ones((H, W), np.float32)

    r_max = rgb.max(axis=-1)          # (H, W)
    r_min = rgb.min(axis=-1)

    is_road = (
        (alpha > 0.5)       &         # inside the actual image (not padding)
        (r_max  < 0.30)     &         # dark overall
        (r_max - r_min < 0.15)        # approximately grey (R ≈ G ≈ B)
    )

    n_road = int(is_road.sum())
    if n_road == 0:
        print(f"  [Warning] No road pixels detected — falling back to {N_SEED_POINTS} random seeds.")
        return make_decay_seed(H, W, n_seeds=N_SEED_POINTS, n_batch=n_batch)

    print(f"  Road seed pixels : {n_road}  ({n_road / (H * W) * 100:.1f}% of grid)")

    # Activate road pixels: ch3 (alpha/living mask) = 1.0, ch4-15 (hidden) = 1.0
    # This matches make_decay_seed() convention: x[:, r, c, 3:] = 1.0
    x[:, is_road, 3:] = 1.0

    return x


def make_noisy_seed_from_target(target_padded, noise_level=0.15, dropout_rate=0.08,
                                 blur_boundaries=True, n_batch=1):
    """
    Generate a noisy initial state from the target image (Path B training mode).

    Simulates 'rough zoning from Pix2Pix': colour deviation, blurred boundaries,
    and partial dropout. The NCA learns to evolve this imperfect initial zoning
    into the clean target — i.e. temporal stabilisation of spatial strategy.

    Noise types (randomly combined each call for training diversity):
      1. RGB noise        : random per-pixel colour shift (simulates Pix2Pix colour bias)
      2. Region dropout   : erase ~20% of active pixels to background
      3. Boundary blur    : Gaussian blur at zone boundaries (simulates fuzzy edges)
      4. Global brightness: small overall brightness/darkness shift

    Returns: float32 array, shape (n_batch, H, W, CHANNEL_N)
    """
    try:
        from scipy.ndimage import gaussian_filter
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    if hasattr(target_padded, 'numpy'):
        arr = target_padded.numpy()
    else:
        arr = np.asarray(target_padded)

    H, W = arr.shape[:2]
    seeds = np.zeros([n_batch, H, W, CHANNEL_N], np.float32)

    for b in range(n_batch):
        seed = np.zeros((H, W, CHANNEL_N), np.float32)
        seed[:, :, :4] = arr[:, :, :4].copy()

        alive = arr[:, :, 3] > 0.5

        # ── Noise 1: RGB colour shift ──────────────────────────────────────────
        if noise_level > 0:
            rgb_noise = np.random.uniform(-noise_level, noise_level,
                                          (H, W, 3)).astype(np.float32)
            seed[:, :, :3] += rgb_noise
            seed[:, :, :3] = np.clip(seed[:, :, :3], 0, 1)
            seed[~alive, :3] = 0.0

        # ── Noise 2: Region dropout ────────────────────────────────────────────
        if dropout_rate > 0:
            dropout_mask = (np.random.random((H, W)) < dropout_rate) & alive
            seed[dropout_mask, :4] = 0.0

        # ── Noise 3: Boundary blur ─────────────────────────────────────────────
        if blur_boundaries and _has_scipy:
            sigma = np.random.uniform(1.0, 3.0)
            for c in range(3):
                blurred = gaussian_filter(seed[:, :, c], sigma=sigma)
                alpha_grad = np.abs(gaussian_filter(seed[:, :, 3], sigma=1.0)
                                    - seed[:, :, 3])
                boundary_mask = alpha_grad > 0.05
                seed[boundary_mask, c] = blurred[boundary_mask]

        # ── Noise 4: Global brightness shift ──────────────────────────────────
        brightness_shift = np.random.uniform(-0.1, 0.1)
        seed[:, :, :3] = np.clip(seed[:, :, :3] + brightness_shift, 0, 1)
        # Recalculate alive after possible dropout changes
        alive_now = seed[:, :, 3] > 0.5
        seed[~alive_now, :3] = 0.0

        # ── ch4-15: small random hidden state for active pixels ────────────────
        n_alive = alive_now.sum()
        if n_alive > 0:
            seed[alive_now, 4:] = np.random.uniform(
                0, 0.1, size=(n_alive, CHANNEL_N - 4)).astype(np.float32)

        seeds[b] = seed

    return seeds


# =============================================================================
# CA MODEL  ── UNCHANGED from Mordvintsev et al. 2020 ──
# =============================================================================
#
# This is the core theoretical contribution of the paper.
# Each cell runs the SAME small neural network as its update rule.
# The network is trained from data — not hand-crafted.
#
# Architecture per step:
#   1. perceive()  → Sobel filters sense local gradients (3 x CHANNEL_N features)
#   2. dmodel      → 1x1 Conv(128, ReLU) → 1x1 Conv(CHANNEL_N, zero-init)
#   3. Stochastic mask (fire_rate=0.5) decides which cells update
#   4. Living mask ensures empty cells stay empty
#
# ~8000 trainable parameters total.

class CAModel(tf.keras.Model):

    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate  = fire_rate

        self.dmodel = tf.keras.Sequential([
            Conv2D(128, 1, activation=tf.nn.relu),
            Conv2D(self.channel_n, 1, activation=None,
                   kernel_initializer=tf.zeros_initializer),
        ])

        self(tf.zeros([1, 3, 3, channel_n]))   # dummy call to build weights

    @tf.function
    def perceive(self, x, angle=0.0):
        identify = np.float32([0, 1, 0])
        identify = np.outer(identify, identify)
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0   # Sobel filter x
        dy = dx.T                                       # Sobel filter y
        c, s = tf.cos(angle), tf.sin(angle)
        kernel = tf.stack([identify, c * dx - s * dy,
                           s * dx + c * dy], -1)[:, :, None, :]
        kernel = tf.repeat(kernel, self.channel_n, 2)
        return tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')

    @tf.function
    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        pre_life_mask = get_living_mask(x)
        y  = self.perceive(x, angle)
        dx = self.dmodel(y) * step_size
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        x += dx * tf.cast(update_mask, tf.float32)
        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * tf.cast(life_mask, tf.float32)


# =============================================================================
# TRAINING UTILITIES  ── UNCHANGED from original ──
# =============================================================================

class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent      = _parent
        self._parent_idx  = _parent_idx
        self._slot_names  = slots.keys()
        self._size        = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx   = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        return SamplePool(**batch, _parent=self, _parent_idx=idx)

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)


@tf.function
def make_circle_masks(n, h, w):
    x = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
    x, y = (x - center[0]) / r, (y - center[1]) / r
    return tf.cast(x * x + y * y < 1.0, tf.float32)


# =============================================================================
# VISUALISATION — decay-appropriate outputs
# =============================================================================

def _safe_show():
    """Show plot if display is available, otherwise skip."""
    try:
        plt.show()
    except Exception:
        pass


def save_target_preview(target_img, output_dir):
    """Save both original-size copy and the 64x64 version used for training."""
    # Copy original to output
    if os.path.isfile(TARGET_IMAGE_PATH):
        import shutil
        ext = os.path.splitext(TARGET_IMAGE_PATH)[1]
        shutil.copy2(TARGET_IMAGE_PATH, os.path.join(output_dir, f"target_original{ext}"))

    # 64x64 preview — show RGBA on white and on chequered background
    rgb = to_rgb(target_img[None]).numpy()[0].clip(0, 1)
    alpha = target_img[..., 3]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    # (a) On white background
    axes[0].imshow(rgb)
    axes[0].set_title(f'On white ({TARGET_SIZE}x{TARGET_SIZE})', fontsize=11)
    axes[0].axis('off')
    # (b) Alpha channel
    axes[1].imshow(alpha, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Alpha channel (white=alive)', fontsize=11)
    axes[1].axis('off')
    # (c) Raw RGBA
    rgba_vis = np.zeros((*target_img.shape[:2], 4))
    rgba_vis[..., :3] = target_img[..., :3]
    rgba_vis[..., 3]  = target_img[..., 3]
    axes[2].imshow(rgba_vis.clip(0, 1))
    axes[2].set_title('RGBA (transparency preserved)', fontsize=11)
    axes[2].axis('off')

    plt.suptitle('Target decay pattern — used for CA training', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_64x64.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def save_training_snapshot(x0, x_out, step_i, output_dir):
    """Save before/after pair during training."""
    rgb0 = to_rgb(x0[0:1]).numpy()[0].clip(0, 1)
    rgb1 = to_rgb(x_out[0:1]).numpy()[0].clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb0)
    axes[0].set_title('Seed state', fontsize=11)
    axes[0].axis('off')
    axes[1].imshow(rgb1)
    axes[1].set_title(f'CA output @ step {step_i}', fontsize=11)
    axes[1].axis('off')
    plt.suptitle('Neural CA — Decay Pattern Learning', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'step_{step_i:05d}.png'),
                dpi=120, bbox_inches='tight')
    plt.close()


def save_pool_grid(pool, step_i, output_dir):
    """Save a grid of the current sample pool (49 samples)."""
    tiled = tile2d(to_rgb(pool.x[:49]).numpy().clip(0, 1))
    imwrite(os.path.join(output_dir, f'pool_{step_i:05d}.png'), tiled)


def plot_loss(loss_log, output_dir):
    """Plot training loss curve and save."""
    plt.figure(figsize=(10, 4))
    plt.title('Training loss (log scale) — decay pattern learning', fontsize=13)
    plt.plot(np.log10(loss_log), '.', alpha=0.1, color='#c0392b')
    plt.xlabel('Training step')
    plt.ylabel('log10( L2 loss )')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
    _safe_show()


def visualize_spread(ca, seed, steps, output_dir):
    """
    Run the trained CA forward and save the decay-spread sequence.

    This is the core 'Simulation as Inquiry' output:
      observe the emergent pattern that the LEARNED rules produce,
      starting from crack nucleation points.
    """
    x = seed.copy()
    # Capture 8 evenly-spaced snapshots
    capture_at = set(np.linspace(0, steps - 1, 8, dtype=int))
    snapshots = {}

    for i in tqdm.trange(steps, desc='  Decay spread'):
        x = ca(x).numpy()
        if i in capture_at:
            snapshots[i] = to_rgb(x[0:1]).numpy()[0].clip(0, 1)
            imwrite(os.path.join(output_dir, f'spread_t{i:04d}.png'), snapshots[i])

    # Summary figure (8 frames in 2 rows)
    sorted_items = sorted(snapshots.items())
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (ax, (t, frame)) in enumerate(zip(axes.flat, sorted_items)):
        ax.imshow(frame)
        ax.set_title(f't = {t}', fontsize=11)
        ax.axis('off')
    plt.suptitle(
        f'Decay spread — Neural CA  |  Experiment: {EXPERIMENT_TYPE}  |  '
        f'Seeds: {N_SEED_POINTS}\n(Simulation as Inquiry)',
        fontsize=13
    )
    plt.tight_layout()
    path = os.path.join(output_dir, 'decay_spread_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    _safe_show()
    return path


def visualize_ensemble(ca, h, w, n_runs, steps, output_dir,
                       use_noisy_seeds=False, use_road_seeds=False,
                       pad_target=None, noise_level=0.15, dropout_rate=0.08):
    """
    Run CA from N different seeds and compare results.

    Path A (default / road seeds):
      Same learned rules + different random seed positions
      → different emergent spatial patterns (Simulation as Inquiry)

    Path B (noisy seeds):
      Same zoning + different noise realisations
      → different evolved boundary forms
      (How does initial uncertainty affect the final stable form?)
    """
    fig, axes = plt.subplots(2, n_runs, figsize=(3.2 * n_runs, 6.5))

    for col in tqdm.trange(n_runs, desc='  Ensemble runs'):
        if use_noisy_seeds and pad_target is not None:
            seed_i = make_noisy_seed_from_target(pad_target,
                         noise_level=noise_level,
                         dropout_rate=dropout_rate, n_batch=1)
            seed_label = 'noisy T\u2080'
        elif use_road_seeds and pad_target is not None:
            seed_i = make_road_seed_from_target(pad_target)
            seed_label = 'road seed'
        else:
            seed_i = make_decay_seed(h, w, n_seeds=N_SEED_POINTS)
            seed_label = 'seed'

        seed_rgb = to_rgb(seed_i[0:1]).numpy()[0].clip(0, 1)
        x = seed_i.copy()
        for _ in range(steps):
            x = ca(x).numpy()
        final_rgb = to_rgb(x[0:1]).numpy()[0].clip(0, 1)

        axes[0, col].imshow(seed_rgb)
        axes[0, col].set_title(f'Run {col + 1}\n{seed_label}', fontsize=9)
        axes[0, col].axis('off')
        axes[1, col].imshow(final_rgb)
        axes[1, col].set_title(f't = {steps}', fontsize=9)
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Initial state', fontsize=11, labelpad=10)
    axes[1, 0].set_ylabel('Evolved state', fontsize=11, labelpad=10)

    if use_noisy_seeds:
        suptitle = (
            'Same zoning, different noise \u2192 diverse evolution paths\n'
            '(Simulation as Inquiry: initial uncertainty \u2192 final form variation)'
        )
    else:
        suptitle = (
            'Same rules, different initiations \u2192 diverse decay patterns\n'
            '(Emergence: local rules \u2192 complex global behaviour)'
        )
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, 'ensemble.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    _safe_show()
    return path


# =============================================================================
# HIDDEN CHANNEL VISUALISATION
# =============================================================================

def _save_channel_grid(state: np.ndarray, step: int, output_dir: str):
    """
    Save a 4×4 grid showing all 16 CA channels at a given time step.

    Layout:
      Row 0: ch0 (R),  ch1 (G),  ch2 (B),   ch3 (alpha)
      Row 1: ch4–ch7   (hidden)
      Row 2: ch8–ch11  (hidden)
      Row 3: ch12–ch15 (hidden)

    Each channel is displayed with its own colourmap and auto-normalised range.
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    channel_names = ['R', 'G', 'B', 'Alpha'] + [f'Hidden {i}' for i in range(4, 16)]
    cmaps = ['Reds', 'Greens', 'Blues', 'gray',
             'viridis', 'plasma', 'inferno', 'magma',
             'cividis', 'twilight', 'coolwarm', 'RdYlBu',
             'PiYG', 'PRGn', 'BrBG', 'RdBu']

    for i, (ax, name, cmap) in enumerate(zip(axes.flat, channel_names, cmaps)):
        ch = state[:, :, i]
        vmin, vmax = float(ch.min()), float(ch.max())
        if vmax - vmin < 1e-6:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(ch, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{name}\n[{vmin:.2f}, {vmax:.2f}]', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f'NCA Channel States — t = {step}\n'
        'ch0-2: RGB (visible)  |  ch3: alpha  |  ch4-15: hidden (learned internal state)',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'channels_t{step:04d}.png'),
                dpi=120, bbox_inches='tight')
    plt.close()


def _save_hidden_summary(captured_states: dict, output_dir: str):
    """
    Summary figure: rows = selected time steps, columns = RGB + alpha + 4 hidden channels.

    The 4 'most active' hidden channels are chosen by highest spatial standard deviation
    averaged across all captured time steps (high std = the channel is 'doing work').
    """
    steps = sorted(captured_states.keys())
    if len(steps) < 1:
        return

    # Pick up to 4 evenly-spaced time steps
    n_show = min(4, len(steps))
    indices = np.linspace(0, len(steps) - 1, n_show, dtype=int)
    selected = [steps[i] for i in indices]

    # Find 4 most active hidden channels (ch4-15)
    avg_std = np.zeros(12, dtype=np.float32)
    for state in captured_states.values():
        for j in range(12):
            avg_std[j] += float(state[:, :, 4 + j].std())
    avg_std /= len(captured_states)
    top4_hidden = list(np.argsort(avg_std)[::-1][:4] + 4)  # channel indices in 0-15 range

    # n_cols = RGB composite + alpha + 4 hidden
    n_cols = 6
    fig, axes = plt.subplots(n_show, n_cols, figsize=(n_cols * 3.2, n_show * 3.2))
    if n_show == 1:
        axes = axes[None, :]

    col_titles = ['RGB', 'Alpha'] + [f'Hidden {c}' for c in top4_hidden]

    for row, t in enumerate(selected):
        state = captured_states[t]

        # RGB composite
        rgb = np.clip(state[:, :, :3], 0, 1)
        a = np.clip(state[:, :, 3:4], 0, 1)
        rgb_on_white = 1.0 - a + rgb * a
        axes[row, 0].imshow(rgb_on_white.clip(0, 1))
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title('RGB', fontsize=10)

        # Alpha
        axes[row, 1].imshow(state[:, :, 3], cmap='gray', vmin=0, vmax=1)
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title('Alpha', fontsize=10)

        # 4 most active hidden channels
        for col_i, ch_idx in enumerate(top4_hidden):
            ch = state[:, :, ch_idx]
            vmin, vmax = float(ch.min()), float(ch.max())
            if vmax - vmin < 1e-6:
                vmin, vmax = 0.0, 1.0
            axes[row, 2 + col_i].imshow(ch, cmap='coolwarm', vmin=vmin, vmax=vmax)
            axes[row, 2 + col_i].axis('off')
            if row == 0:
                axes[row, 2 + col_i].set_title(f'Hidden {ch_idx}', fontsize=10)

        axes[row, 0].set_ylabel(f't = {t}', fontsize=10, labelpad=8, rotation=0,
                                 ha='right', va='center')

    plt.suptitle(
        'NCA Hidden State Summary\n'
        f'Columns: RGB, Alpha, top-4 active hidden channels (ch {top4_hidden})\n'
        'High activity = channel is encoding boundary detection / colour memory / diffusion gradient',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hidden_channels_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Hidden summary saved: hidden_channels_summary.png")
    print(f"  Most active hidden channels: {top4_hidden}")


def visualize_hidden_channels(ca, seed, steps, output_dir,
                               capture_steps=None):
    """
    Visualise all 16 NCA state channels over time.

    For each capture_step, saves a 4×4 channel grid PNG.
    Also saves a summary figure showing RGB + alpha + 4 most-active hidden channels
    at selected time steps.

    Args:
        ca            : trained CAModel
        seed          : initial state (1, H, W, 16) float32
        steps         : total evolution steps to run
        output_dir    : directory for output files
        capture_steps : list of steps to visualise (default: 7 evenly-spaced)
    """
    if capture_steps is None:
        capture_steps = [0, 5, 10, 20, 50, 100, 200, min(300, steps)]
    capture_steps = sorted(set(s for s in capture_steps if s <= steps))

    hidden_dir = os.path.join(output_dir, 'hidden_channels')
    os.makedirs(hidden_dir, exist_ok=True)

    capture_set = set(capture_steps)
    captured_states = {}

    x = seed.copy()
    max_step = max(capture_steps)

    for step in tqdm.trange(max_step + 1, desc='  Hidden channel viz'):
        if step in capture_set:
            state = x[0] if isinstance(x, np.ndarray) else x.numpy()[0]
            _save_channel_grid(state, step, hidden_dir)
            captured_states[step] = state.copy()
        if step < max_step:
            x = ca(tf.constant(x)).numpy()

    _save_hidden_summary(captured_states, output_dir)
    print(f"  Per-step channel grids: {hidden_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def print_banner():
    print("=" * 70)
    print("  Neural CA — Architectural Decay Simulation")
    print("  Based on Mordvintsev et al. (2020) Growing Neural Cellular Automata")
    print("=" * 70)


def print_environment():
    """Print system and GPU info for reproducibility."""
    print(f"\n  Python      : {sys.version.split()[0]}")
    print(f"  TensorFlow  : {tf.__version__}")
    print(f"  NumPy       : {np.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            print(f"  GPU         : {g.name}")
        print("  (Training will use GPU — expect ~30-60 min)")
    else:
        print("  GPU         : None (CPU only)")
        print("  WARNING: CPU training is much slower (~3-5 hours).")
        print("           Use --quick for a fast test run first.")
    print()


def print_file_summary(files):
    """Print a summary of all generated files."""
    print("\n" + "=" * 70)
    print("  GENERATED FILES")
    print("=" * 70)
    for desc, path in files:
        exists = "OK" if os.path.isfile(path) else "??"
        size_kb = os.path.getsize(path) / 1024 if os.path.isfile(path) else 0
        print(f"  [{exists}] {desc}")
        print(f"       {path}  ({size_kb:.0f} KB)")
    print("=" * 70)


def main():
    # ── Parse arguments ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Neural CA Decay Simulation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 500 steps (~5 min on GPU)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Override number of training steps')
    parser.add_argument('--seeds', type=int, default=None,
                        help='Override number of random seed points (ignored if --road_seeds)')
    parser.add_argument('--experiment', choices=['Growing', 'Persistent', 'Regenerating'],
                        default=None, help='Override experiment type')
    # ── New arguments for strategy map training ──────────────────────────────
    parser.add_argument('--target', type=str, default=None,
                        help='Path to target image (default: nca_target_256.png)')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output directory (default: generate_neural_ca/)')
    parser.add_argument('--road_seeds', action='store_true',
                        help='Use road/infrastructure pixels from target as seed points '
                             '(detects dark-grey pixels; for Dagenham strategy map)')
    parser.add_argument('--log_interval', type=int, default=None,
                        help='Save snapshot every N steps (default: TRAIN_STEPS // 10)')
    # ── Path B: noisy-seed training mode ──────────────────────────────────────
    parser.add_argument('--noisy_seeds', action='store_true',
                        help='Path B training: use noisy target image as initial state '
                             '(NCA learns to stabilise imperfect zoning over time). '
                             'Designed for Pix2Pix→NCA pipeline.')
    parser.add_argument('--noise_level', type=float, default=0.15,
                        help='RGB noise amplitude (0-1), default 0.15')
    parser.add_argument('--dropout_rate', type=float, default=0.08,
                        help='Fraction of active pixels randomly erased, default 0.08')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Skip training: load existing checkpoint and generate results only. '
                             'Requires --output pointing to an already-trained run directory.')
    parser.add_argument('--no_hidden_viz', action='store_true',
                        help='Skip hidden channel visualisation (saves ~2 min). '
                             'Hidden viz is also skipped automatically in --quick mode.')
    args = parser.parse_args()

    # Apply overrides
    global TRAIN_STEPS, N_SEED_POINTS, EXPERIMENT_TYPE
    global EXPERIMENT_N, USE_PATTERN_POOL, DAMAGE_N
    global TARGET_IMAGE_PATH, OUTPUT_DIR
    global DIR_INPUT, DIR_CHECKPOINTS, DIR_TRAINING, DIR_RESULTS

    if args.quick:
        TRAIN_STEPS = 500
    if args.steps is not None:
        TRAIN_STEPS = args.steps
    if args.seeds is not None:
        N_SEED_POINTS = args.seeds
    if args.experiment is not None:
        EXPERIMENT_TYPE  = args.experiment
        EXPERIMENT_N     = EXPERIMENT_MAP[EXPERIMENT_TYPE]
        USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
        DAMAGE_N         = [0, 0, 3][EXPERIMENT_N]
    if args.target is not None:
        TARGET_IMAGE_PATH = args.target
    if args.output is not None:
        OUTPUT_DIR     = args.output
        DIR_INPUT      = os.path.join(OUTPUT_DIR, "input")
        DIR_CHECKPOINTS= os.path.join(OUTPUT_DIR, "checkpoints")
        DIR_TRAINING   = os.path.join(OUTPUT_DIR, "training")
        DIR_RESULTS    = os.path.join(OUTPUT_DIR, "results")

    # ── Setup ─────────────────────────────────────────────────────────────────
    print_banner()
    print_environment()

    for d in [DIR_INPUT, DIR_CHECKPOINTS, DIR_TRAINING, DIR_RESULTS]:
        os.makedirs(d, exist_ok=True)

    generated_files = []   # (description, path) tuples for final summary

    # ── 1. Load target image ──────────────────────────────────────────────────
    print("[1/5] Loading target decay image...")
    try:
        target_img = load_decay_image(TARGET_IMAGE_PATH)
        print(f"  Loaded: {TARGET_IMAGE_PATH}")
    except Exception as e:
        print(f"  [!] Could not load image: {e}")
        print(f"  Generating synthetic crack pattern for demonstration...")
        target_img = make_synthetic_crack_target(TARGET_SIZE)

    save_target_preview(target_img, DIR_INPUT)
    generated_files.append(("Target preview (64x64)",
                            os.path.join(DIR_INPUT, 'target_64x64.png')))

    print(f"  Target shape: {target_img.shape}")
    print(f"  Output dir  : {OUTPUT_DIR}")

    # ── 2. Pad target and create seed ─────────────────────────────────────────
    print("\n[2/5] Preparing grid and seed...")
    p          = TARGET_PADDING
    pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])
    h, w       = pad_target.shape[:2]

    if args.noisy_seeds:
        print("  Seed mode   : noisy target (Path B — temporal evolution)")
        seed = make_noisy_seed_from_target(pad_target,
                                            noise_level=args.noise_level,
                                            dropout_rate=args.dropout_rate,
                                            n_batch=1)
    elif args.road_seeds:
        print("  Seed mode   : road skeleton (Path A — morphogenesis)")
        seed = make_road_seed_from_target(pad_target)
    else:
        seed = make_decay_seed(h, w, n_seeds=N_SEED_POINTS)

    if args.noisy_seeds:
        seed_mode_str = f'noisy target (noise={args.noise_level}, dropout={args.dropout_rate})'
    elif args.road_seeds:
        seed_mode_str = 'road skeleton'
    else:
        seed_mode_str = f'{N_SEED_POINTS} random points'

    print(f"  Grid size   : {h} x {w}  (target {TARGET_SIZE} + padding {p}x2)")
    print(f"  Seed mode   : {seed_mode_str}")
    print(f"  Experiment  : {EXPERIMENT_TYPE}")
    print(f"  Train steps : {TRAIN_STEPS}")

    def loss_f(x):
        return tf.reduce_mean(tf.square(to_rgba(x) - pad_target), [-2, -3, -1])

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("\n[3/5] Building Neural CA model...")
    ca = CAModel()
    ca.dmodel.summary()

    n_params = sum(np.prod(w.shape) for w in ca.weights)
    print(f"  Total parameters: {n_params:,}")

    lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [4000], [LR_INIT, LR_INIT * 0.1])
    trainer  = tf.keras.optimizers.Adam(lr_sched)
    loss_log = []

    pool = SamplePool(x=np.repeat(seed, POOL_SIZE, 0))

    @tf.function
    def train_step(x):
        iter_n = tf.random.uniform([], 64, 96, tf.int32)
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
                x = ca(x)
            loss = tf.reduce_mean(loss_f(x))
        grads = g.gradient(loss, ca.weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        trainer.apply_gradients(zip(grads, ca.weights))
        return x, loss

    # ── 4. Training loop (or load checkpoint) ────────────────────────────────
    if args.visualize_only:
        print(f"\n[4/5] Skipping training — loading existing checkpoint...")
        final_ckpt = os.path.join(DIR_CHECKPOINTS, 'final')
        if not os.path.isfile(final_ckpt + '.index'):
            print(f"  [!] No checkpoint found at: {final_ckpt}")
            print(f"  Run without --visualize_only to train first.")
            return
        ca.load_weights(final_ckpt)
        print(f"  Loaded: {final_ckpt}")
        generated_files.append(("Final model weights", final_ckpt + '.index'))
    else:
        print(f"\n[4/5] Training ({TRAIN_STEPS} steps)...")
        t_start   = time.time()
        if args.log_interval is not None:
            save_every = args.log_interval
        else:
            save_every = max(TRAIN_STEPS // 10, 100)   # save ~10 snapshots

        for i in range(TRAIN_STEPS + 1):

            if USE_PATTERN_POOL:
                batch     = pool.sample(BATCH_SIZE)
                x0        = batch.x
                loss_rank = loss_f(x0).numpy().argsort()[::-1]
                x0        = x0[loss_rank]
                if args.noisy_seeds:
                    # Path B: inject a freshly-noised seed every iteration
                    x0[:1] = make_noisy_seed_from_target(pad_target,
                                 noise_level=args.noise_level,
                                 dropout_rate=args.dropout_rate, n_batch=1)
                else:
                    x0[:1] = seed
                if DAMAGE_N:
                    damage = 1.0 - make_circle_masks(DAMAGE_N, h, w).numpy()[..., None]
                    x0[-DAMAGE_N:] *= damage
            else:
                if args.noisy_seeds:
                    x0 = make_noisy_seed_from_target(pad_target,
                             noise_level=args.noise_level,
                             dropout_rate=args.dropout_rate, n_batch=BATCH_SIZE)
                else:
                    x0 = np.repeat(seed, BATCH_SIZE, 0)

            x, loss = train_step(x0)

            if USE_PATTERN_POOL:
                batch.x[:] = x
                batch.commit()

            step_i = len(loss_log)
            loss_log.append(loss.numpy())

            # Save periodic snapshots
            if step_i % save_every == 0:
                save_training_snapshot(x0, x.numpy(), step_i, DIR_TRAINING)
                ca.save_weights(os.path.join(DIR_CHECKPOINTS, f'{step_i:05d}'))
                if USE_PATTERN_POOL:
                    save_pool_grid(pool, step_i, DIR_TRAINING)

            # Progress
            elapsed = time.time() - t_start
            if step_i > 0:
                eta = elapsed / step_i * (TRAIN_STEPS - step_i)
                eta_str = time.strftime('%M:%S', time.gmtime(eta))
            else:
                eta_str = '--:--'
            print(f'\r  step {step_i:5d}/{TRAIN_STEPS} | '
                  f'loss: {np.log10(loss.numpy()):.3f} | '
                  f'elapsed: {elapsed:.0f}s | ETA: {eta_str}', end='')

        t_total = time.time() - t_start
        print(f'\n  Training complete in {t_total / 60:.1f} minutes.')

        # Save final checkpoint
        final_ckpt = os.path.join(DIR_CHECKPOINTS, 'final')
        ca.save_weights(final_ckpt)
        generated_files.append(("Final model weights", final_ckpt + '.index'))

    # Loss curve (only if we actually trained)
    if loss_log:
        plot_loss(loss_log, DIR_TRAINING)
        generated_files.append(("Loss curve",
                                os.path.join(DIR_TRAINING, 'loss_curve.png')))

    # ── 5. Simulation as Inquiry outputs ──────────────────────────────────────
    print(f"\n[5/5] Generating thesis figures...")

    # (a) Spread / evolution sequence
    print("\n  (a) Spread sequence...")
    if args.noisy_seeds:
        spread_seed = make_noisy_seed_from_target(pad_target,
                          noise_level=args.noise_level,
                          dropout_rate=args.dropout_rate, n_batch=1)
    elif args.road_seeds:
        spread_seed = make_road_seed_from_target(pad_target)
    else:
        spread_seed = make_decay_seed(h, w, n_seeds=N_SEED_POINTS)
    spread_steps = 80  # must match training iter_n (64-96); beyond ~100 causes over-evolution
    path = visualize_spread(ca, spread_seed, spread_steps, DIR_RESULTS)
    generated_files.append(("Spread summary (THESIS FIGURE)", path))

    # (b) Hidden channel visualisation — NCA internal "thought process"
    if not args.quick and not args.no_hidden_viz:
        print("\n  (b) Hidden channel visualisation...")
        visualize_hidden_channels(ca, spread_seed,
                                   steps=min(300, TRAIN_STEPS),
                                   output_dir=DIR_RESULTS,
                                   capture_steps=[0, 5, 10, 20, 40, 60, 80])
        generated_files.append(("Hidden channel summary (THESIS FIGURE)",
                                os.path.join(DIR_RESULTS, 'hidden_channels_summary.png')))
    else:
        print("\n  (b) Hidden channel visualisation skipped (--quick or --no_hidden_viz)")

    # (d) Ensemble: same rules, different initiations
    print("\n  (d) Ensemble comparison...")
    ensemble_steps = 80  # must match training iter_n (64-96)
    path = visualize_ensemble(ca, h, w, n_runs=6, steps=ensemble_steps,
                              output_dir=DIR_RESULTS,
                              use_noisy_seeds=args.noisy_seeds,
                              use_road_seeds=args.road_seeds,
                              pad_target=pad_target,
                              noise_level=args.noise_level,
                              dropout_rate=args.dropout_rate)
    generated_files.append(("Ensemble comparison (THESIS FIGURE)", path))

    # ── Summary ───────────────────────────────────────────────────────────────
    # Add training snapshots to list
    for fn in sorted(os.listdir(DIR_TRAINING)):
        if fn.startswith('step_'):
            generated_files.append(("Training snapshot",
                                    os.path.join(DIR_TRAINING, fn)))
            break   # just list the first one as representative

    print_file_summary(generated_files)

    print(f"\n  All output saved to: {OUTPUT_DIR}")
    print(f"  Directory structure:")
    print(f"    input/         — target image (original + 64x64)")
    print(f"    checkpoints/   — model weights (can reload for further experiments)")
    print(f"    training/      — loss curve + training progress snapshots")
    print(f"    results/       — thesis figures (spread + ensemble)")
    print(f"\n  Done.\n")


if __name__ == '__main__':
    main()
