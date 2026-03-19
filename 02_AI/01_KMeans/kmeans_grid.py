# -*- coding: utf-8 -*-
r"""
K-Means Clustering for Programming Decay — Barking & Dagenham
Author: Jiajia Jiang | UCL Bartlett MArch RC18
Script: kmeans_grid.py

Input:  D:\Claude_jiajia\Documents\Grid_Features.csv
Output: D:\Claude_jiajia\Documents\K-Means\
        - Grid_Clustered.csv
        - elbow_plot.png
        - silhouette_plot.png
        - cluster_matrix_8x8.png
        - cluster_radar.png
        - cluster_pca.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_CSV = r"D:\Claude_jiajia\Documents\Grid_Features.csv"
OUTPUT_DIR = r"D:\Claude_jiajia\Documents\K-Means"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "Brownfield_Pct",
    "Building_Coverage_Pct",
    "Waste_Site_Count",
    "Flood_Zone_Pct",
    "Population",
    "Sensitive_Area_Pct",
]
FEATURE_LABELS = [
    "F1 Brownfield",
    "F2 Building",
    "F3 Waste",
    "F4 Flood",
    "F5 Population",
    "F6 Sensitive",
]

RANDOM_STATE = 42
K_RANGE = range(2, 11)

# Portfolio colour palette (muted, print-friendly)
CLUSTER_COLORS = [
    "#C0392B",  # deep red
    "#E67E22",  # burnt orange
    "#F1C40F",  # amber
    "#27AE60",  # forest green
    "#2980B9",  # steel blue
    "#8E44AD",  # plum
    "#16A085",  # teal
    "#D35400",  # rust
    "#7F8C8D",  # slate grey
]


# ─────────────────────────────────────────────
# 1. LOAD & NORMALISE
# ─────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(INPUT_CSV)
print(f"  {len(df)} cells loaded. Columns: {list(df.columns)}")

X_raw = df[FEATURES].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

print("  MinMaxScaler applied (0–1 range).")


# ─────────────────────────────────────────────
# 2. ELBOW METHOD
# ─────────────────────────────────────────────
print("\nElbow Method …")
inertia = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    print(f"  K={k}  Inertia={km.inertia_:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(K_RANGE), inertia, "o-", color="#C0392B", linewidth=2, markersize=7)
ax.fill_between(list(K_RANGE), inertia, alpha=0.08, color="#C0392B")
ax.set_xlabel("Number of Clusters  K", fontsize=12)
ax.set_ylabel("Inertia  (WCSS)", fontsize=12)
ax.set_title("Elbow Method — Optimal K Selection", fontsize=13, fontweight="bold")
ax.set_xticks(list(K_RANGE))
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
elbow_path = os.path.join(OUTPUT_DIR, "elbow_plot.png")
plt.savefig(elbow_path, dpi=200)
plt.close()
print(f"  Saved: {elbow_path}")


# ─────────────────────────────────────────────
# 3. SILHOUETTE SCORES
# ─────────────────────────────────────────────
print("\nSilhouette Scores …")
sil_scores = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)
    print(f"  K={k}  Silhouette={score:.4f}")

best_k_sil = list(K_RANGE)[int(np.argmax(sil_scores))]
print(f"  → Best K by Silhouette: {best_k_sil}")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(list(K_RANGE), sil_scores, color="#2980B9", alpha=0.8, edgecolor="white")
bars[int(np.argmax(sil_scores))].set_color("#C0392B")
ax.axhline(y=max(sil_scores), color="#C0392B", linestyle="--", alpha=0.5, linewidth=1)
ax.set_xlabel("Number of Clusters  K", fontsize=12)
ax.set_ylabel("Silhouette Score", fontsize=12)
ax.set_title("Silhouette Score — Optimal K Selection", fontsize=13, fontweight="bold")
ax.set_xticks(list(K_RANGE))
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate best K
best_idx = int(np.argmax(sil_scores))
ax.annotate(
    f"Best K={best_k_sil}\n({max(sil_scores):.3f})",
    xy=(best_k_sil, max(sil_scores)),
    xytext=(best_k_sil + 0.5, max(sil_scores) - 0.03),
    fontsize=9,
    color="#C0392B",
    arrowprops=dict(arrowstyle="->", color="#C0392B"),
)
plt.tight_layout()
sil_path = os.path.join(OUTPUT_DIR, "silhouette_plot.png")
plt.savefig(sil_path, dpi=200)
plt.close()
print(f"  Saved: {sil_path}")


# ─────────────────────────────────────────────
# 4. CHOOSE FINAL K
# ─────────────────────────────────────────────
# Auto-select: prefer silhouette best, but cap at 6 for semantic clarity.
# Override manually here if needed after inspecting the plots.
FINAL_K = best_k_sil
if FINAL_K > 6:
    FINAL_K = 6
    print(f"\n  K capped at 6 for semantic clarity (silhouette suggested {best_k_sil}).")
print(f"\n>>> Final K = {FINAL_K}")


# ─────────────────────────────────────────────
# 5. RUN K-MEANS WITH FINAL K
# ─────────────────────────────────────────────
print("\nRunning K-Means …")
km_final = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=50)
df["Cluster"] = km_final.fit_predict(X_scaled)

# Re-label clusters by mean Brownfield+Waste (descending = most contaminated first)
centers_df = pd.DataFrame(
    scaler.inverse_transform(km_final.cluster_centers_), columns=FEATURES
)
centers_df["Cluster_old"] = range(FINAL_K)
contamination_score = (
    centers_df["Brownfield_Pct"] / centers_df["Brownfield_Pct"].max().clip(1e-9)
    + centers_df["Waste_Site_Count"] / centers_df["Waste_Site_Count"].max().clip(1e-9)
)
centers_df["Rank"] = contamination_score.rank(ascending=False, method="first").astype(int) - 1
remap = dict(zip(centers_df["Cluster_old"], centers_df["Rank"]))
df["Cluster"] = df["Cluster"].map(remap)

# Rebuild centers in new order
km_final_centers_scaled = km_final.cluster_centers_
new_order = [centers_df.loc[centers_df["Rank"] == r, "Cluster_old"].values[0] for r in range(FINAL_K)]
centers_scaled_reordered = km_final_centers_scaled[new_order]
centers_raw_reordered = scaler.inverse_transform(centers_scaled_reordered)
centers_final = pd.DataFrame(centers_raw_reordered, columns=FEATURES)
centers_final.index.name = "Cluster"

print("  Cluster means (raw values):")
print(centers_final.round(2).to_string())

# Print cluster sizes
for c in sorted(df["Cluster"].unique()):
    n = (df["Cluster"] == c).sum()
    print(f"  Cluster {c}: {n} cells")


# ─────────────────────────────────────────────
# 6. SAVE Grid_Clustered.csv
# ─────────────────────────────────────────────
out_cols = ["Cell_ID"] + FEATURES + ["Cluster"]
df_out = df[out_cols].copy()
clustered_path = os.path.join(OUTPUT_DIR, "Grid_Clustered.csv")
df_out.to_csv(clustered_path, index=False)
print(f"\n  Saved: {clustered_path}")


# ─────────────────────────────────────────────
# 7A. VISUALISATION — 8×8 Matrix Heatmap
# ─────────────────────────────────────────────
print("\nGenerating 8×8 cluster matrix …")

# Cell_IDs 1-64, row=top-to-bottom, col=left-to-right
grid = np.full((8, 8), np.nan)
for _, row in df_out.iterrows():
    cid = int(row["Cell_ID"])
    r = (cid - 1) // 8
    c = (cid - 1) % 8
    grid[r, c] = row["Cluster"]

cmap = mcolors.ListedColormap(CLUSTER_COLORS[:FINAL_K])
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, FINAL_K + 0.5), ncolors=FINAL_K)

# Cluster labels for legend — based on actual K=6 cluster means:
# C0: Brown=7.0  Waste=6.6  Flood=127  Pop=32k  → Brownfield + Waste + Flood (priority target)
# C1: Brown=0    Waste=19.4 Flood=43   Pop=5k   → Industrial Waste Concentration (fringe)
# C2: Brown=2.3  Waste=4.6  Flood=71   Pop=87k  → Flood-Exposed Mixed Industrial
# C3: Brown=0.9  Waste=3.6  Flood=11   Pop=37k  → General Urban Mixed (largest cluster)
# C4: Brown=0.6  Waste=1.6  Flood=7    Pop=97k  → Dense Residential Core
# C5: Brown=0    Waste=3    Flood=37   Pop=71k  → Ecological Sensitive Zone (1 cell)
cluster_desc = {
    0: "High Contamination Zone",
    1: "Industrial Waste Concentration",
    2: "Flood-Exposed Mixed",
    3: "General Urban Mixed",
    4: "Dense Residential Core",
    5: "Ecological Sensitive Zone",
}

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="equal")

# Grid lines
for x in range(9):
    ax.axvline(x - 0.5, color="white", linewidth=1.5)
for y in range(9):
    ax.axhline(y - 0.5, color="white", linewidth=1.5)

# Cell ID labels
for _, row in df_out.iterrows():
    cid = int(row["Cell_ID"])
    r = (cid - 1) // 8
    c = (cid - 1) % 8
    cluster = int(row["Cluster"])
    # White or dark text depending on background brightness
    bg = mcolors.hex2color(CLUSTER_COLORS[cluster])
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    fc = "white" if lum < 0.5 else "#1a1a1a"
    ax.text(c, r, str(cid), ha="center", va="center", fontsize=8, color=fc, fontweight="bold")

ax.set_xticks([])
ax.set_yticks([])
ax.set_title(
    f"K-MEANS CLUSTERING  /  K = {FINAL_K}\nBarking & Dagenham  —  64-Cell Grid",
    fontsize=13,
    fontweight="bold",
    pad=12,
)

# Legend
patches = []
for c in range(FINAL_K):
    label = f"Cluster {c}  —  {cluster_desc.get(c, '')}"
    patches.append(mpatches.Patch(color=CLUSTER_COLORS[c], label=label))
ax.legend(
    handles=patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    fontsize=9,
    frameon=False,
)

plt.tight_layout()
matrix_path = os.path.join(OUTPUT_DIR, "cluster_matrix_8x8.png")
plt.savefig(matrix_path, dpi=250, bbox_inches="tight")
plt.close()
print(f"  Saved: {matrix_path}")


# ─────────────────────────────────────────────
# 7B. VISUALISATION — Radar Chart
# ─────────────────────────────────────────────
print("\nGenerating radar chart …")

centers_scaled_df = pd.DataFrame(
    centers_scaled_reordered, columns=FEATURE_LABELS
)

N = len(FEATURE_LABELS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for c in range(FINAL_K):
    values = centers_scaled_df.iloc[c].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, color=CLUSTER_COLORS[c], label=f"Cluster {c}")
    ax.fill(angles, values, alpha=0.12, color=CLUSTER_COLORS[c])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(FEATURE_LABELS, fontsize=11)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")
ax.set_ylim(0, 1)
ax.set_title(
    f"Cluster Feature Profiles  /  K = {FINAL_K}",
    fontsize=13,
    fontweight="bold",
    pad=20,
)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9, frameon=False)
ax.grid(color="grey", linestyle="--", alpha=0.4)
ax.spines["polar"].set_color("grey")

plt.tight_layout()
radar_path = os.path.join(OUTPUT_DIR, "cluster_radar.png")
plt.savefig(radar_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {radar_path}")


# ─────────────────────────────────────────────
# 7C. VISUALISATION — PCA 2D Scatter
# ─────────────────────────────────────────────
print("\nGenerating PCA scatter plot …")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
var1, var2 = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(9, 7))
for c in range(FINAL_K):
    mask = df["Cluster"] == c
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        color=CLUSTER_COLORS[c],
        label=f"Cluster {c}",
        s=80,
        edgecolors="white",
        linewidths=0.6,
        alpha=0.9,
        zorder=3,
    )
    # Annotate Cell_IDs
    for idx in df[mask].index:
        cid = int(df.loc[idx, "Cell_ID"])
        ax.annotate(
            str(cid),
            (X_pca[idx, 0], X_pca[idx, 1]),
            fontsize=6,
            color="#333333",
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

ax.set_xlabel(f"PC1  ({var1:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2  ({var2:.1f}% variance)", fontsize=11)
ax.set_title(
    f"PCA — 2D Cluster Projection  /  K = {FINAL_K}\nBarking & Dagenham Grid Cells",
    fontsize=13,
    fontweight="bold",
)
ax.legend(fontsize=9, frameon=True, framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
pca_path = os.path.join(OUTPUT_DIR, "cluster_pca.png")
plt.savefig(pca_path, dpi=200)
plt.close()
print(f"  Saved: {pca_path}")


# ─────────────────────────────────────────────
# 8. SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"FINAL K = {FINAL_K}  |  Silhouette = {sil_scores[list(K_RANGE).index(FINAL_K)]:.4f}")
print("=" * 60)
print("\nCluster Summary (raw feature means):")
summary = df.groupby("Cluster")[FEATURES].mean().round(3)
summary["n_cells"] = df.groupby("Cluster").size()
print(summary.to_string())
print("\n>>> Look for the cluster with HIGH F1_Brownfield + HIGH F3_Waste")
print("    → That cluster = 'High Contamination Zone' = primary site selection pool")
print("=" * 60)
print("\nAll outputs saved to:", OUTPUT_DIR)
