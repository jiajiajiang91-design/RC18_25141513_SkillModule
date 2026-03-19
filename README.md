# PROGRAMMING DECAY — Skill Module Repository

**A Multi-Scalar Design Intelligence Framework for Post-Industrial Bioremediation at Dagenham Dock, East London**

| | |
|---|---|
| **Cluster** | RC18 |
| **Student Number** | 25141513 |
| **Programme** | UCL Bartlett B-Pro MArch Urban Design |
| **Academic Year** | 2025–2026 |
| **Tutors** | Dimitra Bra (GIS) · William Huang (AI) · Enriqueta Llabres-Valls (UE) |

> **OneDrive (Large Files):**
> - [ARCGIS.zip (GIS Project Files)](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucbv512_ucl_ac_uk/IQDIDfrpRPtwT5bGW-8U2O0MAcz4RAC6Ww1Hy1svnYwgE58?e=ncj8z1)
> - [Niagara_test2.zip (UE Project Files)](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucbv512_ucl_ac_uk/IQAcYDZaTZBETae4gxXVY24-AcoZLthi7GDia0bszxxIPM4?e=0UsAl3)

> **Video Links:**
> - [Latent Walk Video (YouTube)](https://youtu.be/XniW9sVhLH0)
> - [N8N Entropy Atlas Demo (YouTube)](https://youtube.com/watch?v=XA8W4Zhc5Ek)
> - [UE Niagara Systems (YouTube)](https://youtu.be/oyOwx5gqYzE)
> - [Entropy Atlas (Live Interactive Map)](https://jiajiajiang91-design.github.io/Jiajia_Jiang.Github.io/N8N_OUTPUT_entropy_atlas.html)

---

## Overview

This repository hosts all code, data, and supporting materials for the **Skill Module Portfolio** — a combined submission across three workshops:

1. **Relational Cartographies (GIS)** — Multi-scalar environmental mapping from planetary to local scale
2. **Generative AI** — Five-system design intelligence pipeline (K-Means → N8N → Latent Walk → Pix2Pix → Neural CA)
3. **Unreal Engine** — Real-time particle systems and materials simulating industrial decay

The three modules are unified by a single design research agenda: **Programming Decay**, a thesis investigating fungal bioremediation strategies for the former Ford Stamping Plant at Dagenham Dock, East London.

---

## Repository Structure

```
RC18_25141513_SkillModule/
│
├── 01_GIS/
│   ├── scripts/
│   │   ├── 01_feature_extraction.ipynb   # ArcPy: 6-feature extraction from 64-cell grid
│   │   └── 02_composite_index.ipynb      # ArcPy: Composite Contamination Index calculation
│   ├── data/
│   │   ├── Grid_Features.csv             # 64 cells × 6 environmental features
│   │   └── east_london_feedback_gis.csv  # 130+ geo-tagged sentiment reviews
│   └── outputs/
│       └── cells/                        # 64 grid cell map tiles (PNG)
│
├── 02_AI/
│   ├── 01_KMeans/
│   │   ├── kmeans_grid.py                # K-Means clustering (K=6, scikit-learn)
│   │   ├── Grid_Clustered.csv            # Clustering results with labels
│   │   ├── elbow_plot.png                # Elbow method visualisation
│   │   └── silhouette_plot.png           # Silhouette score analysis
│   │
│   ├── 02_N8N/
│   │   ├── AA_Overpass_Dagenham_LandUse.json   # N8N workflow: Overpass API extraction
│   │   ├── AA_Reddit_Dagenham_Feedback.json    # N8N workflow: Reddit sentiment scraping
│   │   ├── overpass_dagenham.csv                # 986 OSM sites extracted
│   │   └── entropy_atlas.html                  # Interactive map (self-contained)
│   │
│   ├── 03_LatentWalk/
│   │   ├── latent_walk_60frames.json     # ComfyUI workflow for SD interpolation
│   │   ├── crop_images.py                # Image preprocessing
│   │   ├── download_images.py            # Dataset acquisition
│   │   └── latent_space_vis_demo_fixed.gh  # Grasshopper: 3D latent space visualisation
│   │
│   ├── 04_Pix2Pix/
│   │   ├── train.py                      # Pix2Pix training loop (PyTorch)
│   │   ├── inference.py                  # Single-image inference
│   │   ├── multi_inference.py            # Multi-epoch batch inference
│   │   ├── config.py                     # Hyperparameters (epochs=480, λ_L1=100)
│   │   ├── prepare_data.py               # Dataset preparation pipeline
│   │   ├── voronoi_postprocess.py        # Delaunay triangulation post-processing
│   │   ├── batch_voronoi.py              # Batch Voronoi for all epochs
│   │   ├── color_stats.py                # Functional zone area statistics
│   │   └── data/                         # train/val/test paired datasets
│   │
│   └── 05_NeuralCA/
│       ├── decay_neural_ca.py            # NCA training (Mordvintsev et al. 2020)
│       ├── pix2pix_to_ca.py              # Pix2Pix→NCA bridge: K-Means quantisation
│       ├── run_ca_from_pix2pix.py        # End-to-end pipeline execution
│       ├── nca_target_256.png            # Target strategy map
│       └── nca_strategy_run_v3/          # Training outputs, checkpoints, figures
│
├── 03_UE/
│   └── README.md                         # UE project description + OneDrive link
│
└── README.md                             # This file
```

---

## Module 01: Relational Cartographies (GIS)

**Tutor: Dimitra Bra**

### Approach
Multi-scalar environmental cartography tracing concrete's contamination lifecycle across four scales:

| Scale | Focus | Key Datasets |
|-------|-------|-------------|
| **Global** | Energy infrastructure, GHG emissions, urban population | WRI Global Power Plant DB, UCDB Health Emissions R2024A, UN World Cities |
| **National (UK)** | Atmospheric emissions, heavy metal soil contamination, waste | NAEI 1×1km gridded, BGS Normal Background Concentrations |
| **Regional (London)** | Multi-risk convergence: chemical × flood × ecology × density | Environment Agency, Natural England, OS OpenMap |
| **Local (Dagenham)** | Digital phenomenology: sentiment analysis of 137 geo-tagged reviews | Google Places API via N8N automation |

### Scripts
- **`01_feature_extraction.ipynb`** — ArcPy script that extracts 6 environmental features (brownfield density, building coverage, waste infrastructure, flood risk, population, ecological sensitivity) from spatial join results across 64 fishnet grid cells. Outputs `Grid_Features.csv`.
- **`02_composite_index.ipynb`** — Computes a min-max normalised Composite Contamination Index and writes it back to the ArcGIS Pro attribute table for cartographic visualisation.

### Large Files (OneDrive)
- [ARCGIS.zip — Full ArcGIS Pro project + geodatabase](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucbv512_ucl_ac_uk/IQDIDfrpRPtwT5bGW-8U2O0MAcz4RAC6Ww1Hy1svnYwgE58?e=ncj8z1)

### Software
- ArcGIS Pro 3.6 (Esri)
- Projections: Orthographic (global sphere), Mercator (continental), British National Grid (UK/London)

---

## Module 02: Generative AI

**Tutor: William Huang**

### Pipeline Overview

```
K-Means → N8N Automation → Latent Walk → Pix2Pix → Neural CA
 (classify)   (map entropy)   (imagine)    (translate)  (simulate)
```

### System 01: K-Means Clustering
- **Input:** 64 cells × 6 features (`Grid_Features.csv`)
- **Method:** K-Means (K=6), validated by Elbow + Silhouette + PCA
- **Output:** Cluster 0 = 7 cells = Dagenham Dock priority intervention zone
- **Script:** `02_AI/01_KMeans/kmeans_grid.py`

### System 02: N8N Automation — Entropy Atlas
- **Pipeline:** N8N workflows automate Overpass API + Google Places scraping
- **Output:** 1,026 geo-located data points; interactive Leaflet map
- **Live URL:** [Entropy Atlas](https://jiajiajiang91-design.github.io/Jiajia_Jiang.Github.io/N8N_OUTPUT_entropy_atlas.html)
- **Video:** [N8N Entropy Atlas Demo (YouTube)](https://youtube.com/watch?v=XA8W4Zhc5Ek)
- **Workflows:** `02_AI/02_N8N/AA_Overpass_Dagenham_LandUse.json`, `02_AI/02_N8N/AA_Reddit_Dagenham_Feedback.json`

### System 03: Latent Walk
- **Input:** 1,208 images of industrial ruins and rewilding vegetation
- **Method:** PCA → t-SNE embedding; Stable Diffusion interpolation in ComfyUI
- **Output:** 90-frame material futures matrix (Concrete×DECAY, Brick×REWILD, Steel×REUSE)
- **Video:** [Latent Walk Video (YouTube)](https://youtu.be/XniW9sVhLH0)
- **Workflow:** `02_AI/03_LatentWalk/latent_walk_60frames.json`

### System 04: Pix2Pix (Conditional GAN)
- **Training data:** 60 floor plans (30 Experience Centres + 30 Recycling Factories)
- **Architecture:** Pix2Pix (Isola et al., 2017), Epochs: 480, λ_L1: 100
- **Inference:** Applied to Ford Stamping Plant outline (unseen during training)
- **Post-processing:** Delaunay triangulation + spatial consensus across stable epochs (220★, 360★, 420★)
- **Scripts:** `02_AI/04_Pix2Pix/train.py`, `02_AI/04_Pix2Pix/inference.py`

### System 05: Neural Cellular Automata
- **Based on:** Mordvintsev et al. (2020) "Growing Neural Cellular Automata"
- **Bridge:** Pix2Pix output → K-Means colour quantisation → 4-colour strategy map (DECAY 19.6%, REWILD 15%, REUSE 65.4%, DISMANTLE 7.46%)
- **Training:** 16 channels, 16,000 steps, 3×3 local perception
- **Key insight:** Identical rules + different initial noise → varied but valid outcomes (Simulation as Inquiry)
- **Scripts:** `02_AI/05_NeuralCA/decay_neural_ca.py`, `02_AI/05_NeuralCA/pix2pix_to_ca.py`

### Dependencies (AI Module)
```
Python 3.10+
torch >= 2.0
torchvision
scikit-learn
pandas
numpy
matplotlib
Pillow
```

---

## Module 03: Unreal Engine

**Tutor: Enriqueta Llabres-Valls**

- **Video:** [UE Niagara Systems Demo (YouTube)](https://youtu.be/oyOwx5gqYzE)
- **Project Files:** [Niagara_test2.zip (OneDrive)](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucbv512_ucl_ac_uk/IQAcYDZaTZBETae4gxXVY24-AcoZLthi7GDia0bszxxIPM4?e=0UsAl3)

### Niagara System 1: Industrial Smoke Plume
- SimpleSpriteBurst emitter, Lifetime 3–5s, Sprite Size 1000
- Modules: Scale Color (white→grey curve), Drag (2.0), Acceleration Force, Scale Sprite Size
- **Material:** T_Smoke_SubUV_01_Mat — Translucent, SubUV flipbook, Particle Color × Texture → Emissive

### Niagara System 2: Spark / Ember Burst
- SimpleSpriteBurst, 500 particles, Lifetime 0.5–3s
- Modules: Add Velocity In Cone (300–500, 45°), Curl Noise Force, Gravity, Ray Traced Collision
- **Material:** flame02_Mat — Translucent, Particle Color × 100 → Emissive (HDR bloom), Alpha-masked flame atlas

---

## References

### Data Sources
- World Resources Institute (2023). *Global Power Plant Database.* https://datasets.wri.org/
- European Commission JRC (2024). *UCDB Health Emissions R2024A.* https://human-settlement.emergency.copernicus.eu/
- United Nations DESA (2024). *World Cities 1950–2035.* https://population.un.org/wup/
- NAEI (2021). *UK Gridded Emissions 1×1 km.* https://naei.beis.gov.uk/
- British Geological Survey. *Normal Background Concentrations in Soils, England.* https://www.bgs.ac.uk/datasets/nbc/
- Environment Agency. *Flood Alert Areas — England.* https://environment.data.gov.uk/
- Natural England. *Sensitive Areas.* https://naturalengland-defra.opendata.arcgis.com/
- Ordnance Survey. *OS Open Roads & OpenMap Local.* https://www.ordnancesurvey.co.uk/
- OpenAQ (2024). *Global Air Quality Monitoring Data.* https://openaq.org/
- LBBD (2024). *Brownfield Land Register.* https://www.lbbd.gov.uk/
- ONS (2021). *Census 2021 — Usual Residents by Output Area.* https://www.ons.gov.uk/

### Academic References
- Batty, M. (2005). *Cities and Complexity.* MIT Press.
- Isola, P. et al. (2017). 'Image-to-Image Translation with Conditional Adversarial Networks.' *CVPR.* https://arxiv.org/abs/1611.07004
- Mordvintsev, A. et al. (2020). 'Growing Neural Cellular Automata.' *Distill.* https://distill.pub/2020/growing-ca/
- Rombach, R. et al. (2022). 'High-Resolution Image Synthesis with Latent Diffusion Models.' *CVPR.* https://arxiv.org/abs/2112.10752
- Gilpin, W. (2019). 'Cellular Automata as Convolutional Neural Networks.' *Physical Review E.* https://arxiv.org/abs/1809.02942

### Software
- Esri (2024). ArcGIS Pro 3.6. https://www.esri.com/
- Epic Games (2024). Unreal Engine 5.4. https://www.unrealengine.com/
- ComfyUI (2024). https://github.com/comfyanonymous/ComfyUI
- n8n GmbH (2024). https://n8n.io/
- Huang, W. (2025–2026). RC18 Generative AI Module. https://github.com/WilliamShengYangHuang/RC18_GenAI

---

## License

This work is submitted as part of academic requirements at UCL Bartlett School of Architecture. All data sources are open-access and cited above. Code is provided for academic reference.
