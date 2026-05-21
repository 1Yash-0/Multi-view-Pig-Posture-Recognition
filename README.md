```
 ╔═╗ ╦ ╔═╗   ╔═╗ ╔═╗ ╔═╗ ╔╦╗ ╦ ╦ ╦═╗ ╔═╗
 ╠═╝ ║ ║ ╦   ╠═╝ ║ ║ ╚═╗  ║  ║ ║ ╠╦╝ ║╣
 ╩   ╩ ╚═╝   ╩   ╚═╝ ╚═╝  ╩  ╚═╝ ╩╚═ ╚═╝
              Multi-view Recognition
```

# Multi-view Pig Posture Recognition

> **Kaggle Competition** · [Multi-view Pig Posture Recognition](https://www.kaggle.com/competitions/multi-view-pig-posture-recognition)

![Public LB](https://img.shields.io/badge/Public%20LB-0.873%20Macro%20F1-4ade9a?style=flat-square)
![Model](https://img.shields.io/badge/EVA02--Enormous-4.4B%20params-6366f1?style=flat-square)
![Model](https://img.shields.io/badge/EVA02--Large-304M%20params-3b82f6?style=flat-square)
![Framework](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)

Classify cropped pig images into **5 postures** across multiple camera views:
`Lateral Lying Left` · `Lateral Lying Right` · `Sitting` · `Standing` · `Sternal Lying`

**Team:** Yash Verdhan, Aditya Singh Sengar

---

## Table of Contents

- [Results](#results)
- [Approach Overview](#approach-overview)
- [Data Pipeline](#data-pipeline)
- [Model A — EVA02-Large (K-Fold)](#model-a--eva02-large-k-fold)
- [Model B — EVA02-Enormous (Multi-Seed)](#model-b--eva02-enormous-multi-seed)
- [Inference & TTA](#inference--tta)
- [Ensembling](#ensembling)
- [What Did Not Work](#what-did-not-work)
- [Repo Structure](#repo-structure)
- [Reproducing Results](#reproducing-results)
- [Dependencies](#dependencies)

---

## Results

| Model | Strategy | Public LB (Macro F1) |
|---|---|:---:|
| EVA02-Large | K-Fold (3 folds), arithmetic mean | 0.837 |
| EVA02-Enormous | 4 seeds, arithmetic mean | 0.864 |
| **Cross-arch Ensemble** | **Geometric mean, 35/65 weight** | **0.873** |

---

## Approach Overview

The final submission is a **cross-architecture ensemble** of two model families. Each model produces softmax probability vectors via **8-way geometric Test-Time Augmentation (TTA)**. The final prediction is a geometric mean of the two ensembles, weighted 35% Large and 65% Enormous.

```
                          TRAINING
     ═══════════════════════════════════════════════

     EVA02-Large (304M)              EVA02-Enormous (4.4B)
     K-Fold CV · 3 folds             LP-FT · 4 seeds
       ├── Fold 0                      ├── Seed 42
       ├── Fold 2                      ├── Seed 62
       └── Fold 3                      ├── Seed 82
                                       └── Seed 120

                          INFERENCE
     ═══════════════════════════════════════════════

     Each model → 8-way geometric TTA
     (identity, HFlip, VFlip, HV, R90, R180, R270, R90+HFlip)
     with LatL ↔ LatR probability swap on mirror augmentations

                          ENSEMBLE
     ═══════════════════════════════════════════════

     Large folds   → arithmetic mean → large_probs.npy
     Enormous seeds → arithmetic mean → enormous_probs.npy

     final = geometric_mean(0.35 × large_probs, 0.65 × enormous_probs)
```

---

## Data Pipeline

The competition provides bounding-box annotations across multiple camera views. We pre-crop all pig instances into individual images using the provided boxes with **zero padding**, stored in a flat directory alongside a master CSV tracking class, camera, pen, and split.

**Validation strategy** — instead of a random split, we hold out images from the 3 camera angles that match the test set distribution (`pen1_tur_cam1`, `pen2_orb_cam2`, `pen2_tur_cam2`). This gives a validation set that closely mirrors the domain shift in test data, producing scores that reliably predict leaderboard performance.

**Class imbalance** — the dataset is heavily skewed toward Standing and Sternal Lying. We use a `WeightedRandomSampler` to balance class representation per batch, with an additional **3× upweight** on the rare Sitting class.

**Label-aware horizontal flip** — when an image is H-flipped during training, we swap the labels for Lateral Lying Left and Lateral Lying Right. This doubles effective training data for the lying classes without introducing label noise.

---

## Model A — EVA02-Large (K-Fold)

**File:** `EVA02_Large_KFold.ipynb`

| Setting | Value |
|---|---|
| Backbone | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` |
| Parameters | 304M |
| Input size | 448 × 448 |
| Head | Linear (1024 → 5) |
| Optimizer | AdamW · LR 2e-5 · WD 0.05 · LLRD 0.85 |
| Scheduler | Linear warmup (3 ep) + cosine decay |
| Label smoothing | 0.2 |
| Drop path | 0.15 |
| EMA decay | 0.9999 |
| Batch size | 32 |
| Epochs | 30 (early stop · patience 10) |

**Augmentations:** RandomResizedCrop, Rotation ±15°, ColorJitter, GaussianBlur, GaussNoise, Downscale, CoarseDropout, CLAHE.

**K-Fold splits (camera-based, 5 total — folds 1 & 4 excluded due to training errors):**

| Fold | Validation Cameras | Val Size |
|:---:|---|:---:|
| 0 | pen1_tur_cam1, pen1_orb_cam2 | ~3,176 |
| 2 | pen2_orb_cam2, pen1_orb_cam1 | ~1,564 |
| 3 | pen1_tur_cam2 | ~16,388 |

Three fold outputs arithmetically averaged → `eva_large_5_fold_ensemble_probs.npy`

---

## Model B — EVA02-Enormous (Multi-Seed)

**File:** `EVA02_Enormous_Train_Infer.ipynb`

| Setting | Value |
|---|---|
| Backbone | `eva02_enormous_patch14_clip_224.laion2b_plus` |
| Parameters | 4.4B |
| Input size | 336 × 336 |
| Head | LayerNorm → Dropout(0.3) → Linear(1792→512) → GELU → Dropout(0.2) → Linear(512→5) |
| Method | LP-FT (Linear Probe → Fine-Tune) |
| LP stage | 3 ep · head-only · LR 1e-3 |
| FT stage | Unfreeze last 24/64 blocks |
| Optimizer | AdamW · LR 1.5e-5 · WD 0.05 · LLRD 0.80 |
| Scheduler | Linear warmup (5 ep) + cosine decay |
| Label smoothing | 0.05 |
| Drop path | 0.35 |
| EMA decay | 0.9995 |
| Batch size | 64 (gradient checkpointing) |
| Epochs | 50 (early stop · patience 25) |

**Why LP-FT?** With 4.4B parameters, direct fine-tuning from epoch 1 causes gradient shattering in early layers. Linear probing first stabilizes the classification head on frozen features. We then selectively unfreeze the top 24 blocks for fine-tuning, keeping the bottom 40 blocks (low-level features) frozen.

**Why LLRD = 0.80?** Layer-wise LR decay assigns progressively smaller learning rates to deeper layers — the bottom unfrozen block trains at ~0.4% of the head's LR. This preserves strong pretrained representations while allowing top layers to specialize.

**Seeds:** 42, 62, 82, 120 — each trained independently. Four outputs arithmetically averaged → `eva_enormous_4seed_ensemble_probs.npy`

**Checkpointing:** Resume-safe mid-epoch checkpointing with bf16 tensor compression (26 GB → ~10 GB). Lightweight saves (no optimizer state) for mid-epoch to avoid disk overflow on Kaggle.

---

## Inference & TTA

**Files:** `ensemble_enormous_seeds.py` (Enormous), `EVA02_Large_KFold.ipynb` (Large, built-in)

Both families use the same **8-way geometric TTA**. On mirror augmentations, `LatL ↔ LatR` probabilities are swapped before averaging since flipping reverses which side the pig is lying on.

| Mode | Transform | LatL/LatR Swap |
|:---:|---|:---:|
| 0 | Identity | — |
| 1 | Horizontal flip | ✓ |
| 2 | Vertical flip | — |
| 3 | HFlip + VFlip | ✓ |
| 4 | Rotate 90° | — |
| 5 | Rotate 180° | — |
| 6 | Rotate 270° | — |
| 7 | Rotate 90° + HFlip | ✓ |

Final probabilities = arithmetic mean of all 8 augmented predictions.

---

## Ensembling

```python
final_probs = geometric_mean(
    large_probs    * 0.35,   # EVA02-Large K-Fold ensemble
    enormous_probs * 0.65    # EVA02-Enormous 4-seed ensemble
)
```

We tested arithmetic mean, geometric mean at various weight ratios, and rank averaging. Geometric mean with **35/65 weighting** performed best on our validation set. The geometric mean rewards agreement between models more heavily than arithmetic mean, which helps when two architectures have complementary failure modes.

---

## What Did Not Work

**LoRA on EVA02-Enormous** — Memory-efficient, but low-rank updates lacked the expressiveness to adapt the CLIP representation to our domain (top-down infrared/RGB pig crops). Full partial-block unfreezing was necessary.

**ExPLoRA domain pre-training (self-supervised MIM)** — LoRA-MAE with 75% patch masking for 15 epochs produced stunning reconstructions but downstream classification collapsed to 0.40 F1. EVA02-Enormous is CLIP-pretrained (semantic features for text-image alignment), not MAE-pretrained. Forcing pixel reconstruction on CLIP features shifted the representation away from classification. The beautiful reconstructions were the red flag.

**Lying specialist overlay** — A separate 3-class lying specialist (LatL/LatR/Sternal) overlaid on the generalist for ambiguous predictions. LB dropped from 0.850 to 0.846. The specialist had overfit (peaked at epoch 1) and its corrections introduced more errors than it fixed.

**Sharpness-Aware Minimization (SAM)** — Halved effective training iterations. The model had not converged enough for flatness-seeking to be beneficial.

**BEiT v2 / EVA02-Small / EfficientNet baselines** — All plateaued around 0.80 LB. The fundamental limitation was model capacity, not training technique.

---

## Repo Structure

```
.
├── EVA02_Enormous_Train_Infer.ipynb    # Model B: Enormous training (LP-FT) + Inference
├── EVA02_Large_KFold.ipynb             # Model A: Large training (K-Fold CV)
├── README.md                           # Documentation
├── ensemble_enormous_seeds.ipynb       # 4-seed ensemble inference + TTA
├── ensemble_large_enormous.py          # Final geometric mean ensemble script
├── inference_and_ensembling_related.md # Additional inference documentation
└── weights_links.txt                   # Links to model checkpoints
```

> **Model weights** are hosted on Google Drive (too large for GitHub). See `weights_links.txt` for the links.

---

## Reproducing Results

### From trained checkpoints

```bash
# 1. Run Enormous inference for each seed
#    Edit seed_checkpoints paths in ensemble_enormous_seeds.ipynb and run all cells

# 2. Large fold probs are generated at the end of EVA02_Large_KFold.ipynb
#    Saved as: eva02_large_fold{0,2,3}_test_probs.npy

# 3. Average Large folds + average Enormous seeds → geometric mean 35/65
#    Run: python ensemble_large_enormous.py
```

### From scratch

```bash
# EVA02-Large: run notebook 3× with FOLD_ID = 0, 2, 3
# EVA02-Enormous: run notebook 4× with seed = 42, 62, 82, 120
# Then run ensemble scripts as above
```

**Required inputs:**
- Cropped pig dataset (`master.csv` + `crops/` directory)
- EVA02-Large pretrained weights (~1.2 GB)
- EVA02-Enormous pretrained weights (~17 GB)

---

## Dependencies

```
Python        3.10+
PyTorch       2.x with CUDA
timm          PyTorch Image Models (EVA02 backbones)
albumentations
scikit-learn
pandas
numpy
opencv-python
matplotlib
```

---

*GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (102 GB VRAM) · Platform: Kaggle Notebooks*
