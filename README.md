```
    ____  _         ____            __                 
   / __ \(_)__     / __ \____  ____/ /_ _  __________ 
  / /_/ / / _ \   / /_/ / __ \/ __/ __/ / / / ___/ _ \
 / ____/ / /__/  / ____/ /_/ (__  ) /_/ /_/ / /  /  __/
/_/   /_/\___/  /_/    \____/____/\__/\__,_/_/   \___/ 
```

# Multi-view Pig Posture Recognition

**Team:** Yash Verdhan, Aditya Singh Sengar  
**Competition:** [Multi-view Pig Posture Recognition](https://www.kaggle.com/competitions/multi-view-pig-posture-recognition) (Kaggle)  
**Best Public LB:** 0.873 (Macro F1)  
**Task:** Classify cropped pig images into 5 postures: Lateral Lying Left, Lateral Lying Right, Sitting, Standing, Sternal Lying

---

## Table of Contents

1. [Approach Overview](#approach-overview)  
2. [Data Pipeline](#data-pipeline)  
3. [Model A: EVA02-Large (K-Fold)](#model-a-eva02-large-k-fold)  
4. [Model B: EVA02-Enormous (Multi-Seed)](#model-b-eva02-enormous-multi-seed)  
5. [Inference and TTA](#inference-and-tta)  
6. [Ensembling Strategy](#ensembling-strategy)  
7. [What Did Not Work](#what-did-not-work)  
8. [Repository Structure](#repository-structure)  
9. [How to Reproduce](#how-to-reproduce)  
10. [Hardware and Environment](#hardware-and-environment)  
11. [Dependencies](#dependencies)

---

## Approach Overview

The final submission is a cross-architecture ensemble of two model families:

- **EVA02-Large** (304M params) trained with camera-based K-Fold cross-validation (3 folds)
- **EVA02-Enormous** (4.4B params) trained with LP-FT across 4 independent seeds

Each model produces softmax probability vectors via 8-way geometric Test-Time Augmentation.
The final prediction is a geometric mean of the two families' ensemble probabilities,
weighted 35% Large and 65% Enormous.

```
                         TRAINING
    ====================================================

    EVA02-Large (304M)              EVA02-Enormous (4.4B)
    K-Fold CV, 3 folds              LP-FT, 4 seeds
      |-- Fold 0                      |-- Seed 42
      |-- Fold 2                      |-- Seed 62
      '-- Fold 3                      |-- Seed 82
                                      '-- Seed 120

                         INFERENCE
    ====================================================

    Each model: 8-way geometric TTA
    (identity, HFlip, VFlip, HV, R90, R180, R270, R90+HFlip)
    with LatL <-> LatR probability swap on mirror augmentations

                         ENSEMBLE
    ====================================================

    Large folds  --> arithmetic mean --> large_probs.npy
    Enormous seeds -> arithmetic mean --> enormous_probs.npy

    Final = geometric_mean(0.35 * Large, 0.65 * Enormous)
          --> submission.csv
```

---

## Data Pipeline

The competition provides bounding-box annotations for pigs across multiple camera views.
We pre-crop all pig instances into individual images using the provided bounding boxes with
no extra padding (padding factor = 0). All crops are stored in a flat directory alongside
a master CSV that tracks each crop's class, camera, pen, and split assignment.

**Validation strategy:** Instead of a random train/val split, we hold out images from the
3 camera angles that match the test set distribution (`pen1_tur_cam1`, `pen2_orb_cam2`,
`pen2_tur_cam2`). This gives a validation set that closely mirrors the domain shift present
in test data, producing validation scores that better predict leaderboard performance than
a random split would.

**Class imbalance:** The dataset is heavily skewed toward Standing and Sternal Lying.
We use a WeightedRandomSampler to balance class representation within each training batch,
with an additional 3x upweight on the rare Sitting class.

**Label-aware horizontal flip:** During training, when an image is horizontally flipped,
we also swap the labels for Lateral Lying Left and Lateral Lying Right. This doubles
effective training data for the lying classes without introducing label noise.

---

## Model A: EVA02-Large (K-Fold)

**File:** `EVA02_Large_KFold.ipynb`

| Setting | Value |
|---------|-------|
| Backbone | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` |
| Parameters | 304M |
| Input size | 448 x 448 |
| Head | Single linear layer (1024 -> 5) |
| Optimizer | AdamW, LR=2e-5, WD=0.05, LLRD=0.85 |
| Scheduler | Linear warmup (3 epochs) + cosine decay |
| Label smoothing | 0.2 |
| Drop path | 0.15 |
| EMA decay | 0.9999 |
| Batch size | 32 |
| Epochs | 30 (early stopping, patience=10) |

**K-Fold split:** 5 camera-based folds, of which we used 3 (folds 0, 2, 3).
Folds 1 and 4 encountered errors during training and were excluded.

| Fold | Validation cameras | Val size |
|------|--------------------|----------|
| 0 | pen1_tur_cam1, pen1_orb_cam2 | ~3,176 |
| 2 | pen2_orb_cam2, pen1_orb_cam1 | ~1,564 |
| 3 | pen1_tur_cam2 | ~16,388 |

**Augmentations:** RandomResizedCrop, Rotation (15 deg), ColorJitter, GaussianBlur,
GaussNoise, Downscale, CoarseDropout, CLAHE.

**Output:** 3 sets of test probabilities (one per fold), arithmetically averaged to
produce `eva_large_5_fold_ensemble_probs.npy`. Standalone LB: 0.837.

---

## Model B: EVA02-Enormous (Multi-Seed)

**File:** `train_eva02_enormous_lpft.ipynb`

| Setting | Value |
|---------|-------|
| Backbone | `eva02_enormous_patch14_clip_224.laion2b_plus` |
| Parameters | 4.4B |
| Input size | 336 x 336 |
| Head | LayerNorm -> Dropout(0.3) -> Linear(1792->512) -> GELU -> Dropout(0.2) -> Linear(512->5) |
| Method | LP-FT (Linear Probing then Fine-Tuning) |
| LP stage | 3 epochs, head-only, LR=1e-3 |
| FT stage | Unfreeze last 24/64 blocks |
| Optimizer | AdamW, LR=1.5e-5, WD=0.05, LLRD=0.80 |
| Scheduler | Linear warmup (5 epochs) + cosine decay |
| Label smoothing | 0.05 |
| Drop path | 0.35 |
| EMA decay | 0.9995 |
| Batch size | 64 (gradient checkpointing enabled) |
| Epochs | 50 (early stopping, patience=25) |
| Time limit | 11.5 hours (Kaggle session constraint) |

**Why LP-FT:** With 4.4B parameters, directly fine-tuning from epoch 1 causes gradient
shattering in early layers. By first training only the classification head (linear probe),
the model learns a stable mapping from frozen features to our 5 classes. Only then do we
selectively unfreeze the top 24 blocks for fine-tuning, keeping the bottom 40 blocks
(low-level features) frozen.

**Why LLRD=0.80:** Layer-wise learning rate decay assigns progressively smaller learning
rates to deeper layers. With decay=0.80, the bottom unfrozen block trains at ~0.4% of
the head's learning rate. This preserves the strong pretrained representations while
allowing the top layers to specialize.

**Seeds:** 42, 62, 82, 120. Each seed trains independently. The 4 probability outputs
are arithmetically averaged to produce `eva_enormous_4seed_ensemble_probs.npy`.
Standalone LB: 0.864.

**Checkpoint system:** The training script includes resume-safe mid-epoch checkpointing
with bf16 tensor compression (reduces checkpoint size from 26 GB to ~10 GB). Lightweight
saves (no optimizer state) are used for mid-epoch saves to avoid disk overflow on Kaggle.

---

## Inference and TTA

**File:** `ensemble_enormous_seeds.py` (Enormous), `EVA02_Large_KFold.ipynb` (Large, built-in)

Both model families use the same 8-way geometric TTA at inference time:

| Mode | Transform | LatL/LatR swap? |
|------|-----------|-----------------|
| 0 | Identity | No |
| 1 | Horizontal flip | Yes |
| 2 | Vertical flip | No |
| 3 | HFlip + VFlip | Yes |
| 4 | 90 degree rotation | No |
| 5 | 180 degree rotation | No |
| 6 | 270 degree rotation | No |
| 7 | 90 degree + HFlip | Yes |

When a transform includes a horizontal flip, the probabilities for Lateral Lying Left
and Lateral Lying Right are swapped before averaging, since flipping the image reverses
which side the pig is lying on.

Final probabilities = arithmetic mean of all 8 augmented predictions.

For a detailed breakdown of the ensemble and inference pipeline, refer to
`ensemble_enormous_seeds.py` which contains the full implementation with
arithmetic, geometric, and rank-average strategies, along with automatic
strategy selection on the validation set.

---

## Ensembling Strategy

The final submission combines both model families:

```
final_probs = geometric_mean(
    large_probs   * 0.35,
    enormous_probs * 0.65
)
```

We tested multiple ensemble strategies (arithmetic mean, geometric mean at various
weight ratios, rank averaging) and found that geometric mean with 35/65 weighting
performed best on our validation set. The geometric mean rewards agreement between
models more heavily than arithmetic mean does, which helps when the two architectures
have complementary failure modes.

**Individual LB scores:**
- EVA02-Large K-Fold ensemble: 0.837
- EVA02-Enormous 4-seed ensemble: 0.864
- Cross-architecture geometric mean: 0.873

---

## What Did Not Work

We tried a lot of things over the course of this competition. Here is what failed
and, more importantly, why.

**LoRA on EVA02-Enormous:** Memory-efficient, but LoRA's low-rank updates lacked
the expressiveness to meaningfully adapt the CLIP representation to our specific domain
(top-down infrared/RGB pig crops). Full partial-block unfreezing was necessary.

**ExPLoRA domain pre-training (self-supervised MIM):** We ran LoRA-MAE with 75% patch
masking for 15 epochs on all 58K pig crops. Reconstructions looked stunning. But
downstream classification collapsed to 0.40 F1. The root cause: EVA02-Enormous is
CLIP-pretrained (semantic features for text-image alignment), not MAE-pretrained.
Forcing pixel reconstruction on CLIP features shifted the representation away from
classification. The beautiful reconstructions were actually the red flag.

**Lying specialist overlay:** We trained a separate 3-class lying specialist
(LatL/LatR/Sternal) and overlaid it on the generalist for ambiguous predictions.
LB dropped from 0.850 to 0.846. The specialist had overfit (peaked at epoch 1)
and its corrections introduced more errors than it fixed.

**SAM (Sharpness-Aware Minimization):** Halved effective training iterations within
the 11-hour Kaggle time limit. The model had not converged enough for flatness-seeking
to be beneficial.

**BEiT v2 / EVA02 Small / EfficientNet baselines:** Hit a plateau around 0.80 LB.
The fundamental limitation was model capacity, not training technique.

---

## Repository Structure

```
.
|-- EVA02_Large_KFold.ipynb          # Large model training (K-Fold CV)
|-- train_eva02_enormous_lpft.ipynb  # Enormous model training (LP-FT)
|-- ensemble_enormous_seeds.py       # 4-seed ensemble inference + TTA
|-- infer_eva02_enormous_lpft.py     # Single-seed inference + dashboard
|-- README.md                        # This file
```

**Model weights** are hosted separately (too large for GitHub):
[Google Drive Link -- INSERT YOUR LINK HERE]

---

## How to Reproduce

### If you have the trained checkpoints:

```bash
# 1. Run Enormous inference for each seed (produces test_probs.npy per seed)
#    Edit seed_checkpoints paths in ensemble_enormous_seeds.py, then:
python ensemble_enormous_seeds.py

# 2. Large fold probs are generated at the end of EVA02_Large_KFold.ipynb
#    (already saved as eva02_large_fold{0,2,3}_test_probs.npy)

# 3. Combine both into final submission
#    Average Large folds, average Enormous seeds, geometric mean 35/65
```

### If training from scratch:

```bash
# EVA02-Large: run the notebook 3 times (FOLD_ID = 0, 2, 3)
# EVA02-Enormous: run the notebook 4 times (seed = 42, 62, 82, 120)
# Then run ensemble scripts as above
```

**Required input data:**
- Cropped pig dataset (master.csv + crops/ directory)
- EVA02-Large pretrained weights (~1.2 GB)
- EVA02-Enormous pretrained weights (~17 GB)

---

## Hardware and Environment

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (102 GB VRAM) |
| Training time (Enormous) | ~11 hours per seed |
| Training time (Large) | ~3 hours per fold |
| Platform | Kaggle Notebooks |
| Time limit | 12 hours per session |

---

## Dependencies

- Python 3.10+
- PyTorch 2.x with CUDA
- timm (PyTorch Image Models)
- albumentations
- scikit-learn
- pandas, numpy, opencv-python
- matplotlib (for training dashboards)
