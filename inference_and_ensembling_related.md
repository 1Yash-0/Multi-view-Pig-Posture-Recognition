# 🎯 Minimal Pipeline: Training → Inference → Ensemble

## 📊 Seeds Used Summary

### **EVA02-LARGE Model**
```
❌ NO SEEDS USED
✅ K-FOLD CROSS-VALIDATION USED

Folds used: 0, 2, 3 (out of 5)
- Fold 0: Camera split (test cam + medium cam)
- Fold 2: Camera split (test cam + light cam)
- Fold 3: Heavy training camera
(Folds 1 & 4 had errors, excluded)

Result: eva_large_5_fold_ensemble_probs.npy (LB: 0.837)
```

### **EVA02-ENORMOUS Model**
```
✅ 4 SEEDS USED:
   - Seed 42
   - Seed 62
   - Seed 82
   - Seed 120

Each seed trained independently with 8-way TTA

Result: eva_enormous_4seed_ensemble_probs.npy (LB: 0.864)
```

---

## 📁 3 Core Files You Need

### **1️⃣ LARGE Training** (K-Fold)
**File:** `train_eva_enormous.py` (but configured for EVA02-Large)

**Key Configuration:**
```python
FOLD_ID = 0  # Change to 0, 1, 2, 3, 4 for each run
             # Only use folds: 0, 2, 3

# Camera-based fold splits (K-fold)
FOLD_CAMERA_GROUPS = {
    0: ['pen1_tur_cam1', 'pen1_orb_cam2'],      # ~3,176 val
    2: ['pen2_orb_cam2', 'pen1_orb_cam1'],      # ~1,564 val
    3: ['pen1_tur_cam2'],                        # ~16,388 val
}

# Model
model_name = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
img_size = 448
num_classes = 5

# Hyperparameters (UNCHANGED from 0.837 LB run)
lr = 2e-5
layer_decay = 0.85
weight_decay = 0.05
label_smoothing = 0.2
drop_path_rate = 0.15
ema_decay = 0.9999
epochs = 30
batch_size = 32

# Output
output_prefix = f'eva02_large_fold{FOLD_ID}'
```

**Steps to run:**
```bash
# Run for each fold: 0, 2, 3 (on different machines/time)
FOLD_ID=0 python train_eva_enormous.py
FOLD_ID=2 python train_eva_enormous.py
FOLD_ID=3 python train_eva_enormous.py

# Output per fold:
# - eva02_large_fold0_best.pth
# - eva02_large_fold0_test_probs.npy
# - eva02_large_fold0_val_probs.npy
# (same for fold 2, 3)
```

**Outputs Generated:**
```
✓ eva02_large_fold0_test_probs.npy (with 8-way TTA)
✓ eva02_large_fold2_test_probs.npy (with 8-way TTA)
✓ eva02_large_fold3_test_probs.npy (with 8-way TTA)
✓ eva02_large_fold0_best.pth
✓ eva02_large_fold2_best.pth
✓ eva02_large_fold3_best.pth
```

---

### **2️⃣ ENORMOUS Training + Inference** (4 Seeds)
**File:** `eva_enormous_inference.py`

**Key Configuration:**
```python
# 4 separate runs needed (different seeds)
SEEDS = [42, 62, 82, 120]

# Model
model_name = 'eva02_enormous_patch14_448.mim_m38m_ft_in22k_in1k'
img_size = 336
num_classes = 5

# For each seed:
seed = 42  # Change to 62, 82, 120

# Linear Probe Fine-Tuning (LP-FT)
checkpoint = f'eva02e_lpft_best_seed{seed}.pth'

# Output
output_prefix = f'eva02e_lpft_seed{seed}'
```

**Steps to run:**
```bash
# Run for each seed: 42, 62, 82, 120 (can run in parallel)
python eva_enormous_inference.py --seed 42
python eva_enormous_inference.py --seed 62
python eva_enormous_inference.py --seed 82
python eva_enormous_inference.py --seed 120

# Output per seed:
# - eva02e_lpft_seed42_test_probs.npy (with 8-way TTA)
# - eva02e_lpft_seed62_test_probs.npy (with 8-way TTA)
# - eva02e_lpft_seed82_test_probs.npy (with 8-way TTA)
# - eva02e_lpft_seed120_test_probs.npy (with 8-way TTA)
```

**Outputs Generated:**
```
✓ eva02e_lpft_seed42_test_probs.npy
✓ eva02e_lpft_seed62_test_probs.npy
✓ eva02e_lpft_seed82_test_probs.npy
✓ eva02e_lpft_seed120_test_probs.npy
```

---

### **3️⃣ FOLD + SEED Ensemble** (Pre-ensemble each model family)

#### **Step A: Average LARGE Folds**
**File:** `inference_5fold_ensemble.py`

**What it does:**
```python
# Load 3 fold predictions
fold_probs_0 = np.load('eva02_large_fold0_test_probs.npy')
fold_probs_2 = np.load('eva02_large_fold2_test_probs.npy')
fold_probs_3 = np.load('eva02_large_fold3_test_probs.npy')

# Average the 3 folds
eva_large_5_fold_ensemble_probs = (fold_probs_0 + fold_probs_2 + fold_probs_3) / 3
np.save('eva_large_5_fold_ensemble_probs.npy', eva_large_5_fold_ensemble_probs)
```

**Output:**
```
✓ eva_large_5_fold_ensemble_probs.npy (shape: 11708 x 5)
```

#### **Step B: Average ENORMOUS Seeds**
**File:** `ensemble_4seed_eva enormous.py`

**What it does:**
```python
seeds = {
    42:  'eva02e_lpft_test_probs_42.npy',
    62:  'eva02e_lpft_test_probs_62.npy',
    82:  'eva02e_lpft_test_probs_82.npy',
    120: 'eva02e_lpft_test_probs_120.npy',
}

# Load all 4 seeds
all_probs = [np.load(path) for path in seeds.values()]

# Average the 4 seeds
eva_enormous_4seed_ensemble_probs = np.stack(all_probs).mean(axis=0)
np.save('eva_enormous_4seed_ensemble_probs.npy', eva_enormous_4seed_ensemble_probs)
```

**Output:**
```
✓ eva_enormous_4seed_ensemble_probs.npy (shape: 11708 x 5)
✓ T2_enormous_4seed_ensemble.csv
```

---

### **4️⃣ FINAL Cross-Architecture Ensemble**
**File:** `ensemble_large_enormous.py`

**What it does:**
```python
# Load pre-ensembled probabilities
large_probs = np.load('eva_large_5_fold_ensemble_probs.npy')      # 0.837 LB
enormous_probs = np.load('eva_enormous_4seed_ensemble_probs.npy') # 0.864 LB

# Multiple strategies tested:
strategies = {
    '50-50 avg': (large_probs + enormous_probs) / 2,
    'geo 35-65': geometric_mean(large_probs, enormous_probs, weights=[0.35, 0.65]),
    'geo 40-60': geometric_mean(large_probs, enormous_probs, weights=[0.40, 0.60]),
    # ... many more
}

# Best result (your CSV):
geo_35_65_probs = strategies['geo 35-65']
geo_35_65_preds = geo_35_65_probs.argmax(1)

# Save submission
sub = pd.DataFrame({'row_id': row_ids, 'class_id': geo_35_65_preds})
sub.to_csv('T2_large_enormous_geo_35_65.csv', index=False)
```

**Output:**
```
✓ T2_large_enormous_geo_35_65.csv  ← YOUR FINAL SUBMISSION
✓ T2_large_enormous_*.csv (all variants tested)
✓ large_enormous_geo_35_65_probs.npy (probabilities)
```

---

## 🔄 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                           │
└─────────────────────────────────────────────────────────────┘

EVA02-LARGE (K-Fold)              EVA02-ENORMOUS (4 Seeds)
  ├─ Fold 0                          ├─ Seed 42
  ├─ Fold 2                          ├─ Seed 62
  └─ Fold 3                          ├─ Seed 82
                                     └─ Seed 120

        ↓                                    ↓
   3 checkpoint files              4 checkpoint files


┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PHASE                           │
└─────────────────────────────────────────────────────────────┘

eva02_large_fold{0,2,3}_test_probs.npy    eva02e_lpft_seed{42,62,82,120}_test_probs.npy
         (3 files × 8-way TTA)                    (4 files × 8-way TTA)

        ↓                                    ↓

┌─────────────────────────────────────────────────────────────┐
│               FOLD/SEED ENSEMBLE PHASE                      │
└─────────────────────────────────────────────────────────────┘

  LARGE: Average 3 folds            ENORMOUS: Average 4 seeds
         ↓                                    ↓
eva_large_5_fold_ensemble_probs.npy  eva_enormous_4seed_ensemble_probs.npy
        (11708 × 5)                        (11708 × 5)

        ↓                                    ↓

┌─────────────────────────────────────────────────────────────┐
│          CROSS-ARCHITECTURE ENSEMBLE PHASE                  │
└─────────────────────────────────────────────────────────────┘

Geometric Mean: 35% LARGE + 65% ENORMOUS
         ↓
   T2_large_enormous_geo_35_65.csv  ← FINAL SUBMISSION
   (11708 rows × 2 cols: row_id, class_id)
```

---

## 📝 File Checklist

**Training Output Files (Required for Inference):**
```
✓ eva02_large_fold0_best.pth
✓ eva02_large_fold2_best.pth
✓ eva02_large_fold3_best.pth
✓ eva02e_lpft_best_seed42.pth
✓ eva02e_lpft_best_seed62.pth
✓ eva02e_lpft_best_seed82.pth
✓ eva02e_lpft_best_seed120.pth
```

**Inference Output Files (Intermediate):**
```
✓ eva02_large_fold0_test_probs.npy
✓ eva02_large_fold2_test_probs.npy
✓ eva02_large_fold3_test_probs.npy
✓ eva02e_lpft_test_probs_42.npy
✓ eva02e_lpft_test_probs_62.npy
✓ eva02e_lpft_test_probs_82.npy
✓ eva02e_lpft_test_probs_120.npy
```

**Pre-Ensemble Output Files:**
```
✓ eva_large_5_fold_ensemble_probs.npy
✓ eva_enormous_4seed_ensemble_probs.npy
```

**Final Output:**
```
✓ T2_large_enormous_geo_35_65.csv  ← SUBMISSION
```

---

## 🎯 Reproduction (Minimal Steps)

### **If you have all checkpoints:**
```bash
# Step 1: Generate inference predictions
python eva_enormous_inference.py --seed 42
python eva_enormous_inference.py --seed 62
python eva_enormous_inference.py --seed 82
python eva_enormous_inference.py --seed 120
python inference_5fold_ensemble.py

# Step 2: Ensemble seeds
python ensemble_4seed_eva_enormous.py

# Step 3: Final ensemble
python ensemble_large_enormous.py

# Output: T2_large_enormous_geo_35_65.csv ✓
```

### **If you have pre-computed probabilities:**
```bash
# Just run final ensemble
python ensemble_large_enormous.py

# Output: T2_large_enormous_geo_35_65.csv ✓
```

---

## 📊 Summary Table

| Component | Count | Type | Result LB |
|-----------|-------|------|-----------|
| **LARGE** | 3 | K-Fold | 0.837 |
| **ENORMOUS** | 4 | Seeds | 0.864 |
| **Ensemble** | 1 | Geo Mean (35-65) | ~0.87+ |

**Key Points:**
- ✅ LARGE = **0 seeds**, uses **K-Fold** (folds: 0, 2, 3)
- ✅ ENORMOUS = **4 seeds** (42, 62, 82, 120)
- ✅ Final = **Geometric mean** combining both
- ✅ Each component uses **8-way TTA** (Test-Time Augmentation)

