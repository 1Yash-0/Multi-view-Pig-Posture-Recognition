"""
EVA Large + EVA Enormous -- Cross-Architecture Ensemble
======================================================
Combines:
  - EVA02-Large  5-fold ensemble probs  (0.837 LB)
  - EVA02-Enormous 4-seed ensemble probs (0.864 LB)

Tries multiple strategies and saves best submission.
"""

import os
import numpy as np
import pandas as pd

FOLDER = os.path.dirname(os.path.abspath(__file__))
CN = {0: 'LatL', 1: 'LatR', 2: 'Sit', 3: 'Stand', 4: 'Stern'}

# ---- Load pre-ensembled probabilities ----
large_path = os.path.join(FOLDER, 'eva_large_5_fold_ensemble_probs.npy')
enormous_path = os.path.join(FOLDER, 'eva_enormous_4seed_ensemble_probs.npy')

large_probs = np.load(large_path)
enormous_probs = np.load(enormous_path)

print(f'  EVA Large  5-fold: {large_probs.shape}')
print(f'  EVA Enormous 4-seed: {enormous_probs.shape}')
assert large_probs.shape == enormous_probs.shape, 'Shape mismatch!'

N = large_probs.shape[0]

# ---- Load row_ids from a submission CSV ----
sub_csv = os.path.join(FOLDER, 'T2_eva02e_lpft_submission.csv')
if not os.path.exists(sub_csv):
    for f in os.listdir(FOLDER):
        if f.startswith('T2_') and f.endswith('.csv'):
            sub_csv = os.path.join(FOLDER, f)
            break

sub_template = pd.read_csv(sub_csv)
assert len(sub_template) == N, f'CSV rows={len(sub_template)} vs probs={N}'
row_ids = sub_template['row_id'].values

# ---- Per-model analysis ----
large_preds = large_probs.argmax(1)
enormous_preds = enormous_probs.argmax(1)
agree = (large_preds == enormous_preds).sum()
print(f'\n  Agreement: {agree}/{N} ({100*agree/N:.1f}%)')
print(f'  Disagreement: {N - agree} samples ({100*(N-agree)/N:.1f}%)')

print(f'\n  Per-class distribution:')
print(f'  {"Class":<8} {"Large":>8} {"Enormous":>10}')
for c in range(5):
    nl = (large_preds == c).sum()
    ne = (enormous_preds == c).sum()
    print(f'  {CN[c]:<8} {nl:>8} {ne:>10}')

# ---- Ensemble strategies ----
strategies = {}

# 1. Simple average (equal weight)
avg_probs = (large_probs + enormous_probs) / 2.0
strategies['50-50 avg'] = avg_probs

# 2. Weighted: favor Enormous (better model gets more weight)
for w_e in [0.55, 0.60, 0.65, 0.70, 0.75]:
    w_l = 1.0 - w_e
    wp = w_l * large_probs + w_e * enormous_probs
    strategies[f'{int(w_l*100)}-{int(w_e*100)} (L-E)'] = wp

# 3. Geometric mean (rewards confident predictions)
eps = 1e-8
log_large = np.log(large_probs + eps)
log_enormous = np.log(enormous_probs + eps)

geo_probs = np.exp((log_large + log_enormous) / 2.0)
geo_probs /= geo_probs.sum(axis=1, keepdims=True)
strategies['geo-mean 50-50'] = geo_probs

for w_e in [0.60, 0.65, 0.70]:
    w_l = 1.0 - w_e
    geo_w = np.exp(w_l * log_large + w_e * log_enormous)
    geo_w /= geo_w.sum(axis=1, keepdims=True)
    strategies[f'geo {int(w_l*100)}-{int(w_e*100)}'] = geo_w

# 4. Power mean (p=2, emphasizes confident predictions)
for w_e in [0.50, 0.60, 0.65]:
    w_l = 1.0 - w_e
    p = 2.0
    pw = (w_l * large_probs**p + w_e * enormous_probs**p) ** (1.0/p)
    pw /= pw.sum(axis=1, keepdims=True)
    strategies[f'power2 {int(w_l*100)}-{int(w_e*100)}'] = pw

# ---- Compare all strategies ----
print(f'\n{"="*70}')
print(f'  ENSEMBLE STRATEGY COMPARISON')
print(f'{"="*70}')
print(f'  {"Strategy":<22} {"HighConf>95%":>12} {"LowConf<50%":>12} {"AvgConf":>8} {"Changed_vs_E":>14}')
print(f'  {"-"*68}')

best_name = None
best_high_conf = -1

for name, probs in strategies.items():
    preds = probs.argmax(1)
    max_conf = probs.max(axis=1)
    high = (max_conf > 0.95).sum()
    low = (max_conf < 0.50).sum()
    avg = max_conf.mean()
    changed = (preds != enormous_preds).sum()

    marker = ''
    if avg > best_high_conf:
        best_high_conf = avg
        best_name = name
        marker = ' <-- best'

    print(f'  {name:<22} {high:>12} {low:>12} {avg:>8.4f} {changed:>14}{marker}')

# ---- Save multiple submissions ----
print(f'\n{"="*70}')
print(f'  SAVING SUBMISSIONS')
print(f'{"="*70}')

saved_files = []
for name, probs in strategies.items():
    preds = probs.argmax(1)
    tag = name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')

    sub = pd.DataFrame({'row_id': row_ids, 'class_id': preds})
    csv_path = os.path.join(FOLDER, f'T2_large_enormous_{tag}.csv')
    sub.to_csv(csv_path, index=False)

    npy_path = os.path.join(FOLDER, f'large_enormous_{tag}_probs.npy')
    np.save(npy_path, probs.astype(np.float32))

    saved_files.append((name, csv_path))

    dist = {CN[k]: v for k, v in zip(*np.unique(preds, return_counts=True))}
    print(f'  OK {name:<22} -> {os.path.basename(csv_path)}')
    print(f'    Dist: {" | ".join(f"{k}:{v}" for k,v in dist.items())}')

# ---- Disagreement analysis ----
print(f'\n{"="*70}')
print(f'  DISAGREEMENT ANALYSIS (where Large != Enormous)')
print(f'{"="*70}')

disagree_mask = large_preds != enormous_preds
disagree_idx = np.where(disagree_mask)[0]

print(f'  Total disagreements: {len(disagree_idx)}')
print(f'\n  Confusion between architectures (Large -> Enormous):')
print(f'  {"Large>":8}', end='')
for c in range(5):
    print(f'  {CN[c]:>6}', end='')
print()
for c_l in range(5):
    print(f'  {CN[c_l]:<8}', end='')
    for c_e in range(5):
        n = ((large_preds[disagree_idx] == c_l) & (enormous_preds[disagree_idx] == c_e)).sum()
        if n > 0:
            print(f'  {n:>6}', end='')
        else:
            print(f'  {".":>6}', end='')
    print()

# ---- Confidence on disagreements ----
print(f'\n  On disagreements:')
large_conf_dis = large_probs[disagree_mask].max(axis=1)
enormous_conf_dis = enormous_probs[disagree_mask].max(axis=1)
print(f'    Large    avg conf: {large_conf_dis.mean():.3f}')
print(f'    Enormous avg conf: {enormous_conf_dis.mean():.3f}')

e_wins = (enormous_conf_dis > large_conf_dis).sum()
l_wins = (large_conf_dis > enormous_conf_dis).sum()
print(f'    Enormous more confident: {e_wins} ({100*e_wins/len(disagree_idx):.1f}%)')
print(f'    Large more confident:    {l_wins} ({100*l_wins/len(disagree_idx):.1f}%)')

print(f'\n{"="*70}')
print(f'  RECOMMENDATION')
print(f'{"="*70}')
print(f'  Best strategy by avg confidence: {best_name}')
print(f'  Submit ALL variants to Kaggle and pick the best LB score!')
print(f'  Recommended to try first: 35-65 (L-E) or geo 35-65')
print(f'{"="*70}')
print('  Done.')
