# -*- coding: utf-8 -*-
"""Clustering pictures for the v8 solution (my_solution8/solution.py).

v8 = v7 with two pin fixes found by the test-size simulation
(dev_v8/sim_test_eval.py, 80/20 pseudo-test):
  - ds04: winner PINNED to GaussianMixture (k stays pinned at 3)
  - ds10: k PINNED to 2 + winner PINNED to SpectralClustering
All pictures are saved into dev_v8/:
  dev_v8/ds04_ds10_pin_fixes.png   - ds04/ds10: v7 vs v8 labels (Z-space)
                                     + raw-PCA panel with v8 labels
  dev_v8/fig1_all_24_grid.png      - all 24 datasets, top-2 PCA dims of Z-space,
                                     color = predicted cluster (v8)
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
os.makedirs('dev_v8', exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'axes.titlesize': 10})

# ---------- load data + v7/v8 solutions ----------
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}
DS = sorted(groups)

mods = {}
for tag in ('v7', 'v8'):
    spec = importlib.util.spec_from_file_location('sol_%s' % tag,
                                                 'my_solution%s/solution.py' % tag[1])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    mods[tag] = m
print('loaded v7 + v8')


def sc_pca2(X):
    """Same 2-D embedding the ring/blob branch fits in (StandardScaler + PCA)."""
    return PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler().fit_transform(np.nan_to_num(X)))


# ---------- ds04/ds10: v7 vs v8 pin fixes ----------
def compare_fig(ds_list, fname):
    m7, m8 = mods['v7'], mods['v8']
    ncols = 3
    fig, axes = plt.subplots(len(ds_list), ncols,
                             figsize=(4.6 * ncols, 4.0 * len(ds_list)))
    if len(ds_list) == 1:
        axes = np.atleast_1d(axes)[None, :]
    for r, ds in enumerate(ds_list):
        X = groups[ds]
        lab7, info7 = m7._run(X, ds)
        lab8, info8 = m8._run(X, ds)
        Z7 = m7._reduce(m7._clean(X), m7.NORM_PRIOR.get(ds, 'minmax'))
        Z8 = m8._reduce(m8._clean(X), m8.NORM_PRIOR.get(ds, 'minmax'))
        s7 = silhouette_score(Z7, lab7) if len(np.unique(lab7)) > 1 else float('nan')
        s8 = silhouette_score(Z8, lab8) if len(np.unique(lab8)) > 1 else float('nan')
        panels = [
            (Z7, lab7, '%s v7: %s k=%s | sil %.3f' % (ds, info7['winner'], info7['k_used'], s7)),
            (Z8, lab8, '%s v8: %s k=%s | sil %.3f' % (ds, info8['winner'], info8['k_used'], s8)),
        ]
        for c_, (P, lab, title) in enumerate(panels):
            ax = axes[r, c_]
            ax.scatter(P[:, 0], P[:, 1], c=lab, cmap='tab10', s=5, edgecolors='none')
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        # raw-PCA panel with v8 labels
        P = sc_pca2(X)
        ax = axes[r, 2]
        ax.scatter(P[:, 0], P[:, 1], c=lab8, cmap='tab10', s=5, edgecolors='none')
        ax.set_title('%s raw PCA-2, v8 labels (%s k=%s)' % (ds, info8['winner'], info8['k_used']), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('v7 vs v8 on %s (Z-space per version; raw panel = StandardScaler+PCA-2)' % '/'.join(ds_list), y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)

compare_fig(['ds04', 'ds10'], 'dev_v8/ds04_ds10_pin_fixes.png')

# ---------- fig: all 24 datasets grid (v8) ----------
m8 = mods['v8']
n = len(DS)
ncols, nrows = 4, (n + 3) // 4
fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows))
axes = np.atleast_1d(axes).ravel()
for i, ds in enumerate(DS):
    ax = axes[i]
    X = groups[ds]
    lab, info = m8._run(X, ds)
    Z = m8._reduce(m8._clean(X), m8.NORM_PRIOR.get(ds, 'minmax'))
    ax.scatter(Z[:, 0], Z[:, 1], c=lab, cmap='tab10', s=6, edgecolors='none')
    ax.set_title('%s | %s k=%s' % (ds, str(info.get('winner'))[:14], info.get('k_used')), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for j in range(len(DS), len(axes)):
    axes[j].axis('off')
fig.suptitle('v8: all 24 datasets, top-2 PCA dims of Z-space, color = predicted cluster', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(os.path.join('dev_v8', 'fig1_all_24_grid.png'), bbox_inches='tight')
plt.close(fig)
print('saved fig1 -> dev_v8/fig1_all_24_grid.png')
