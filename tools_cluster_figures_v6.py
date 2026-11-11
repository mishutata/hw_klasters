# -*- coding: utf-8 -*-
"""Clustering pictures for the v6 solution (my_solution6/solution.py).

v6 = v5 + ds18 pinned to k=4/Spectral, ds16/ds23 pinned to k=4/Spectral.
Saves:
  dev_v6/ds18_k4_spectral.png        - ds18: v5 (k=2) vs v6 (Spectral k=4)
  dev_v6/ds16_ds23_k4_spectral.png   - ds16/ds23: v5 ring/blob vs v6 Spectral k=4
  clustering_pictures_v6/fig1_all_24_grid.png - all datasets, Z-space, v6 labels
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

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
for d in ('dev_v6', 'clustering_pictures_v6'):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'axes.titlesize': 10})

# ---------- load data + v5/v6 solutions ----------
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}
DS = sorted(groups)

spec5 = importlib.util.spec_from_file_location('sol_v5', 'my_solution5/solution.py')
v5 = importlib.util.module_from_spec(spec5)
spec5.loader.exec_module(v5)
spec6 = importlib.util.spec_from_file_location('sol_v6', 'my_solution6/solution.py')
v6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(v6)
print('loaded v5 + v6')

raw_pca2 = {ds: PCA(n_components=2, random_state=42).fit_transform(
    np.nan_to_num(groups[ds])) for ds in DS}

# ---------- ds18: v5 (k=2) vs v6 (Spectral k=4) ----------
def compare_fig(ds_list, fname, ncols=4):
    nrows = len(ds_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.9 * nrows))
    if nrows == 1:
        axes = np.atleast_1d(axes)[None, :]
    for r, ds in enumerate(ds_list):
        X = groups[ds]
        lab5, info5 = v5._run(X, ds)
        lab6, info6 = v6._run(X, ds)
        Z = v6._reduce(v6._clean(X), v6.NORM_PRIOR.get(ds, 'minmax'))
        sil5 = silhouette_score(Z, lab5) if len(np.unique(lab5)) > 1 else float('nan')
        sil6 = silhouette_score(Z, lab6) if len(np.unique(lab6)) > 1 else float('nan')
        panels = [
            (Z, lab5, '%s v5: %s k=%s | sil %.3f' % (ds, info5['winner'], info5['k_used'], sil5)),
            (Z, lab6, '%s v6: %s k=%s | sil %.3f' % (ds, info6['winner'], info6['k_used'], sil6)),
            (raw_pca2[ds], lab5, 'raw PCA-2, v5 labels'),
            (raw_pca2[ds], lab6, 'raw PCA-2, v6 labels'),
        ]
        for c, (P, lab, title) in enumerate(panels):
            ax = axes[r, c]
            ax.scatter(P[:, 0], P[:, 1], c=lab, cmap='tab10', s=5, edgecolors='none')
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('v6 vs v5 on %s (Z-space = v6 norm+PCA; sil = silhouette in Z)' % '/'.join(ds_list), y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)

compare_fig(['ds18'], 'dev_v6/ds18_k4_spectral.png')
compare_fig(['ds16', 'ds23'], 'dev_v6/ds16_ds23_k4_spectral.png')

# ---------- fig 1: all 24 datasets grid (v6) ----------
n = len(DS)
ncols, nrows = 4, (n + 3) // 4
fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows))
axes = np.atleast_1d(axes).ravel()
for i, ds in enumerate(DS):
    ax = axes[i]
    X = groups[ds]
    lab, info = v6._run(X, ds)
    Z = v6._reduce(v6._clean(X), v6.NORM_PRIOR.get(ds, 'minmax'))
    ax.scatter(Z[:, 0], Z[:, 1], c=lab, cmap='tab10', s=6, edgecolors='none')
    ax.set_title('%s | %s k=%s' % (ds, str(info.get('winner'))[:14], info.get('k_used')), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for j in range(len(DS), len(axes)):
    axes[j].axis('off')
fig.suptitle('v6: all 24 datasets, top-2 PCA dims of Z-space, color = predicted cluster', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(os.path.join('clustering_pictures_v6', 'fig1_all_24_grid.png'), bbox_inches='tight')
plt.close(fig)
print('saved fig1 -> clustering_pictures_v6/fig1_all_24_grid.png')