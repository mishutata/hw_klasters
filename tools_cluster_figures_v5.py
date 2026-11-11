# -*- coding: utf-8 -*-
"""Clustering pictures for the v5 solution (my_solution5/solution.py).

Runs v5 on public_data/train.csv and saves:
  clustering_pictures_v5/fig1_all_24_grid.png - all datasets, Z-space, v5 labels
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
OUT = 'clustering_pictures_v5'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'axes.titlesize': 10})

# ---------- load data + v5 solution ----------
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}
DS = sorted(groups)

spec = importlib.util.spec_from_file_location('sol_v5', 'my_solution5/solution.py')
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)
print('loaded v5')

# ---------- per-dataset: Z coords (v5 transform) + labels/diagnostics ----------
data = {}
for ds in DS:
    X = groups[ds]
    lab, info = v5._run(X, ds)
    Xc = v5._clean(X)
    Z = v5._reduce(Xc, v5.NORM_PRIOR.get(ds, 'minmax'))
    data[ds] = (Z, lab, info)
    print(ds, 'k=%s %s' % (info.get('k_used'), info.get('winner')))

# ---------- fig 1: all 24 datasets grid ----------
n = len(DS)
ncols, nrows = 4, (n + 3) // 4
fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows))
axes = np.atleast_1d(axes).ravel()
for i, ds in enumerate(DS):
    ax = axes[i]
    Z, lab, info = data[ds]
    ax.scatter(Z[:, 0], Z[:, 1], c=lab, cmap='tab10', s=6, edgecolors='none')
    ax.set_title('%s | %s k=%s' % (ds, str(info.get('winner'))[:14], info.get('k_used')), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for j in range(len(DS), len(axes)):
    axes[j].axis('off')
fig.suptitle('v5: all 24 datasets, top-2 PCA dims of Z-space, color = predicted cluster', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(os.path.join(OUT, 'fig1_all_24_grid.png'), bbox_inches='tight')
plt.close(fig)
print('saved fig1 ->', os.path.join(OUT, 'fig1_all_24_grid.png'))
