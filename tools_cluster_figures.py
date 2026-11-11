# -*- coding: utf-8 -*-
"""Clustering pictures for hw_klasters.

Runs the v4 solution (dev_v4/solution_v4.py) on public_data/train.csv and saves:
  clustering_pictures/fig1_all_24_grid.png        - all datasets, Z-space, v4 labels
  clustering_pictures/fig2_ring_blob_ds16_23.png  - ds16/ds23 ring+blob with fitted boundary
  clustering_pictures/fig3_k_selection_silhouette.png - silhouette vs k, prior & chosen marked
  clustering_pictures/fig4_v3_vs_v4_compare.png   - v3 vs v4 labels on the hardest datasets
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
OUT = 'clustering_pictures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'axes.titlesize': 10})

# ---------- load data + solutions ----------
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}
DS = sorted(groups)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

v4 = load('sol_v4', 'dev_v4/solution_v4.py')
v3 = load('sol_v3', 'my_solution3/solution.py')
print('loaded v4 + v3')

# ---------- per-dataset: Z coords (v4 transform) + labels/diagnostics ----------
data = {}
for ds in DS:
    X = groups[ds]
    lab, info = v4._run(X, ds)
    art = v4.ART[ds]
    Xc = v4._clean(X)
    Z0 = v4._apply_norm(Xc, art['norm'], art['nprm'])
    Z0 = np.where(np.isfinite(Z0), Z0, 0.0)
    Z = v4._pca_transform(Z0, art['d_pca'], np.asarray(art['pmean']), np.asarray(art['pcomp']))
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
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=lab, cmap='tab10', s=6, edgecolors='none')
    ax.set_title('%s | %s k=%s' % (ds, str(info.get('winner'))[:14], info.get('k_used')), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for j in range(len(DS), len(axes)):
    axes[j].axis('off')
fig.suptitle('v4: all 24 datasets, top-2 PCA dims of Z-space, color = predicted cluster', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(os.path.join(OUT, 'fig1_all_24_grid.png'), bbox_inches='tight')
plt.close(fig)
print('saved fig1')

# ---------- fig 2: ds16 / ds23 ring + blob with fitted ellipse boundary ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
for ax, ds in zip(axes, ('ds16', 'ds23')):
    Z, lab, info = data[ds]
    p = Z[:, :2]
    c, th, R = v4._ellipse_boundary(p)
    ang = np.linspace(-np.pi, np.pi, 720)
    t_ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
    R_ext = np.concatenate([R, R, R])
    ax.scatter(p[:, 0], p[:, 1], c=lab, cmap='tab10', s=8, edgecolors='none',
               label='labels (inside=%d)' % int((lab == 1).sum()))
    ax.plot(c[0] + np.interp(ang, t_ext, R_ext) * np.cos(ang),
            c[1] + np.interp(ang, t_ext, R_ext) * np.sin(ang), 'r--', lw=2,
            label='fitted ellipse boundary')
    ax.plot(*c, 'k+', ms=10)
    ax.set_title('%s: ring (outside) vs central blob (inside)' % ds)
    ax.legend(fontsize=8, loc='best')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_ring_blob_ds16_23.png'), bbox_inches='tight')
plt.close(fig)
print('saved fig2')

# ---------- fig 3: silhouette vs k (prior & chosen marked) ----------
ncols, nrows = 4, (n + 3) // 4
fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows))
axes = np.atleast_1d(axes).ravel()
for i, ds in enumerate(DS):
    ax = axes[i]
    Z, lab, info = data[ds]
    sil = {int(k): float(v) for k, v in (info.get('sil_by_k') or {}).items()}
    ks = sorted(sil)
    if len(sil) > 1:
        ks = sorted(sil)
        ax.plot(ks, [sil[k] for k in ks], 'o-', ms=3, lw=1.2, color='tab:blue')
    kp = info.get('k_prior'); ku = info.get('k_used')
    if kp is not None:
        ax.axvline(kp, color='orange', ls=':', lw=1)
    if ku is not None and ks:
        if int(ku) in sil:
            ax.plot([ku], [sil[int(ku)]], 'r*', ms=12)
    ax.set_title('%s (prior k=%s, used k=%s)' % (ds, kp, ku), fontsize=9)
    ax.tick_params(labelsize=7)
for j in range(len(DS), len(axes)):
    axes[j].axis('off')
fig.suptitle('Silhouette score vs k on pooled Z-space; red star = chosen k, dotted line = a-priori k', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(os.path.join(OUT, 'fig3_k_selection_silhouette.png'), bbox_inches='tight')
plt.close(fig)
print('saved fig3')

# ---------- fig 4: v3 vs v4 labels on the hardest datasets ----------
hard = ['ds08', 'ds09', 'ds13', 'ds14']
fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
for ax, ds in zip(axes.ravel(), hard):
    Z, lab4, info4 = data[ds]
    lab3 = v3.cluster(groups[ds], ds)
    ax.scatter(Z[:, 0], Z[:, 1], c=lab3, cmap='tab10', s=16, alpha=0.9,
               edgecolors='k', linewidths=0.3)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(lab3, lab4)
    ax.set_title('%s: v3 labels (KMeans-era pipeline) | ARI(v3,v4)=%.3f' % (ds, ari), fontsize=10)
    ax.set_xlabel('Z1'); ax.set_ylabel('Z2')
fig.suptitle('v3 (cluster only the ~20% test points) vs v4 (anchored on public train)', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(OUT, 'fig4_v3_vs_v4_compare.png'), bbox_inches='tight')
plt.close(fig)
print('saved fig4')
print('DONE ->', OUT)
