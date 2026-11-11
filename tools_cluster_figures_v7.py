# -*- coding: utf-8 -*-
"""Clustering pictures for the v7 solution (my_solution7/solution.py).

v7 = v6 with ds16/ds23 back to the geometric ring/blob split (k=2) instead of
Spectral k=4, using per-dataset tuned boundary radii
(RING_FRAC: ds16 0.93, ds23 0.90, tuned in dev_v6/ring_radius_tune3/4).
Saves:
  dev_v7/ds16_ds23_ring_blob.png    - ds16/ds23: v5 (frac .80) vs v6 (Spectral k=4)
                                     vs v7 (tuned frac), + raw-PCA panel with the
                                     fitted boundary overlaid
  clustering_pictures_v7/fig1_all_24_grid.png - all datasets, Z-space, v7 labels
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
for d in ('dev_v7', 'clustering_pictures_v7'):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'axes.titlesize': 10})

# ---------- load data + v5/v6/v7 solutions ----------
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}
DS = sorted(groups)

mods = {}
for tag in ('v5', 'v6', 'v7'):
    spec = importlib.util.spec_from_file_location('sol_%s' % tag,
                                                 'my_solution%s/solution.py' % tag[1])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    mods[tag] = m
print('loaded v5 + v6 + v7')

raw_pca2 = {ds: PCA(n_components=2, random_state=42).fit_transform(
    np.nan_to_num(groups[ds])) for ds in DS}


def sc_pca2(X):
    """Same 2-D embedding the ring/blob branch fits in (StandardScaler + PCA)."""
    return PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler().fit_transform(np.nan_to_num(X)))


def boundary_curve(m7, p2, ds):
    """Closed curve of the fitted ring boundary at the tuned frac."""
    c, th, R = m7._ellipse_boundary(p2)
    a = np.linspace(-np.pi, np.pi, 400)
    t_ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
    R_ext = np.concatenate([R, R, R])
    Rloc = np.interp(a, t_ext, R_ext)
    frac = m7.RING_FRAC.get(ds, 0.80)
    return c[0] + frac * Rloc * np.cos(a), c[1] + frac * Rloc * np.sin(a), frac


# ---------- ds16/ds23: v5 (frac .80) vs v6 (Spectral k=4) vs v7 (tuned frac) ----------
def compare_fig(ds_list, fname, ncols=4):
    m5, m6, m7 = mods['v5'], mods['v6'], mods['v7']
    nrows = len(ds_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.9 * nrows))
    if nrows == 1:
        axes = np.atleast_1d(axes)[None, :]
    for r, ds in enumerate(ds_list):
        X = groups[ds]
        lab5, info5 = m5._run(X, ds)
        lab6, info6 = m6._run(X, ds)
        lab7, info7 = m7._run(X, ds)
        Z5 = m5._reduce(m5._clean(X), m5.NORM_PRIOR.get(ds, 'minmax'))
        Z6 = m6._reduce(m6._clean(X), m6.NORM_PRIOR.get(ds, 'minmax'))
        Z7 = m7._reduce(m7._clean(X), m7.NORM_PRIOR.get(ds, 'minmax'))
        s5 = silhouette_score(Z5, lab5) if len(np.unique(lab5)) > 1 else float('nan')
        s6 = silhouette_score(Z6, lab6) if len(np.unique(lab6)) > 1 else float('nan')
        s7 = silhouette_score(Z7, lab7) if len(np.unique(lab7)) > 1 else float('nan')
        panels = [
            (Z5, lab5, '%s v5: %s k=%s | sil %.3f' % (ds, info5['winner'], info5['k_used'], s5)),
            (Z6, lab6, '%s v6: %s k=%s | sil %.3f' % (ds, info6['winner'], info6['k_used'], s6)),
            (Z7, lab7, '%s v7: %s k=%s | sil %.3f' % (ds, info7['winner'], info7['k_used'], s7)),
        ]
        for c_, (P, lab, title) in enumerate(panels):
            ax = axes[r, c_]
            ax.scatter(P[:, 0], P[:, 1], c=lab, cmap='tab10', s=5, edgecolors='none')
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        # raw-PCA panel with v7 labels; for ring datasets overlay the fitted boundary
        P = sc_pca2(X)
        ax = axes[r, 3]
        inner = float(lab7.mean())
        ax.scatter(P[:, 0], P[:, 1], c=lab7, cmap='tab10', s=5, edgecolors='none')
        if ds in m7.RING_DS:
            bx, by, frac = boundary_curve(m7, P, ds)
            ax.plot(bx, by, 'k--', lw=1.6, label='fitted boundary (frac=%.2f)' % frac)
            ax.legend(fontsize=8, loc='best')
        ax.set_title('%s raw PCA-2, v7 labels | inner mass %.0f%%' % (ds, 100 * inner), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('v5 vs v6 vs v7 on %s (Z-space per version; raw panel = StandardScaler+PCA-2)' % '/'.join(ds_list), y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)

compare_fig(['ds16', 'ds23'], 'dev_v7/ds16_ds23_ring_blob.png')

# ---------- fig 1: all 24 datasets grid (v7) ----------
m7 = mods['v7']
n = len(DS)
ncols, nrows = 4, (n + 3) // 4
fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows))
axes = np.atleast_1d(axes).ravel()
for i, ds in enumerate(DS):
    ax = axes[i]
    X = groups[ds]
    lab, info = m7._run(X, ds)
    Z = m7._reduce(m7._clean(X), m7.NORM_PRIOR.get(ds, 'minmax'))
    ax.scatter(Z[:, 0], Z[:, 1], c=lab, cmap='tab10', s=6, edgecolors='none')
    ax.set_title('%s | %s k=%s' % (ds, str(info.get('winner'))[:14], info.get('k_used')), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for j in range(len(DS), len(axes)):
    axes[j].axis('off')
fig.suptitle('v7: all 24 datasets, top-2 PCA dims of Z-space, color = predicted cluster', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(os.path.join('clustering_pictures_v7', 'fig1_all_24_grid.png'), bbox_inches='tight')
plt.close(fig)
print('saved fig1 -> clustering_pictures_v7/fig1_all_24_grid.png')
