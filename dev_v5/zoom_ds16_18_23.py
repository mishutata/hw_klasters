# -*- coding: utf-8 -*-
"""Zoom views: ds16/ds23 boundary problem + ds18 geometry.
Saves dev_v5/zoom_ds16_18_23.png
"""
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

spec = importlib.util.spec_from_file_location('sol_v5', 'my_solution5/solution.py')
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)

df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}

fig, axes = plt.subplots(3, 4, figsize=(22, 16))
for r, ds in enumerate(('ds16', 'ds18', 'ds23')):
    X = groups[ds]
    Xc = v5._clean(X)
    Z = v5._reduce(Xc, v5.NORM_PRIOR.get(ds, 'minmax'))
    cur, info = v5._run(X, ds)

    # col 0: current labels, big points
    ax = axes[r, 0]
    ax.scatter(Z[:, 0], Z[:, 1], c=cur, cmap='tab10', s=28, edgecolors='none')
    ax.set_title('%s current (%s) k=%s' % (ds, info['winner'], info['k_used']), fontsize=11)

    if ds in ('ds16', 'ds23'):
        # col 1: current + fitted ellipse boundary -> shows the cut
        ax = axes[r, 1]
        p = Z[:, :2]
        c, th, R = v5._ellipse_boundary(p)
        ang = np.linspace(-np.pi, np.pi, 720)
        t_ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
        R_ext = np.concatenate([R, R, R])
        ax.scatter(p[:, 0], p[:, 1], c=cur, cmap='tab10', s=28, edgecolors='none')
        ax.plot(c[0] + np.interp(ang, t_ext, R_ext) * np.cos(ang),
                c[1] + np.interp(ang, t_ext, R_ext) * np.sin(ang), 'r--', lw=2)
        # mislabeled w.r.t. boundary: inside-labeled but outside ellipse (and vice versa)
        from matplotlib.patches import Ellipse
        ax.set_title('%s: fitted boundary (red) vs labels' % ds, fontsize=11)
        sp4 = SpectralClustering(n_clusters=4, affinity='nearest_neighbors',
                                 n_neighbors=min(len(Z) - 1, max(10, int(np.sqrt(len(Z))))),
                                 random_state=42).fit_predict(Z)
        ax = axes[r, 2]
        ax.scatter(Z[:, 0], Z[:, 1], c=sp4, cmap='tab10', s=28, edgecolors='none')
        ax.set_title('%s: Spectral k=4' % ds, fontsize=11)
        # col 3: spectral k=4 with cluster sizes
        sizes = np.bincount(sp4)
        ax = axes[r, 3]
        ax.scatter(Z[:, 0], Z[:, 1], c=sp4, cmap='tab10', s=28, edgecolors='none')
        ax.set_title('%s: Spectral k=4 sizes=%s' % (ds, list(sizes)), fontsize=11)
    else:  # ds18
        km2 = KMeans(n_clusters=2, n_init=15, random_state=42).fit(Z)
        sp2 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                 n_neighbors=min(len(Z) - 1, max(10, int(np.sqrt(len(Z))))),
                                 random_state=42).fit_predict(Z)
        km3 = KMeans(n_clusters=3, n_init=15, random_state=42).fit(Z)
        sp3 = SpectralClustering(n_clusters=3, affinity='nearest_neighbors',
                                 n_neighbors=min(len(Z) - 1, max(10, int(np.sqrt(len(Z))))),
                                 random_state=42).fit_predict(Z)
        ax = axes[r, 1]
        ax.scatter(Z[:, 0], Z[:, 1], c=sp2, cmap='tab10', s=28, edgecolors='none')
        ax.set_title('ds18 Spectral k=2 (ARI vs KM: %.3f)' % adjusted_rand_score(km2.labels_, sp2), fontsize=11)
        ax = axes[r, 2]
        ax.scatter(Z[:, 0], Z[:, 1], c=km3.labels_, cmap='tab10', s=28, edgecolors='none')
        ax.set_title('ds18 KMeans k=3 sil=%.3f' % silhouette_score(Z, km3.labels_), fontsize=11)
        ax = axes[r, 3]
        ax.scatter(Z[:, 0], Z[:, 1], c=sp3, cmap='tab10', s=28, edgecolors='none')
        ax.set_title('ds18 Spectral k=3 sil=%.3f' % silhouette_score(Z, sp3), fontsize=11)
    for c in range(4):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
fig.suptitle('zoom: ds16/ds23 boundary + ds18 candidates (v5 Z-space)', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig('dev_v5/zoom_ds16_18_23.png', bbox_inches='tight')
print('saved dev_v5/zoom_ds16_18_23.png')
