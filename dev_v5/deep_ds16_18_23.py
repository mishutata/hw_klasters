# -*- coding: utf-8 -*-
"""Deep check for ds16 / ds18 / ds23 under the v5 transform.

For ds16/ds23: current ring/blob (k=2) vs k=4 Spectral (user's proposal).
For ds18:     probe k in {2,3,4} x {KMeans, GMM, Spectral} to find a better split.
Saves dev_v5/deep_ds16_18_23.png
"""
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

spec = importlib.util.spec_from_file_location('sol_v5', 'my_solution5/solution.py')
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)

df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}

rows = []  # (ds, k, method, labels)
for ds in ('ds16', 'ds18', 'ds23'):
    X = groups[ds]
    Z = v5._reduce(v5._clean(X), v5.NORM_PRIOR.get(ds, 'minmax'))
    cur, info = v5._run(X, ds)
    print('=' * 16, ds, ' n=%d' % len(X), ' current:', info['winner'], 'k=', info['k_used'])
    rows.append((ds, int(info['k_used']), 'CURRENT(%s)' % info['winner'], cur))
    for k in (2, 3, 4):
        km = KMeans(n_clusters=k, n_init=15, random_state=42).fit(Z)
        sp = SpectralClustering(n_clusters=k, affinity='nearest_neighbors',
                                n_neighbors=min(len(Z) - 1, max(10, int(np.sqrt(len(Z))))),
                                random_state=42).fit_predict(Z)
        gm = GaussianMixture(n_components=k, covariance_type='full', n_init=3,
                             random_state=42).fit(Z)
        print('  k=%d  KMeans sil=%.3f | GMM sil=%.3f | Spectral sil=%.3f' % (
            k, silhouette_score(Z, km.labels_), silhouette_score(Z, gm.predict(Z)),
            silhouette_score(Z, sp)))
        rows.append((ds, k, 'KMeans', km.labels_))
        rows.append((ds, k, 'GMM', gm.predict(Z)))
        rows.append((ds, k, 'Spectral', sp))
    # ARI of k=4 spectral vs current
    sp4 = SpectralClustering(n_clusters=4, affinity='nearest_neighbors',
                             n_neighbors=min(len(Z) - 1, max(10, int(np.sqrt(len(Z))))),
                             random_state=42).fit_predict(Z)
    print('  ARI(k=4 spectral vs current) = %.3f' % adjusted_rand_score(sp4, cur))

# ---------- figure: one row per dataset, all candidates ----------
cur_lab = {}
for ds in ('ds16', 'ds18', 'ds23'):
    lab, info = v5._run(groups[ds], ds)
    cur_lab[ds] = (lab, info['winner'])
look = {t[:3]: t for t in rows}
fig, axes = plt.subplots(3, 8, figsize=(26, 10.5))
for r, ds in enumerate(('ds16', 'ds18', 'ds23')):
    X = groups[ds]
    Z = v5._reduce(v5._clean(X), v5.NORM_PRIOR.get(ds, 'minmax'))
    keys = ['cur', (2, 'KMeans'), (2, 'Spectral'), (2, 'GMM'),
            (3, 'KMeans'), (3, 'Spectral'), (3, 'GMM'), (4, 'Spectral')]
    for c, key in enumerate(keys):
        ax = axes[r, c]
        if key == 'cur':
            lab, w = cur_lab[ds]
            t = 'current (%s)' % w
        else:
            k, m = key
            lab = look[(ds, k, m)][3]
            t = '%s k=%d' % (m, k)
        ax.scatter(Z[:, 0], Z[:, 1], c=lab, cmap='tab10', s=5, edgecolors='none')
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('ds16 / ds18 / ds23: current labels vs k=2..4 candidates (v5 Z-space)')
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig('dev_v5/deep_ds16_18_23.png', bbox_inches='tight')
print('saved dev_v5/deep_ds16_18_23.png')
