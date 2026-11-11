# -*- coding: utf-8 -*-
"""ds18 deep probe: raw-space geometry, norms, density methods. Saves dev_v5/probe_ds18.png"""
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

_here = os.path.dirname(os.path.abspath(__file__))
_spec_path = os.path.join(_here, '..', 'my_solution5', 'solution.py')
spec = importlib.util.spec_from_file_location('sol_v5', _spec_path)
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)

_root = os.path.dirname(_here)
df = pd.read_csv(os.path.join(_root, 'public_data', 'train.csv'))
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}
X = groups['ds18']
print('ds18 shape:', X.shape, 'cols:', xcols)

# norms comparison at k=2/3 (Z-space sil + ARI between norms)
norms = ['minmax', 'standard', 'robust', 'log-sym']
print('\n-- silhouette by norm --')
for nm in norms:
    Z = v5._reduce(v5._clean(X), nm)
    s2 = silhouette_score(Z, KMeans(n_clusters=2, n_init=15, random_state=42).fit_predict(Z))
    s3 = silhouette_score(Z, KMeans(n_clusters=3, n_init=15, random_state=42).fit_predict(Z))
    print('  %-8s k2=%.3f k3=%.3f' % (nm, s2, s3))

# raw-space structure: PCA-2 of RAW X (no norm), KMeans in raw space
Xr = v5._clean(X)
print('\n-- raw space --')
pca_raw = PCA(n_components=2, random_state=42).fit_transform(Xr)
for k in (2, 3):
    km_r = KMeans(n_clusters=k, n_init=15, random_state=42).fit(Xr)
    print('  raw KMeans k=%d sil(raw)=%.3f' % (k, silhouette_score(Xr, km_r.labels_)))

# density on Z (log-sym, current norm)
Z = v5._reduce(Xr, 'log-sym')
print('\n-- HDBSCAN on Z --')
for mcs in (10, 20, 30, 50):
    try:
        h = HDBSCAN(min_cluster_size=mcs).fit_predict(Z)
        ncl = len(np.unique(h)) - (1 if (h < 0).any() else 0)
        print('  mcs=%d -> clusters=%d noise=%.3f' % (mcs, ncl, float((h < 0).mean())))
    except Exception as e:
        print('  mcs=%d err %s' % (mcs, e))

# figure: raw PCA-2 with k=2/k=3 raw labels; Z with HDBSCAN mcs=30 + KMeans k=2/3
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
km2r = KMeans(n_clusters=2, n_init=15, random_state=42).fit(Xr)
km3r = KMeans(n_clusters=3, n_init=15, random_state=42).fit(Xr)
hd = HDBSCAN(min_cluster_size=30).fit_predict(Z)
axes[0].scatter(pca_raw[:, 0], pca_raw[:, 1], c=km2r.labels_, cmap='tab10', s=20, edgecolors='none')
axes[0].set_title('RAW PCA-2, KMeans k=2 (raw)')
axes[1].scatter(pca_raw[:, 0], pca_raw[:, 1], c=km3r.labels_, cmap='tab10', s=20, edgecolors='none')
axes[1].set_title('RAW PCA-2, KMeans k=3 (raw)')
Zcur, _ = v5._run(X, 'ds18')
axes[2].scatter(Z[:, 0], Z[:, 1], c=Zcur, cmap='tab10', s=20, edgecolors='none')
axes[2].set_title('Z (log-sym), current k=2')
axes[3].scatter(Z[:, 0], Z[:, 1], c=hd, cmap='tab10', s=20, edgecolors='none')
axes[3].set_title('Z, HDBSCAN mcs=30 (noise=-1)')
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
_out = os.path.join(_here, 'probe_ds18.png')
fig.savefig(_out, bbox_inches='tight')
print('\nsaved', _out)
