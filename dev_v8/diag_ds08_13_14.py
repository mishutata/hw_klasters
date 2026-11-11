# -*- coding: utf-8 -*-
"""Diagnose ds08/ds13/ds14 (the 8-dim-cap datasets, lowest sim ARIs besides rings).

Part 1: pictures -> dev_v8/diag_<ds>.png
   panel a: raw PCA-2 (StandardScaler) with v8 full-train labels
   panel b: Z-space (current _reduce, cap 8) top-2 PCA with v8 labels
   panel c: Z-space ALL-DIMS (cap lifted to 14) top-2 PCA with ref labels
Part 2: test-size probe (same 80/20 protocol as sim_test_eval.py)
   for each ds: current pipeline vs all-dims variant, k in 2..5,
   KMeans/GMM/Spectral, ARI vs reference + seed stability.
"""
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

spec = importlib.util.spec_from_file_location('sol', 'my_solution8/solution.py')
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

DS = ['ds08', 'ds13', 'ds14']
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}

ref = {ds: sol.cluster(groups[ds], ds) for ds in DS}


def reduce_cap(X, norm, cap):
    Z = sol._apply_norm(X, norm)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    p = PCA(random_state=42).fit(Z)
    cum = np.cumsum(p.explained_variance_ratio_)
    n = max(2, min(cap, int(np.searchsorted(cum, 0.95) + 1), X.shape[1]))
    return PCA(n_components=n, random_state=42).fit_transform(Z), n


for ds in DS:
    X = groups[ds]
    lab = ref[ds]
    norm = sol.NORM_PRIOR.get(ds, 'minmax')
    Z8, n8 = reduce_cap(X, norm, 8)
    Z14, n14 = reduce_cap(X, norm, 14)
    raw2 = PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler().fit_transform(np.nan_to_num(X)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, P, t in [
        (axes[0], raw2, 'a) raw PCA-2 | v8 labels'),
        (axes[1], Z8[:, :2], 'b) Z cap=8 (%d dims) top-2 | v8 labels' % n8),
        (axes[2], Z14[:, :2], 'c) Z all-dims (%d dims) top-2 | ref labels' % n14),
    ]:
        ax.scatter(P[:, 0], P[:, 1], c=lab, cmap='tab10', s=5, edgecolors='none')
        ax.set_title(t, fontsize=9)
    fig.suptitle('%s | k=%d | sil(cap8)=%.3f sil(all)=%.3f' % (
        ds, len(np.unique(lab)),
        silhouette_score(Z8, lab), silhouette_score(Z14, lab)), y=1.02)
    fig.tight_layout()
    fig.savefig('dev_v8/diag_%s.png' % ds, bbox_inches='tight')
    plt.close(fig)
    print('saved dev_v8/diag_%s.png  (n8=%d n14=%d)' % (ds, n8, n14))

# ---------- part 2: test-size probe ----------
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

print('\n' + '=' * 78)
rng = np.random.RandomState(42)
for ds in DS:
    X = groups[ds]
    R = ref[ds]
    tidx = []
    for c in np.unique(R):
        idx = np.where(R == c)[0]
        tidx.append(rng.choice(idx, size=max(1, int(round(len(idx) * 0.2))), replace=False))
    ti = np.sort(np.concatenate(tidx))
    T, refT = X[ti], R[ti]
    norm = sol.NORM_PRIOR.get(ds, 'minmax')
    Z8t, n8t = reduce_cap(T, norm, 8)
    Z14t, n14t = reduce_cap(T, norm, 14)
    print('\n=== %s  n_test=%d ===' % (ds, len(T)))
    # baseline: current full pipeline at test size
    base = sol.cluster(T, ds)
    print('  v8 pipeline as-is          ARI=%.3f  k=%d' % (
        adjusted_rand_score(refT, base), len(np.unique(base))))
    for tag, Zt in (('cap8', Z8t), ('alldims', Z14t)):
        for k in (2, 3, 4, 5):
            row = []
            km = [KMeans(n_clusters=k, n_init=15, random_state=s).fit(Zt).labels_
                  for s in (42, 7, 123)]
            stab = float(np.mean([adjusted_rand_score(km[i], km[j])
                                  for i, j in ((0, 1), (0, 2), (1, 2))]))
            row.append('KM k=%d ARI=%.3f st=%.2f' % (
                k, adjusted_rand_score(refT, km[0]), stab))
            try:
                g = [GaussianMixture(n_components=k, covariance_type='full',
                                     n_init=3, random_state=s).fit(Zt)
                     for s in (42, 7)]
                row.append('GM k=%d ARI=%.3f st=%.2f' % (
                    k, adjusted_rand_score(refT, g[0].predict(Zt)),
                    adjusted_rand_score(g[0].predict(Zt), g[1].predict(Zt))))
            except Exception:
                pass
            try:
                sp = SpectralClustering(n_clusters=k, affinity='nearest_neighbors',
                                        n_neighbors=max(10, int(np.sqrt(len(Zt)))),
                                        random_state=42).fit_predict(Zt)
                row.append('SP k=%d ARI=%.3f' % (k, adjusted_rand_score(refT, sp)))
            except Exception:
                pass
            print('  %-8s %s' % (tag, ' | '.join(row)))
