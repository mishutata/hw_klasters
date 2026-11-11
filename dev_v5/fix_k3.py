# -*- coding: utf-8 -*-
"""Candidate partitions for ds04/ds09/ds19 at k=2..4 (KMeans/GMM/Spectral),
zoooomed PCA-2 figures + sil/stability table, to pick final k and winner.
Also shows the raw (unlabeled) geometry of each dataset."""
import importlib.util
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
spec = importlib.util.spec_from_file_location('sol', 'my_solution5/solution.py')
sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)

df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]

for ds in ['ds04', 'ds09', 'ds19']:
    g = df[df.dataset_id == ds]
    X = sol._clean(g[xcols].values.astype(float))
    Z = sol._reduce(X, sol.NORM_PRIOR.get(ds, 'minmax'))
    sc = sol._apply_norm(X, sol.NORM_PRIOR.get(ds, 'minmax'))
    P = PCA(2, random_state=42).fit_transform(sc)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for r, k in enumerate([2, 3, 4]):
        cands = {}
        km = KMeans(k, n_init=15, random_state=42).fit(Z)
        cands['KMeans'] = km.labels_
        try:
            gm = GaussianMixture(k, covariance_type='full', n_init=3,
                                 random_state=42).fit(Z)
            lab_gm = gm.predict(Z)
            if len(np.unique(lab_gm)) == k:
                cands['GMM'] = lab_gm
        except Exception as e:
            print(ds, k, 'GMM fail', repr(e))
        try:
            sp = SpectralClustering(k, affinity='nearest_neighbors',
                                    n_neighbors=min(len(Z) - 1,
                                                    max(10, int(np.sqrt(len(Z))))),
                                    random_state=42).fit_predict(Z)
            if len(np.unique(sp)) == k:
                cands['Spectral'] = sp
        except Exception as e:
            print(ds, k, 'Spec fail', repr(e))
        fits = [KMeans(k, n_init=15, random_state=s).fit(Z) for s in (42, 7, 123)]
        stab = float(np.mean([adjusted_rand_score(fits[i].labels_, fits[j].labels_)
                              for i, j in ((0, 1), (0, 2), (1, 2))]))
        for c, nm in enumerate(cands):
            ax = axes[r][c]
            lab = cands[nm]
            ax.scatter(P[:, 0], P[:, 1], c=lab, s=6, cmap='tab10', alpha=0.7)
            sil = silhouette_score(Z, lab)
            ax.set_title(f'{ds} k={k} {nm}\nsil={sil:.3f}'
                         + (f' stabKM={stab:.2f}' if nm == 'KMeans' else ''),
                         fontsize=9)
            ax.axis('off')
        for c in range(len(cands), 4):
            axes[r][c].axis('off')
    plt.tight_layout()
    plt.savefig(f'dev_v5/k3_{ds}.png', dpi=100)
    plt.close()
    print('saved dev_v5/k3_%s.png' % ds)

# raw geometry, no labels
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, ds in zip(axes, ['ds04', 'ds09', 'ds19']):
    g = df[df.dataset_id == ds]
    X = sol._clean(g[xcols].values.astype(float))
    sc = sol._apply_norm(X, sol.NORM_PRIOR.get(ds, 'minmax'))
    P = PCA(2, random_state=42).fit_transform(sc)
    ax.scatter(P[:, 0], P[:, 1], s=6, c='k', alpha=0.5)
    ax.set_title(f'{ds} raw (PCA-2)')
plt.tight_layout()
plt.savefig('dev_v5/raw_geom.png', dpi=100)
print('saved dev_v5/raw_geom.png')