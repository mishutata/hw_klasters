"""Verify k=4 spectral for ds16/ds23; deep ds18 diagnostics."""
import importlib.util
import os

import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
spec = importlib.util.spec_from_file_location('sol_v5', os.path.join(_root, 'my_solution5', 'solution.py'))
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)

df = pd.read_csv(os.path.join(_root, 'public_data', 'train.csv'))
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score


def sp_lab(Z, k, rs=42):
    return SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                              n_neighbors=min(len(Z) - 1, max(10, int(np.sqrt(len(Z))))),
                              random_state=rs).fit_predict(Z)


for ds in ('ds16', 'ds23'):
    X = groups[ds]
    Z = v5._reduce(X, v5.NORM_PRIOR.get(ds, 'minmax'))
    cur = v5.cluster(X, ds)
    print(f'== {ds} ==  n={len(X)} Zdim={Z.shape[1]}')
    lab = sp_lab(Z, 4)
    print('  spectral k=4 sizes:', np.bincount(lab).tolist())
    print('  sil(k4 spectral) = %.3f' % silhouette_score(Z, lab))
    for rs in (7, 123):
        lab2 = sp_lab(Z, 4, rs)
        print('  stability seed=%d: ARI vs seed42 = %.3f' % (rs, adjusted_rand_score(lab, lab2)))
    gm = GaussianMixture(n_components=4, covariance_type='full', n_init=5, random_state=42).fit(Z)
    print('  GMM k=4: BIC=%.0f  ARI vs spectral = %.3f' % (gm.bic(Z), adjusted_rand_score(lab, gm.predict(Z))))
    print('  ARI(spectral k=4, current ring/blob) = %.3f' % adjusted_rand_score(lab, cur))
    # arc check: angle histogram of outer shell in PCA-2
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    p2 = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
    c0 = p2.mean(0)
    d0 = np.linalg.norm(p2 - c0, axis=1)
    outer = d0 >= np.quantile(d0, 0.55)
    ang = (np.arctan2(p2[:, 1] - c0[1], p2[:, 0] - c0[0]) + np.pi) % (2 * np.pi)
    hist, _ = np.histogram(ang[outer], bins=36, range=(0, 2 * np.pi))
    print('  outer-shell angle hist (36 bins):', (hist / hist.sum()).round(3).tolist())

print()
X18 = groups['ds18']
Z = v5._reduce(X18, 'log-sym')
p = __import__('sklearn.decomposition', fromlist=['PCA']).PCA(random_state=42).fit(v5._apply_norm(X18, 'log-sym'))
evr = p.explained_variance_ratio_
print('ds18 PCA evr:', evr.round(3).tolist(), 'cum95 at dim', int(np.searchsorted(np.cumsum(evr), 0.95) + 1))
Zfull = v5._apply_norm(X18, 'log-sym')
km8 = KMeans(2, n_init=15, random_state=42).fit(Z).labels_
kmf = KMeans(2, n_init=15, random_state=42).fit(Zfull).labels_
print('ds18 KMeans k=2: Z(8dim) vs full 14-dim log-sym ARI = %.3f' % adjusted_rand_score(km8, kmf))
for rs in (7, 123):
    print('  seed=%d ARI vs 42: %.3f' % (rs, adjusted_rand_score(km8, KMeans(2, n_init=15, random_state=rs).fit(Z).labels_)))
print('ds18 GMM BIC on Z:', [round(GaussianMixture(k, covariance_type='full', n_init=3, random_state=42).fit(Z).bic(Z)) for k in (2, 3, 4)])
# which dims separate the two clouds?
sc = v5._apply_norm(X18, 'log-sym')
mu = np.array([sc[km8 == c].mean(0) for c in (0, 1)])
sd = np.sqrt(np.array([sc[km8 == c].var(0) for c in (0, 1)]))
d = np.abs(mu[0] - mu[1]) / (sd.mean(0) + 1e-9)
order = np.argsort(d)[::-1][:6]
print('ds18 most separating dims (0-based):', [(i, round(float(d[i]), 2)) for i in order])
