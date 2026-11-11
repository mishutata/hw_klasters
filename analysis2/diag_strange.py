# -*- coding: utf-8 -*-
"""Deep geometry check on the datasets that look strange in the pictures:
ds04, ds09, ds10 (right k, but 2 clusters look merged) and
ds11, ds15, ds19 (partition looks odd).

For each: silhouette for k=2..8 (KMeans + GMM BIC), HDBSCAN natural count,
and a zoom figure with several candidate partitions side by side.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location('sol3', 'my_solution3/solution.py')
sol3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol3)

DS = ['ds04', 'ds09', 'ds10', 'ds11', 'ds15', 'ds19']
df = pd.read_csv('public_data/train.csv')

rows = []
for ds in DS:
    Xs = df[df['dataset_id'] == ds].drop(columns=['id', 'dataset_id'])
    Xc = sol3._clean(Xs.to_numpy(float))
    Z = sol3._reduce(Xc, sol3.NORM_PRIOR.get(ds, 'minmax'))
    k0 = sol3.K_PRIOR[ds]
    print(f'\n=== {ds}  n={len(Z)}  dims={Z.shape[1]}  prior k={k0}')
    line = {'ds': ds, 'k_prior': k0}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=15, random_state=42).fit(Z)
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             n_init=3, random_state=42).fit(Z)
        line[f'sil_k{k}'] = round(silhouette_score(Z, km.labels_), 3)
        line[f'bic_k{k}'] = int(gm.bic(Z))
    hd_counts = {}
    for mcs in (5, 10, 20, 40):
        try:
            h = HDBSCAN(min_cluster_size=mcs).fit_predict(Z)
            ncl = len(np.unique(h)) - (1 if (h < 0).any() else 0)
            hd_counts[mcs] = (ncl, round(float((h < 0).mean()), 2))
        except Exception as e:
            hd_counts[mcs] = str(e)[:40]
    print('  sil:', {k: line[f'sil_k{k}'] for k in range(2, 9)})
    print('  bic:', {k: line[f'bic_k{k}'] for k in range(2, 9)})
    print('  hdbscan (mcs -> n_clusters, noise):', hd_counts)
    rows.append(line)
pd.DataFrame(rows).to_csv('analysis2/diag_strange.csv', index=False)
