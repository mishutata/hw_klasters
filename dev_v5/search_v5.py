# -*- coding: utf-8 -*-
"""v5 candidate search for the six fragile datasets.

Base file = analysis2/v3_solution.py (the exact 0.45 submission). For each of
ds04/ds09/ds10 (k-prior 3) and ds11/ds15/ds19 (k-prior 4) we test:
  - 'auto'      : v3 behavior unchanged (baseline)
  - forced k in {k0-1, k0, k0+1} x forced winner in {KMeans, GMM, Spectral}
each configuration is run on R random ~20% subsamples (scoring scale, ~190 pts):
  stab_small    : mean pairwise ARI between subsample runs
  consist_full  : mean ARI(subsample labels, full-set labels) on the overlap
  sil_full      : silhouette of the full-set partition
Acceptance rule: a forced config must beat 'auto' on stab_small and not worsen
sil_full by more than SIL_EPS. No transductive pooling, no embedded artifacts.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location('v3', 'analysis2/v3_solution.py')
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

DS = ['ds04', 'ds09', 'ds10', 'ds11', 'ds15', 'ds19']
R = 8          # subsamples per config
FRAC = 0.2     # scoring-scale subsample size
SIL_EPS = 0.02
SEEDS = list(range(100, 100 + R))

df = pd.read_csv('public_data/train.csv')


def run_config(Xc, ds, k_forced=None, winner_forced=None):
    """Run the v3 pipeline with optional forced k / winner. Returns labels."""
    Z = v3._reduce(Xc, v3.NORM_PRIOR.get(ds, 'minmax'))
    n = len(Z)
    k0 = v3.K_PRIOR.get(ds, 3)
    ks = sorted({max(2, k0 - 1), k0, min(k0 + 1, max(2, n // 5))})
    sils = {k: v3._sil(Z, v3.KMeans(n_clusters=k, n_init=15, random_state=42).fit(Z).labels_)
            for k in ks}
    best_k = max(ks, key=lambda k: -sils[k])
    if best_k != k0 and sils[best_k] - sils[k0] < 0.04:
        best_k = k0
    best_k = min(best_k, max(2, n // 3))
    if k_forced is not None:
        best_k = k_forced
    fits = [v3.KMeans(n_clusters=best_k, n_init=15, random_state=s).fit(Z)
            for s in (42, 7, 123)]
    stab = float(np.mean([adjusted_rand_score(fits[i].labels_, fits[j].labels_)
                          for i, j in ((0, 1), (0, 2), (1, 2))]))
    core_m, frac_rm = v3._core_mask(Z)
    use_core = stab < 0.8 and frac_rm > 0
    Zfit = Z[core_m] if use_core else Z
    cands = []
    km = v3.KMeans(n_clusters=best_k, n_init=15, random_state=42).fit(Zfit)
    cands.append(('KMeans', v3._extend(Zfit, km.labels_, Z)))
    try:
        gm = v3.GaussianMixture(n_components=best_k, covariance_type='full',
                                n_init=3, random_state=42).fit(Zfit)
        lab_gm = gm.predict(Z)
        if len(np.unique(lab_gm)) == best_k:
            cands.append(('GaussianMixture', lab_gm))
    except Exception:
        pass
    try:
        sp = v3.SpectralClustering(
            n_clusters=best_k, affinity='nearest_neighbors',
            n_neighbors=min(len(Zfit) - 1, max(10, int(np.sqrt(len(Zfit))))),
            random_state=42).fit_predict(Zfit)
        if len(np.unique(sp)) == best_k:
            cands.append(('SpectralClustering', v3._extend(Zfit, sp, Z)))
    except Exception:
        pass
    if winner_forced is not None:
        keep = [c for c in cands if c[0] == winner_forced]
        if keep:
            cands = keep
    i = int(np.argmax(np.array([v3._sil(Z, c) for _, c in cands])
                      + 0.3 * v3._consensus([c for _, c in cands])))
    return np.asarray(cands[i][1], dtype=int), best_k


rows = []
for ds in DS:
    Xs = df[df['dataset_id'] == ds].drop(columns=['id', 'dataset_id'])
    Xc = v3._clean(Xs.to_numpy(float))
    k0 = v3.K_PRIOR[ds]
    configs = [('auto', None, None)]
    for k in sorted({max(2, k0 - 1), k0, min(k0 + 1, 8)}):
        for w in ('KMeans', 'GaussianMixture', 'SpectralClustering'):
            configs.append((f'k{k}/{w}', k, w))
    for name, kf, wf in configs:
        # full set
        lab_full, k_used = run_config(Xc, ds, kf, wf)
        sil_full = v3._sil(v3._reduce(Xc, v3.NORM_PRIOR.get(ds, 'minmax')), lab_full)
        runs, idxs = [], []
        for s in SEEDS:
            rng = np.random.RandomState(s)
            m = len(Xc)
            take = rng.choice(m, size=int(FRAC * m), replace=False)
            lab_sub, _ = run_config(Xc[take], ds, kf, wf)
            runs.append(lab_sub)
            idxs.append(take)
        pair = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                # align on the common points of runs i and j
                inter = np.intersect1d(idxs[i], idxs[j])
                if len(inter) < 20:
                    continue
                pi = {v: t for t, v in enumerate(idxs[i])}
                pj = {v: t for t, v in enumerate(idxs[j])}
                li = runs[i][np.array([pi[v] for v in inter])]
                lj = runs[j][np.array([pj[v] for v in inter])]
                pair.append(adjusted_rand_score(li, lj))
        stab_small = float(np.mean(pair)) if pair else 0.0
        cf = []
        for lab_sub, take in zip(runs, idxs):
            cf.append(adjusted_rand_score(lab_full[take], lab_sub))
        consist = float(np.mean(cf))
        rows.append({'ds': ds, 'config': name, 'k_used': k_used,
                    'stab_small': round(stab_small, 4),
                    'consist_full': round(consist, 4),
                    'sil_full': round(sil_full, 4)})
        print(f"{ds} {name:28s} stab={stab_small:.4f} consist={consist:.4f} sil={sil_full:.4f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv('dev_v5/search_results.csv', index=False)
print('\nsaved dev_v5/search_results.csv')
