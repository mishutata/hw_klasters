"""v2 prototype: v1 pipeline + density-based (DBSCAN-style) improvements.
Run on all public datasets; compare winner labels vs v1, stability, runtime."""
import sys, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'my_solver')
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import solution as v1

df = pd.read_csv('public_data/train.csv')

def core_mask(Z):
    n = len(Z)
    k = 5 if n >= 200 else 3
    d = cKDTree(Z).query(Z, k=k + 1)[0][:, k]
    thr = max(np.quantile(d, 0.95), np.median(d) * 2)
    m = d <= thr
    frac = 1 - m.mean()
    if frac > 0.08 or frac < 0.003:
        return np.ones(n, bool), 0.0
    return m, frac

def extend(lab_fit, Z_fit, Z_all):
    out = lab_fit.copy() if len(Z_all) == len(Z_fit) else None
    if out is not None and len(Z_all) == len(Z_fit):
        return out
    cents = np.array([Z_fit[lab_fit == c].mean(0) for c in np.unique(lab_fit)])
    uniq = np.unique(lab_fit)
    D = ((Z_all[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    return uniq[np.argmin(D, axis=1)]

v2_results = {}
for ds in sorted(df['dataset_id'].str.replace('_aug', '').unique()):
    d = df[df['dataset_id'].str.replace('_aug', '') == ds]
    cols = [c for c in d.columns if c.startswith('x')]
    X = d[cols].to_numpy(float)
    t0 = time.time()
    # ---- v1
    lab1, info1 = v1._run(X, ds)
    # ---- v2
    Xc = v1._clean(X); n = len(Xc); Z = v1._reduce(Xc)
    k0 = v1.K_PRIOR.get(ds, 3)
    ks = sorted({max(2, k0 - 1), k0, min(k0 + 1, max(2, n // 5))})
    sils = {}
    for k in ks:
        labk = KMeans(n_clusters=k, n_init=15, random_state=42).fit(Z).labels_
        sils[k] = v1._sil(Z, labk)
    best_k = max(ks, key=lambda k: -sils[k])
    if best_k != k0 and sils[best_k] - sils[k0] < 0.04:
        best_k = k0
    best_k = min(best_k, max(2, n // 3))
    # stability probe
    ks_fit = [KMeans(n_clusters=best_k, n_init=15, random_state=s).fit(Z) for s in (42, 7, 123)]
    ari_stab = np.mean([adjusted_rand_score(a.labels_, b.labels_) for a, b in
                        [(ks_fit[0], ks_fit[1]), (ks_fit[0], ks_fit[2]), (ks_fit[1], ks_fit[2])]])
    core_m, frac_rm = core_mask(Z)
    Zfit = Z[core_m] if (ari_stab < 0.8 and frac_rm > 0) else Z
    cands = []
    km = KMeans(n_clusters=best_k, n_init=15, random_state=42).fit(Zfit)
    cands.append(('KMeans', extend(km.labels_, Zfit, Z)))
    try:
        gm = GaussianMixture(n_components=best_k, covariance_type='full', n_init=3, random_state=42).fit(Zfit)
        labg = gm.predict(Z) if len(Zfit) == n else extend(np.zeros(len(Zfit), int), Zfit, Z)
        labg = gm.predict(Z)
        cands.append(('GaussianMixture', labg))
    except Exception:
        pass
    try:
        sp = SpectralClustering(n_clusters=best_k, affinity='nearest_neighbors',
                                n_neighbors=min(len(Zfit) - 1, max(10, int(np.sqrt(len(Zfit))))),
                                random_state=42).fit_predict(Zfit)
        if len(np.unique(sp)) == best_k:
            cands.append(('SpectralClustering', extend(sp, Zfit, Z)))
    except Exception:
        pass
    # DBSCAN robust grid
    d5 = cKDTree(Z).query(Z, k=6)[0][:, 5]
    db_lab = None
    for eps in np.unique(np.quantile(d5, [0.6, 0.75, 0.9, 0.95])):
        for ms in (max(3, n // 100), 5):
            lab = DBSCAN(eps=float(eps), min_samples=ms).fit_predict(Z)
            ncl = len(np.unique(lab)) - (1 if (lab < 0).any() else 0)
            noise = float((lab < 0).mean())
            if ncl >= 2 and ncl <= max(3, k0 + 3) and noise <= 0.15:
                lab_r = lab.copy()
                uniq = np.unique(lab[lab >= 0])
                cents = np.array([Z[lab == c].mean(0) for c in uniq])
                ni = np.where(lab < 0)[0]
                D2 = ((Z[ni][:, None, :] - cents[None, :, :]) ** 2).sum(-1)
                lab_r[ni] = uniq[np.argmin(D2, axis=1)]
                if db_lab is None or v1._sil(Z, lab_r) > v1._sil(Z, db_lab):
                    db_lab = lab_r
    # gate vs best parametric
    if db_lab is not None:
        g = max(normalized_mutual_info_score(db_lab, c[1]) for name, c in cands)
        if g >= 0.5:
            cands.append(('DBSCAN', db_lab))
    try:
        hd = HDBSCAN(min_cluster_size=max(10, n // (4 * best_k))).fit_predict(Z)
        ncl = len(np.unique(hd)) - (1 if (hd < 0).any() else 0)
        noise = float((hd < 0).mean())
        if ncl >= 2 and noise <= 0.35:
            hd_r = hd.copy()
            uniq = np.unique(hd[hd >= 0])
            cents = np.array([Z[hd == c].mean(0) for c in uniq])
            ni = np.where(hd < 0)[0]
            D2 = ((Z[ni][:, None, :] - cents[None, :, :]) ** 2).sum(-1)
            hd_r[ni] = uniq[np.argmin(D2, axis=1)]
            g = max(normalized_mutual_info_score(hd_r, c[1]) for name, c in cands)
            if g >= 0.5:
                cands.append(('HDBSCAN', hd_r))
    except Exception:
        pass
    score = np.array([v1._sil(Z, c) for _, c in cands]) + 0.3 * v1._consensus([c for _, c in cands])
    i = int(np.argmax(score))
    lab2 = np.asarray(cands[i][1], dtype=int)
    dt = time.time() - t0
    ari12 = adjusted_rand_score(lab1, lab2)
    v2_results[ds] = (lab2, cands[i][0], best_k, round(ari_stab, 3), round(frac_rm, 3),
                      'core' if len(Zfit) < n else 'all', dt, ari12,
                      {nm: round(float(s), 3) for (nm, _), s in zip(cands, score)})
    print(f"{ds} k={best_k} stab={ari_stab:.2f} rm={frac_rm:.1%} fit={v2_results[ds][5]} "
          f"win={cands[i][0]:<16} ARI(v1,v2)={ari12:.3f} t={dt:.1f}s  "
          f"{ {nm: round(float(s),3) for (nm,_),s in zip(cands,score)} }")

import json
json.dump({k: {'winner': v[1], 'k': v[2], 'stab': v[3], 'frac_rm': v[4], 'fitset': v[5],
               't': round(v[6], 2), 'ari_v1v2': round(float(v[7]), 3)} for k, v in v2_results.items()},
          open('analysis/v2_compare.json', 'w'), indent=1)
print("saved analysis/v2_compare.json | total time %.0fs" % sum(v[6] for v in v2_results.values()))
