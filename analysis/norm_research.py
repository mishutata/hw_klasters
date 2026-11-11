"""Research: which normalization method works best for each dataset.

Runs the full v2 pipeline (k-selection + stability/core-trim + KMeans/GMM/Spectral
+ DBSCAN/HDBSCAN confirmers) under 7 different normalizations per dataset and
records the final ensemble score, silhouette, winner, k and timing.
"""
import importlib.util, json, sys, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler,
                                   QuantileTransformer, PowerTransformer)

spec = importlib.util.spec_from_file_location("v2", r"../my_solution2/solution.py")
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)

def apply_norm(X, method):
    X = np.asarray(X, float)
    if method == 'standard':  out = StandardScaler().fit_transform(X)
    elif method == 'robust':  out = RobustScaler().fit_transform(X)
    elif method == 'minmax':  out = MinMaxScaler().fit_transform(X)
    elif method == 'quantile':
        q = min(len(X), 1000)
        out = QuantileTransformer(n_quantiles=q, output_distribution='normal',
                                  random_state=42).fit_transform(X)
    elif method == 'yeo-johnson':
        out = PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(X)
    elif method == 'box-cox':
        Xs = np.where(np.min(X, axis=0) <= 0, X - np.nanmin(X, axis=0) + 1.0, X)
        out = PowerTransformer(method='box-cox', standardize=True).fit_transform(Xs)
    elif method == 'log-sym':
        sgn = np.sign(X); Xa = np.log1p(np.abs(X)) * sgn
        out = StandardScaler().fit_transform(Xa)
    else: raise ValueError(method)
    return out

def make_reduce(method):
    def _reduce(X):
        Xn = np.where(np.isfinite(apply_norm(X, method)), 0.0, 0.0) if False else apply_norm(X, method)
        Xn = np.where(np.isfinite(Xn), Xn, 0.0)
        if X.shape[1] > 3:
            p = PCA(random_state=42).fit(Xn)
            cum = np.cumsum(p.explained_variance_ratio_)
            ncomp = max(2, min(8, int(np.searchsorted(cum, 0.95) + 1), X.shape[1]))
            Xn = PCA(n_components=ncomp, random_state=42).fit_transform(Xn)
        return Xn
    return _reduce

METHODS = ['standard', 'robust', 'minmax', 'quantile', 'yeo-johnson', 'box-cox', 'log-sym']

df = pd.read_csv(r'../public_data/train.csv')
groups = {ds: g[['id','dataset_id']].shape[1] and g for ds, g in df.groupby('dataset_id')}

rows, t0 = [], time.time()
out = open('norm_research.csv', 'w')
out.write("dataset_id,method,k_prior,k_used,sil_best,ens_score,winner,km_stab,core_rm,secs\n")
for ds in sorted(groups):
    X = df[df['dataset_id'] == ds].drop(columns=['id','dataset_id']).values.astype(float)
    for m in METHODS:
        v2._reduce = make_reduce(m)
        t1 = time.time()
        try:
            lab, info = v2._run(X, ds)
            best = max(info['methods'].values()) if info['methods'] else -1.0
            rows.append(dict(dataset_id=ds, method=m, k_prior=info['k_prior'],
                             k_used=info['k_used'], sil_best=round(best,3),
                             ens_score=round(best,3), winner=info['winner'],
                             km_stab=info['km_stability'], core_rm=info['core_frac_removed'],
                             secs=round(time.time()-t1,2)))
        except Exception as e:
            rows.append(dict(dataset_id=ds, method=m, k_prior='', k_used='', sil_best=np.nan,
                             ens_score=-9.0, winner=f'ERR:{type(e).__name__}', km_stab='', core_rm='',
                             secs=round(time.time()-t1,2)))
        r = rows[-1]
        out.write(f"{r['dataset_id']},{r['method']},{r['k_prior']},{r['k_used']},{r['sil_best']},{r['ens_score']},{r['winner']},{r['km_stab']},{r['core_rm']},{r['secs']}\n")
        out.flush()
print(f"done in {time.time()-t0:.0f}s", file=sys.stderr)

# per-dataset winner table
R = pd.DataFrame(rows)
piv = R.pivot(index='dataset_id', columns='method', values='ens_score')
best_m = piv.idxmax(axis=1); best_s = piv.max(axis=1)
print("\n=== best normalization per dataset ===")
for ds in sorted(groups):
    print(f"{ds}: {best_m[ds]:>12}  score={best_s[ds]:.3f}")
R.to_json('norm_research.json', orient='records')
json.dump({ds: {'method': best_m[ds], 'score': float(best_s[ds])} for ds in sorted(groups)},
          open('norm_best.json','w'), indent=1)
