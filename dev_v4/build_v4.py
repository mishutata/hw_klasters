# -*- coding: utf-8 -*-
"""Build v4 solution (public-train artifacts embedded) + holdout simulation
comparing v3 (cluster test-only) vs v4 (pooled transductive pipeline)."""
import json, os, shutil, sys, time
import numpy as np, pandas as pd

ROOT = r'C:\AI_master\ML\hw_klasters'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

df = pd.read_csv(os.path.join('public_data', 'train.csv'))
xcols = [c for c in df.columns if c.startswith('x')]

tpl = open('dev_v4/solution_template.py', encoding='utf-8').read()
assert '__ART__' in tpl

# backup of current (v3) final solution for comparison
shutil.copy('my_solution_final/solution.py', 'dev_v4/sol3_backup.py')
import importlib.util
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
sol3 = load('sol3', 'dev_v4/sol3_backup.py')

# ---------- 1. build artifacts from FULL public train ----------
def make_artifacts(mod, groups):
    art = {}
    for ds, g in groups:
        Xc = mod._clean(g[xcols].values.astype(float))
        nm = mod.NORM_PRIOR.get(ds, 'minmax')
        prm = mod._fit_norm_params(Xc, nm)
        d_pca, pmean, pcomp = mod._fit_pca(mod._apply_norm(Xc, nm, prm))
        Zraw = mod._apply_norm(Xc, nm, prm)
        Zraw = np.where(np.isfinite(Zraw), Zraw, 0.0)
        Ztr = mod._pca_transform(Zraw, d_pca, pmean, pcomp).astype(np.float32)
        # round-trip check: transform must equal direct PCA on train points
        from sklearn.decomposition import PCA as _PCA
        if d_pca < Zraw.shape[1]:
            ref = _PCA(n_components=d_pca, random_state=42).fit_transform(Zraw)
            assert np.allclose(Ztr, ref, atol=1e-3), ds
        art[ds] = {'d_in': int(Xc.shape[1]), 'norm': nm,
                   'nprm': [[float(v) for v in a] for a in prm],
                   'pmean': [float(v) for v in pmean],
                   'pcomp': [[float(v) for v in r] for r in pcomp],
                   'd_pca': int(d_pca), 'n_tr': int(len(Ztr)),
                   'ztr': np.ascontiguousarray(Ztr).tobytes().hex()}
    return art

open('dev_v4/solution_v4.py', 'w', encoding='utf-8').write(tpl)  # ART=None stub
sol4 = load('sol4', 'dev_v4/solution_v4.py')
art_full = make_artifacts(sol4, df.groupby('dataset_id'))
txt = tpl.replace('ART = None  # __ART__', 'ART = ' + repr(art_full))
open('dev_v4/solution_v4.py', 'w', encoding='utf-8').write(txt)
sol4 = load('sol4b', 'dev_v4/solution_v4.py')
print('built dev_v4/solution_v4.py, size KB:', round(len(txt)/1024))

# ---------- 2. holdout simulation: v3 vs v4 ----------
def run_v4_holdout(mod, ds, tr_g, te_g):
    """artifacts from TR only; label TE points via pooled pipeline."""
    Xtr = mod._clean(tr_g[xcols].values.astype(float))
    Xte = mod._clean(te_g[xcols].values.astype(float))
    nm = mod.NORM_PRIOR.get(ds, 'minmax')
    prm = mod._fit_norm_params(Xtr, nm)
    d_pca, pmean, pcomp = mod._fit_pca(mod._apply_norm(Xtr, nm, prm))
    Zraw_tr = mod._apply_norm(Xtr, nm, prm); Zraw_tr = np.where(np.isfinite(Zraw_tr), Zraw_tr, 0.0)
    Zraw_te = mod._apply_norm(Xte, nm, prm); Zraw_te = np.where(np.isfinite(Zraw_te), Zraw_te, 0.0)
    Ztr = mod._pca_transform(Zraw_tr, d_pca, pmean, pcomp)
    Zte = mod._pca_transform(Zraw_te, d_pca, pmean, pcomp)
    labs, info = mod._pipeline(Ztr, Zte, ds)
    return labs[len(Ztr):], info

from sklearn.metrics import adjusted_rand_score
rng = np.random.RandomState(42)
rows = []
for ds, g in df.groupby('dataset_id'):
    X = g[xcols].values.astype(float)
    idx = rng.permutation(len(X))
    n_te = max(15, int(0.2*len(X)))
    te_i, tr_i = idx[:n_te], idx[n_te:]
    t0 = time.time()
    L3, i3 = sol3._run(X[te_i], ds)
    rt3 = time.time() - t0
    wins4, ks4, ari34s, rts4 = [], [], [], []
    for seed in (0, 1, 2):   # 3 re-splits: winner/k must be stable
        rng2 = np.random.RandomState(seed)
        idx2 = rng2.permutation(len(X))
        n_te2 = max(15, int(0.2*len(X)))
        te_pts = X[idx2[:n_te2]]
        L3s, _ = sol3._run(te_pts, ds)   # v3 on the SAME points as v4
        t0 = time.time()
        L4, i4 = run_v4_holdout(sol4, ds, g.iloc[idx2[n_te2:]], g.iloc[idx2[:n_te2]])
        rts4.append(time.time() - t0)
        wins4.append(i4['winner']); ks4.append(i4['k_used'])
        ari34s.append(adjusted_rand_score(L3s, L4))
    rows.append({'ds': ds, 'k3': i3['k_used'], 'ks4': '/'.join(map(str, ks4)),
                 'w4': wins4[0] if len(set(wins4)) == 1 else 'UNSTABLE:' + '|'.join(wins4),
                 'ag34': round(float(np.mean(ari34s)), 3),
                 'rt3': round(rt3, 2), 'rt4max': round(max(rts4), 2)})

T = pd.DataFrame(rows).set_index('ds')
open('dev_v4/holdout_report.txt', 'w').write(T.to_string())
print('written dev_v4/holdout_report.txt; mean ag34 =',
      round(float(T.ag34.mean()), 3), '; max rt4 =', T.rt4max.max())