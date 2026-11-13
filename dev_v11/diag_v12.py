# -*- coding: utf-8 -*-
"""v12 candidate diagnostics on public train (my_solution12 == v6 base).

For each of the 24 datasets:
  A) PCA variance kept at v6's cap (min(8, dims@95%)) vs full rank
  B) explain() winner / k_used / score margin (top1-top2), km_stability
  C) test-size fragility: winner and k_used over 4 random 20% subsamples
     vs the full-train decision (hidden test is ~20% of each ds)
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SKIP_PART1 = '--skip-part1' in sys.argv  # parts 1+2 already captured in earlier run

spec = importlib.util.spec_from_file_location('sol', 'my_solution12/solution.py')
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]

# ---------------- part 2: median-distance PCA rule vs v6 cap-8 ----------------
# Rule (user proposal, no hard cap): keep components while
#   median_i sqrt(sum_{j>k} T_ij^2)  <=  tau * median_pairwise_dist(Z)
# i.e. reconstruction error in Z-space below a fraction of the typical
# point-to-point distance; medians throughout (robust to outliers).
from sklearn.metrics import adjusted_rand_score

def pca_dims_median(Xc, norm, tau, rng):
    Z = sol._apply_norm(Xc, norm)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    if Xc.shape[1] <= 3:
        return Xc.shape[1], Z
    p = PCA(random_state=42).fit(Z)
    T = p.transform(Z)                      # scores
    sq = T ** 2
    tail = np.cumsum(sq[:, ::-1], axis=1)[:, ::-1]   # tail[i, k] = sum_{j>k} T_ij^2
    err = np.sqrt(np.maximum(tail, 0.0))   # per-point recon error for each k
    m = min(len(Z), 300)
    idx = rng.choice(len(Z), size=m, replace=False) if len(Z) > m else np.arange(len(Z))
    Zs = Z[idx]
    d2 = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(m, k=1)
    dmed = float(np.median(np.sqrt(d2[iu])))
    med_err = np.median(err, axis=0)       # median over points, per k
    p_full = Xc.shape[1]
    for k in range(2, p_full + 1):
        if med_err[k - 1] <= tau * dmed:
            ncomp = max(2, k)
            break
    else:
        ncomp = p_full
    Zr = PCA(n_components=ncomp, random_state=42).fit_transform(Z)
    return ncomp, Zr

print('\n=== median-distance rule: chosen dims per ds (full train) ===')
rng3 = np.random.RandomState(0)
hdr = f"{'ds':6}" + ''.join(f" {'tau='+str(t):>8}" for t in (0.1, 0.2, 0.3)) + f" | {'v6':>4} | dims@95%"
print(hdr)
for ds, g in df.groupby('dataset_id'):
    X = g[xcols].values.astype(float)
    Xc = sol._clean(X)
    norm = sol.NORM_PRIOR.get(ds, 'minmax')
    Zraw = np.where(np.isfinite(sol._apply_norm(Xc, norm)), 0.0, 0.0)
    pf = PCA(random_state=42).fit(sol._apply_norm(Xc, norm))
    cum = np.cumsum(pf.explained_variance_ratio_)
    d95 = int(np.searchsorted(cum, 0.95) + 1)
    dims = []
    for t in (0.1, 0.2, 0.3):
        nc, _ = pca_dims_median(Xc, norm, t, rng3)
        dims.append(nc)
    v6n = max(2, min(8, d95, Xc.shape[1]))
    print(f"{ds:6}" + ''.join(f" {d:>8}" for d in dims) + f" | {v6n:>4} | {d95:>8}")

# ---------------- part 3: end-to-end label shift vs v6 for each tau ----------------
import importlib.util as iu2
from sklearn.metrics import adjusted_rand_score

def load(name, path):
    s = iu2.spec_from_file_location(name, path)
    m = iu2.module_from_spec(s)
    s.loader.exec_module(m)
    return m

v6m = load('v6m', 'my_solution6/solution.py')
m12s = {}
for t in (0.1, 0.2, 0.3):
    mm = load(f'v12_{t}', 'my_solution12/solution.py')
    mm.MEDIAN_TAU = t
    m12s[t] = mm
rng4 = np.random.RandomState(7)
rows = []
for ds, g in df.groupby('dataset_id'):
    X = g[xcols].values.astype(float)
    n = len(X)
    lab6 = v6m.explain(X, ds)['labels']
    rec = [ds, n]
    for t in (0.1, 0.2, 0.3):
        m12 = m12s[t]
        i12 = m12.explain(X, ds)
        ari_full = adjusted_rand_score(lab6, i12['labels'])
        aris = []
        for _ in range(4):
            idx = rng4.choice(n, size=max(8, n // 5), replace=False)
            a = v6m.explain(X[idx], ds)['labels']
            b = m12.explain(X[idx], ds)['labels']
            aris.append(adjusted_rand_score(a, b))
        rec += [ari_full, np.mean(aris), i12['winner']]
    rows.append(rec)

print('\n=== end-to-end: v6 vs median-rule (full-train ARI | mean test-size ARI) ===')
print(f"{'ds':6} {'n':>5} | " + ' | '.join(f"ARI{t}: full/test  winner@{t}" for t in (0.1, 0.2, 0.3)))
for r in rows:
    ds, n = r[0], r[1]
    parts = []
    for i in range(3):
        af, at, w = r[2 + i * 3], r[3 + i * 3], r[4 + i * 3]
        parts.append(f"{af:5.3f}/{at:5.3f}   {w[:4]:4}")
    print(f"{ds:6} {n:>5} | " + ' | '.join(parts))

if SKIP_PART1:
    print('DONE parts 2+3', flush=True)
    raise SystemExit(0)

print(f"{'ds':6} {'n':>5} {'p':>3} | {'var8':>6} {'varF':>6} | {'winner':14} {'k0':>2} {'kU':>2} {'margin':>7} {'stab':>5} | frag(winner/k)")
print('-' * 100)
rng = np.random.RandomState(0)
for ds, g in df.groupby('dataset_id'):
    X = g[xcols].values.astype(float)
    n, p = X.shape
    Xc = sol._clean(X)
    Zraw = sol._apply_norm(Xc, sol.NORM_PRIOR.get(ds, 'minmax'))
    Zraw = np.where(np.isfinite(Zraw), Zraw, 0.0)
    if p > 3:
        pf = PCA(random_state=42).fit(Zraw)
        cum = np.cumsum(pf.explained_variance_ratio_)
        d95 = int(np.searchsorted(cum, 0.95) + 1)
        n_used = max(2, min(8, d95, p))
        var_kept = float(cum[n_used - 1])
    else:
        d95, n_used, var_kept = p, p, 1.0
    info = sol.explain(X, ds)
    m = sorted(info['methods'].values(), reverse=True)
    margin = round(m[0] - m[1], 3) if len(m) > 1 else float('nan')
    # test-size fragility: 4 random 20% subsamples
    wins, ks = [], []
    for _ in range(4):
        idx = rng.choice(n, size=max(8, n // 5), replace=False)
        i2 = sol.explain(X[idx], ds)
        wins.append(i2['winner'])
        ks.append(i2['k_used'])
    frag_w = sum(w != info['winner'] for w in wins)
    frag_k = sum(k != info['k_used'] for k in ks)
    print(f"{ds:6} {n:>5} {p:>3} | {var_kept:>6.3f} {min(1.0, float(cum[-1])) if p>3 else 1.0:>6.3f} "
          f"| {info['winner']:14} {info['k_prior']:>2} {info['k_used']:>2} {margin:>7.3f} {info['km_stability']:>5.2f} "
          f"| {frag_w}/4 {frag_k}/4  k~{sorted(set(ks))}")
