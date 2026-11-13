# -*- coding: utf-8 -*-
"""v15 pin search: apply the v14 M1/M2 logic to the 13 free-vote datasets.
Stage 1 (this file): static read of dev_v14/stability_results.csv + tier1 CSVs
-> per-dataset table: current pipe_stab, current-config M1/M2 scores, top grid
candidates, and a flag if the evidence suggests a pin.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

g = pd.read_csv('dev_v14/stability_results.csv')
t1 = pd.read_csv('dev_v14/tier1_pipeline_stability.csv').set_index('ds')
dec = pd.read_csv('dev_v14/v5_decisions.csv').set_index('ds')

K_PRIOR = {"ds00": 2, "ds01": 3, "ds02": 2, "ds03": 2, "ds04": 3, "ds05": 3,
    "ds06": 3, "ds07": 3, "ds08": 2, "ds09": 3, "ds10": 3, "ds11": 4,
    "ds12": 2, "ds13": 2, "ds14": 2, "ds15": 4, "ds16": 2, "ds17": 2,
    "ds18": 2, "ds19": 4, "ds20": 4, "ds21": 3, "ds22": 2, "ds23": 2}
K_PIN = {"ds04": 3, "ds09": 3, "ds16": 4, "ds18": 4, "ds19": 4, "ds20": 4, "ds23": 4}
V14_PINS = {'ds00': (3, 'KMeans'), 'ds08': (None, 'KMeans'),
            'ds10': (None, 'KMeans'), 'ds13': (None, 'SpectralClustering')}
NORM_PRIOR = {"ds00": "log-sym", "ds05": "log-sym", "ds06": "log-sym",
              "ds07": "log-sym", "ds10": "log-sym", "ds18": "log-sym",
              "ds21": "log-sym", "ds22": "log-sym", "ds02": "robust"}
FREE = [d for d in sorted(dec.index)
        if d not in K_PIN and d not in V14_PINS]
print('free-vote datasets (%d):' % len(FREE), FREE)
print()

rows = []
for ds in FREE:
    cur_k = K_PRIOR[ds]
    cur_w = dec.loc[ds, 'winner']
    norm = NORM_PRIOR.get(ds, 'minmax')
    t1r = t1.loc[ds]
    sub = g[(g.ds == ds) & (g.dim == 'auto') & (g.norm == norm)]
    # current winner's grid score at current (norm, k)
    cur_score = np.nan
    if cur_w == 'KMeans':
        r0 = sub[(sub.k == cur_k) & (sub.alg == 'KMeans')]
        cur_score = r0.M1.iloc[0] if len(r0) else np.nan
    elif cur_w == 'GaussianMixture':
        r0 = sub[(sub.k == cur_k) & (sub.alg == 'GaussianMixture')]
        cur_score = r0.M1.iloc[0] if len(r0) else np.nan
    elif cur_w == 'SpectralClustering':
        r0 = sub[(sub.k == cur_k) & (sub.alg == 'Spectral')]
        cur_score = r0.M2.iloc[0] if len(r0) else np.nan
    # top grid candidates overall (any norm/dim)
    dsg = g[g.ds == ds]
    m1c = dsg[dsg.alg.isin(['KMeans', 'GaussianMixture']) & dsg.M1.notna()]
    m1c = m1c.sort_values('M1', ascending=False).head(3)
    m2c = dsg[(dsg.alg == 'Spectral') & (dsg.M2.notna())].sort_values('M2', ascending=False).head(1)
    top_str = ' | '.join(
        f"{r.alg} k={r.k} {r.norm}/{r.dim} M1={r.M1:.3f}" for r in m1c.itertuples())
    top_m2 = (' | Spectral ' + ' '.join(
        f"k={r.k} {r.norm}/{r.dim} M2={r.M2:.3f}" for r in m2c.itertuples())) if len(m2c) else ''
    flag = t1r.pipe_stab < 0.95 or (cur_score == cur_score and cur_score < 0.80)
    rows.append({'ds': ds, 'k': cur_k, 'winner_now': cur_w, 'norm': norm,
                 'pipe_stab': round(t1r.pipe_stab, 3),
                 'cur_grid_score': round(cur_score, 3) if cur_score == cur_score else np.nan,
                 'FLAG': flag})
    print(f"{ds} k={cur_k} {cur_w:<20s} pipe_stab={t1r.pipe_stab:.3f} "
          f"cur_score={cur_score if cur_score == cur_score else float('nan'):.3f}")
    print(f"   top M1: {top_str}{top_m2}")

out = pd.DataFrame(rows).set_index('ds')
out.to_csv('dev_v14/v15_pin_stage1.csv')
print()
print(out.to_string())
print('\nFLAGGED:', out[out.FLAG].index.tolist())

# ---------------------------------------------------------------- stage 2
# End-to-end head-to-head at test size with the EXACT v14 pipeline:
# variant A = as-is; variant B = grid top candidate (pinned k/norm/winner).
# Score: 8-slice transfer ARI vs full-data partition + winner tally.
import importlib.util
spec = importlib.util.spec_from_file_location('sol14', 'my_solution14/solution.py')
sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
from sklearn.metrics import adjusted_rand_score

df = pd.read_csv('public_data/train.csv')
XS = [c for c in df.columns if c.startswith('x')]
DATA = {ds: sol._clean(g[XS].values) for ds, g in df.groupby('dataset_id')}

def slices_of(n, r=8, frac=0.25):
    rng = np.random.RandomState(42)
    m = max(int(frac * n), 20)
    return [rng.choice(n, m, replace=False) for _ in range(r)]

def eval_variant(ds, X, pin_k=None, pin_w=None, norm=None):
    saved = (sol.K_PIN.get(ds), sol.WINNER_PIN.get(ds), sol.NORM_PRIOR.get(ds))
    if pin_k is not None: sol.K_PIN[ds] = pin_k
    if pin_w is not None: sol.WINNER_PIN[ds] = pin_w
    if norm is not None: sol.NORM_PRIOR[ds] = norm
    try:
        lab_full = sol.cluster(X, ds)
        aris, wins = [], []
        for ix in slices_of(len(X)):
            lab_i, info = sol._run(X[ix], ds)
            aris.append(adjusted_rand_score(lab_i, lab_full[ix]))
            wins.append(info['winner'])
        return round(float(np.mean(aris)), 3), round(float(min(aris)), 3), '/'.join(sorted(set(wins)))
    finally:
        if saved[0] is None: sol.K_PIN.pop(ds, None)
        else: sol.K_PIN[ds] = saved[0]
        if saved[1] is None: sol.WINNER_PIN.pop(ds, None)
        else: sol.WINNER_PIN[ds] = saved[1]
        if saved[2] is None: sol.NORM_PRIOR.pop(ds, None)
        else: sol.NORM_PRIOR[ds] = saved[2]

# (ds, variant-B spec) from stage-1 grid top candidates
VARIANTS = {
    'ds05': dict(pin_k=3, pin_w='KMeans', norm='log-sym'),
    'ds06': dict(pin_k=3, pin_w='KMeans', norm='log-sym'),
    'ds07': dict(pin_k=3, pin_w='KMeans', norm='log-sym'),
    'ds14': dict(pin_k=2, pin_w='SpectralClustering', norm='log-sym'),
    'ds15': dict(pin_k=2, pin_w='KMeans', norm='robust'),
    'ds21': dict(pin_k=3, pin_w='KMeans', norm='log-sym'),
}
res = []
for ds in ['ds01', 'ds05', 'ds06', 'ds07', 'ds14', 'ds15', 'ds21', 'ds22']:
    X = DATA[ds]
    a = eval_variant(ds, X)
    row = {'ds': ds, 'A_mean': a[0], 'A_min': a[1], 'A_winners': a[2]}
    if ds in VARIANTS:
        b = eval_variant(ds, X, **VARIANTS[ds])
        row.update({'B_spec': str(VARIANTS[ds]), 'B_mean': b[0], 'B_min': b[1], 'B_winners': b[2]})
    res.append(row)
    print(row, flush=True)
pd.DataFrame(res).set_index('ds').to_csv('dev_v14/v15_pin_stage2.csv')
