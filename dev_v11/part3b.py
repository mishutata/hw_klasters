"""Part 3b: v6 vs median-rule (tau=0.1/0.2/0.3) at TEST size only, chunked.
Usage: python part3b.py <i0> <i1>   (chunk over sorted dataset_ids)
Per ds: S shared random subsamples of size n//5; on each, run v6 + 3 tau variants
on the SAME rows; report mean ARI(v6, tau) and winners.
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

v6m = load('v6m', 'my_solution6/solution.py')
TAUS = (0.1, 0.2, 0.3)
mods = {}
for t in TAUS:
    mm = load(f'v12_{t}', 'my_solution12/solution.py')
    mm.MEDIAN_TAU = t
    mods[t] = mm

df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
dss = sorted(df['dataset_id'].unique())
i0, i1 = int(sys.argv[1]), int(sys.argv[2])
S = 3
rng = np.random.RandomState(7)

for ds in dss[i0:i1]:
    g = df[df['dataset_id'] == ds]
    X = g[xcols].values.astype(float)
    n = len(X)
    subs = [rng.choice(n, size=max(8, n // 5), replace=False) for _ in range(S)]
    ari_acc = {t: [] for t in TAUS}
    w6s, w12s = [], {t: [] for t in TAUS}
    for idx in subs:
        r6 = v6m.explain(X[idx], ds)
        w6s.append(r6['winner'])
        for t in TAUS:
            r12 = mods[t].explain(X[idx], ds)
            ari_acc[t].append(adjusted_rand_score(r6['labels'], r12['labels']))
            w12s[t].append(r12['winner'])
    line = f"{ds:6} {n:>5} |"
    for t in TAUS:
        line += f"  t={t}: ARI={np.mean(ari_acc[t]):.3f} v6w={w6s[0]} w12={'/'.join(w12s[t])}"
    print(line, flush=True)
print(f"CHUNK {i0}-{i1} DONE", flush=True)
