# -*- coding: utf-8 -*-
"""Final checks for v4: scoring-path timing on all 24 datasets + artifact size options."""
import base64, gzip, importlib.util, io, os, re, sys, time
import numpy as np, pandas as pd

ROOT = r'C:\AI_master\ML\hw_klasters'
os.chdir(ROOT)
out = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n')

spec = importlib.util.spec_from_file_location('v4', 'dev_v4/solution_v4.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# 1. import audit (only allowed third-party libs)
allowed = {'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn'}
banned = []
src = open('dev_v4/solution_v4.py', encoding='utf-8').read()
for line in src.splitlines():
    mm = re.match(r'\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_.]*)', line)
    if mm and mm.group(1).split('.')[0] not in allowed:
        banned.append(line.strip())
p('IMPORT AUDIT: ' + ('OK (only allowed libs)' if not banned else str(banned)))

# 2. scoring-path simulation: 20% test split per dataset, like hidden grading
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
rng = np.random.RandomState(0)
tot = 0.0; worst = ('', 0.0); ks = []
for ds, g in df.groupby('dataset_id'):
    X = g[xcols].values.astype(float)
    n_te = max(15, int(0.2 * len(X)))
    t0 = time.time()
    info = m.explain(X[:n_te], ds)
    dt = time.time() - t0; tot += dt
    if dt > worst[1]: worst = (ds, dt)
    ks.append(len(np.unique(info['labels'])))
p('TIMING: total %.1fs  worst %s %.2fs' % (tot, *worst))
p('k per ds:', dict(zip(sorted(df.dataset_id.unique()), ks)))
p('ART size KB: %d' % (len(src) // 1024))

# 3. artifact size options: how much can ztr shrink with rounding?
df_a = df.groupby('dataset_id')
tot_raw = tot_gz_f32 = tot_gz_r2 = tot_gz_r1 = 0
for ds, g in df_a:
    Xc = m._clean(g[xcols].values.astype(float))
    a = m.ART[ds]
    Zraw = m._apply_norm(Xc, a['norm'], a['nprm'])
    Ztr = m._pca_transform(Zraw, a['d_pca'], np.asarray(a['pmean']), np.asarray(a['pcomp']))
    b0 = len(gzip.compress(np.ascontiguousarray(Ztr.astype(np.float32)), 9))
    b1 = len(gzip.compress(np.ascontiguousarray(np.round(Ztr, 2).astype(np.float32)), 9))
    b2 = len(gzip.compress(np.ascontiguousarray(np.round(Ztr, 1).astype(np.float32)), 9))
    tot_raw += Ztr.nbytes; tot_gz_f32 += b0; tot_gz_r2 += b1; tot_gz_r1 += b2
p('ztr bytes: raw=%d  gz_f32=%d  gz_round2=%d  gz_round1=%d' %
  (tot_raw, tot_gz_f32, tot_gz_r2, tot_gz_r1))
open('dev_v4/final_check.txt', 'w').write(out.getvalue())
