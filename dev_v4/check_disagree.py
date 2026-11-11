# -*- coding: utf-8 -*-
"""ds08/ds13/ds14: which partition (v3 vs v4) is geometrically better?
Common judge space: norm+PCA fitted on the 80% holdout train."""
import importlib.util, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import silhouette_score

ROOT = r'C:\AI_master\ML\hw_klasters'
sys.path.insert(0, ROOT); os.chdir(ROOT)
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
sol3 = load('sol3', 'dev_v4/sol3_backup.py')
sol4 = load('sol4', 'dev_v4/solution_v4.py')

rng = np.random.RandomState(42)
for ds in ('ds08', 'ds13', 'ds14'):
    g = df[df['dataset_id'] == ds]
    Xfull = sol4._clean(g[xcols].values.astype(float))   # consistent columns
    idx = rng.permutation(len(Xfull)); n_te = max(15, int(0.2*len(Xfull)))
    te_i, tr_i = idx[:n_te], idx[n_te:]
    L3, i3 = sol3._run(Xfull[te_i], ds)
    Xtr, Xte = Xfull[tr_i], Xfull[te_i]
    nm = sol4.NORM_PRIOR.get(ds, 'minmax')
    prm = sol4._fit_norm_params(Xtr, nm)
    d_pca, pm, pc = sol4._fit_pca(np.nan_to_num(sol4._apply_norm(Xtr, nm, prm)))
    Ztr = sol4._pca_transform(np.nan_to_num(sol4._apply_norm(Xtr, nm, prm)), d_pca, pm, pc)
    Zte = sol4._pca_transform(np.nan_to_num(sol4._apply_norm(Xte, nm, prm)), d_pca, pm, pc)
    L4, i4 = sol4._pipeline(Ztr, Zte, ds)
    lab_te = L4[len(Ztr):]
    print(ds, '| n_tr=%d n_te=%d dim=%d d_pca=%d' % (len(Xtr), len(Xte), Xfull.shape[1], d_pca))
    for tag, lab in (('v3', L3), ('v4', lab_te)):
        u = np.unique(lab)
        s = round(silhouette_score(Zte, lab), 3) if len(u) >= 2 else None
        cents = np.array([Zte[lab == c].mean(0) for c in u])
        intra = float(np.mean([np.linalg.norm(Zte[lab == c] - cents[k], axis=1).mean() for k, c in enumerate(u)]))
        inter = float(np.mean([np.linalg.norm(cents[k] - cents[j]) for k in range(len(u)) for j in range(k + 1, len(u))]))
        print('  %s: n_cl=%d sil_Z=%.3f intra=%.3f inter=%.3f ratio=%.2f' % (tag, len(u), s, intra, inter, inter / max(intra, 1e-9)))
    print('  v4 winner=%s k_used=%s | v3 k=%s' % (i4['winner'], i4['k_used'], i3['k_used']))
