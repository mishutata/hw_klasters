"""Build + verify the two single-change k=2 probes (one-probe strategy).

probes/k2_ds03  = q6 + ds03:(3->2,'gmm')      only
probes/ds13k2   = q6 + ds13:(3->2,'gmm_tied') only
vs q6 (my_solution_final, BEST 0.538102). Combo probe k2_ds03_ds13 scored 0.536072.
"""
import importlib.util, time, zipfile, os, shutil
import numpy as np, pandas as pd

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

champ = load('my_solution_final/solution.py', 'solc')
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}

for pname in ['k2_ds03', 'ds13k2']:
    probe = load(f'probes/{pname}/solution.py', f'sol_{pname}')
    diff, same, t_all = [], [], 0.0
    for ds, X in groups.items():
        t0 = time.time()
        lc = np.asarray(champ.cluster(X, ds), dtype=int)
        lp = np.asarray(probe.cluster(X, ds), dtype=int)
        t_all += time.time() - t0
        if np.array_equal(lc, lp):
            same.append(ds)
        else:
            diff.append((ds, list(np.bincount(lc)), list(np.bincount(lp))))
    print(f"== {pname}: identical {len(same)}/24; runtime all-24 {t_all:.1f}s")
    for ds, sc, sp in diff:
        print(f"   CHANGED {ds}: q6 sizes={sc} -> probe sizes={sp}")

for pname in ['k2_ds03', 'ds13k2']:
    zp = f'probes/{pname}/submission.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(f'probes/{pname}/solution.py', 'solution.py')
        z.write(f'probes/{pname}/notebook.ipynb', 'notebook.ipynb')
    shutil.copy('my_solution_final/submission.zip', f'probes/{pname}/solution_old_DO_NOT_SUBMIT.zip')
    z = zipfile.ZipFile(zp)
    assert z.read('solution.py') == open(f'probes/{pname}/solution.py','rb').read()
    assert z.read('notebook.ipynb') == open(f'probes/{pname}/notebook.ipynb','rb').read()
    print(f"wrote {zp} ({os.path.getsize(zp)} bytes); zip integrity OK")