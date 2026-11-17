import importlib.util, time, zipfile, os, hashlib
import numpy as np, pandas as pd

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

champ = load('my_solution_final/solution.py', 'solc')
probe = load('probes/k2_ds03_ds13/solution.py', 'solp')

df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
diff, same = [], []
t_all = 0
for ds, g in df.groupby('dataset_id'):
    X = g[xcols].values.astype(float)
    t0 = time.time()
    lc = np.asarray(champ.cluster(X, ds), dtype=int)
    lp = np.asarray(probe.cluster(X, ds), dtype=int)
    rt = time.time() - t0
    t_all += rt
    if np.array_equal(np.asarray(lc), np.asarray(lp)):
        same.append(ds)
    else:
        sc = np.bincount(np.asarray(lc, dtype=int)); sp = np.bincount(np.asarray(lp, dtype=int))
        diff.append((ds, list(sc), list(sp)))

print(f"identical: {len(same)}/24 datasets;  total runtime all-24: {t_all:.1f}s")
for ds, sc, sp in diff:
    print(f"CHANGED {ds}: champion sizes={sc} -> probe sizes={sp}")

# zip exactly like other probes: solution.py + notebook.ipynb at root
zp = 'probes/k2_ds03_ds13/submission.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('probes/k2_ds03_ds13/solution.py', 'solution.py')
    z.write('probes/k2_ds03_ds13/notebook.ipynb', 'notebook.ipynb')
print("wrote", zp, os.path.getsize(zp), "bytes")

# reference copy of the champion package (same convention as ds19parab)
import shutil
shutil.copy('my_solution_final/submission.zip', 'probes/k2_ds03_ds13/solution_old_DO_NOT_SUBMIT.zip')

# integrity: unzipped solution.py must match the file we tested
z = zipfile.ZipFile(zp)
assert z.read('solution.py') == open('probes/k2_ds03_ds13/solution.py','rb').read()
assert z.read('notebook.ipynb') == open('probes/k2_ds03_ds13/notebook.ipynb','rb').read()
print("zip integrity OK")
