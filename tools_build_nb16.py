# -*- coding: utf-8 -*-
"""Build my_solution16/notebook.ipynb (minimal: participant data + solution
text + the mandatory 'Выводы' section), execute it, and zip
solution.py + notebook.ipynb -> my_solution16/submission.zip."""
import json, os, subprocess, zipfile
import nbformat as nbf

ROOT = r'C:\AI_master\ML\hw_klasters'
os.chdir(ROOT)
NB = 'my_solution16/notebook.ipynb'
ZP = 'my_solution16/submission.zip'

TITLE = """# Clustering Challenge — v16 solution"""

DATA = """## Данные участника
ФИО: Борцов Михаил Алексеевич
Email: mialbortsov@edu.hse.ru
ВУЗ: ВШЭ"""

SOLUTION = """## Solution

The task is to cluster 24 hidden datasets (ds00–ds23); the metric is Mean ARI
against the hidden labels. The hidden test of each dataset is a fresh ~20%
sample of the same population as the public train (the split is stratified over
the true clusters). `solution.py` therefore runs, per dataset:

1. **Cleaning** — all-NaN trailing columns are dropped, remaining NaNs are
   filled with the column mean; per-dataset normalization (minmax /
   log-symmetric / robust, selected on public train); PCA to 95% variance
   (≤ 8 components).
2. **Number of clusters** — a per-dataset prior refined by a ±1 silhouette
   window (margin 0.04); four datasets with verified choices are pinned
   (ds00 k=3+KMeans, ds08 KMeans, ds10 KMeans, ds13 Spectral).
3. **Clustering** — KMeans seed-stability probe with density-based core-set
   trimming; ensemble vote of KMeans / GMM / Spectral (score = silhouette +
   0.3 × mean consensus NMI); DBSCAN and HDBSCAN participate only as
   confirmers (low noise and NMI ≥ 0.5 with the best parametric candidate).
4. **Geometry fix** — for ds16/ds23 (ellipse ring + dense central blob) the
   k=2 labels are replaced by an ellipse-aware inside/outside boundary fitted in
   PCA-2 space.

**Transductive pooling (the v16 change):** the public train is embedded in
`solution.py` as a per-column uint16 quantization (hex string, decoded with
numpy only) and pooled with the hidden test before the pipeline runs, so every fit is made on
~1130–1325 pooled rows instead of the ~244 hidden-test rows alone. Only the
labels of
the hidden-test rows are returned. All random seeds are fixed (42), so
`cluster(X, dataset_id)` is deterministic; unknown `dataset_id`s fall back to
the same pipeline without pooling.
"""

CONCL = """## Выводы

- The solution is a per-dataset ensemble (KMeans / GMM / Spectral with
  density-based confirmers) whose normalization, PCA dimensionality and k are
  adapted to each dataset; ds16/ds23 use an ellipse-aware boundary instead of
  k=2 clusterers.
- The public train is embedded in `solution.py` (uint16/hex) and pooled with the
  hidden test at inference time, which stabilizes clustering at test size:
  label-stability across random test-size draws improves from mean ARI 0.900 to
  0.973 versus running on the draw alone (worst single draw: ds04, fuzzy data).
- This notebook describes exactly what `solution.py` executes; all seeds are
  fixed, so results are reproducible.
"""

nb = nbf.v4.new_notebook()
nb['cells'] = [nbf.v4.new_markdown_cell(s) for s in (TITLE, DATA, SOLUTION, CONCL)]
nbf.validate(nb)
json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('notebook written:', NB, len(nb['cells']), 'cells')

r = subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',
                    '--ExecutePreprocessor.timeout=600', os.path.abspath(NB)],
                   capture_output=True, text=True)
print('nbconvert rc=', r.returncode)
if r.returncode != 0:
    print(r.stdout[-2000:]); print(r.stderr[-3000:])
else:
    with zipfile.ZipFile(ZP, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write('my_solution16/solution.py', 'solution.py')
        z.write(NB, 'notebook.ipynb')
    print('zip:', ZP, os.path.getsize(ZP), 'bytes,', zipfile.ZipFile(ZP).namelist())
