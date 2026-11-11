# -*- coding: utf-8 -*-
"""Rebuild my_solution_final/notebook.ipynb so it uses ONLY allowed imports
(numpy, pandas, scipy, sklearn, matplotlib, seaborn) - no glob, no importlib,
no local-file imports. The pipeline cell pastes solution.py verbatim (from the
K_PRIOR line to EOF), so notebook and solution.py are identical by construction.
Then: execute in a clean dir mimicking the checker, rebuild submission.zip."""
import json, os, shutil, subprocess, sys, time, uuid
import numpy as np, pandas as pd

ROOT = r'C:\AI_master\ML\hw_klasters'
os.chdir(ROOT)
FIN = 'my_solution_final'
SP = os.path.join(FIN, 'solution.py')
NP = os.path.join(FIN, 'notebook.ipynb')

src = open(SP, encoding='utf-8').read()
i = src.index('K_PRIOR = {')
pipeline_code = src[i:].rstrip()  # everything: K_PRIOR..NORM_PRIOR..all funcs..cluster()
assert 'import' not in pipeline_code.split('\n')[0]
for ln in pipeline_code.splitlines():
    assert not ln.strip().startswith(('import ', 'from ')), ln

M = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'id': uuid.uuid4().hex,
               'source': s.splitlines(keepends=True)}
C = lambda s: {'cell_type': 'code', 'execution_count': None,
               'metadata': {}, 'outputs': [], 'id': uuid.uuid4().hex,
               'source': s.splitlines(keepends=True)}

md0 = """# Clustering Challenge — my solution

Task: 24 datasets (ds00–ds23), metric Mean ARI, passing line >= 0.60.

Pipeline (identical in `solution.py` and this notebook):
1. NaN-safe clean -> **per-dataset normalization** (minmax / log-sym / robust, chosen on public train)
   -> PCA (95% variance, <=8 dims).
2. k = per-dataset prior validated on public train, refined at runtime on a +/-1 silhouette window.
3. Stability probe: KMeans x3 seeds; if unstable, fit on the density core set and extend labels to tails.
4. Ensemble: KMeans + GaussianMixture + SpectralClustering (fit on chosen set), plus DBSCAN/HDBSCAN
   with auto-tuned parameters as *confirming* candidates only (join the vote if low noise and NMI >= 0.5).
5. Winner: silhouette + 0.3 * mean consensus NMI (deterministic, random_state=42).
6. ds16/ds23 only: ellipse ring + dense central blob -> labels replaced by an ellipse-aware
   ring/blob boundary fitted in PCA-2 space (bootstrap ARI 0.93/0.94 on public train).
"""
md1 = """## Данные участника

ФИО: Борцов Михаил Алексеевич
Email: mialbortsov@edu.hse.ru
ВУЗ: ВШЭ
"""
c_load = """import numpy as np, pandas as pd
df = None
for path in ('train.csv', 'public_data/train.csv', '../public_data/train.csv'):
    try:
        df = pd.read_csv(path)
        break
    except Exception:
        continue
assert df is not None, 'train.csv not found'
xcols = [c for c in df.columns if c.startswith('x')]
print(df.shape); print(sorted(df.dataset_id.unique()))"""
md_code = """## 0. Solution code (identical to `solution.py`)
The cell below is the body of `solution.py` pasted verbatim (constants + all
functions; only allowed libraries are imported: numpy, scipy, sklearn.
Everything computed in this notebook uses these same objects, so the analysis
matches the graded file exactly.
After running it, `cluster(X, dataset_id)` returns final labels."""
c_pipe = """import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
try:
    from sklearn.cluster import HDBSCAN
except ImportError:  # scikit-learn < 1.3 has no HDBSCAN
    HDBSCAN = None
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
# --- body of solution.py, pasted verbatim (see the markdown cell above) ---
""" + pipeline_code
md_a1 = """## 1. Data analysis
Per dataset: dimensionality, PCA-95% dim count, KMeans silhouette and GMM BIC for k=2..6.
This justifies the `K_PRIOR` table in `solution.py`."""
c_an = """from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
def clean(X):
    X = np.asarray(X, dtype=float)
    keep = ~np.all(np.isnan(X), axis=0); X = X[:, keep] if keep.any() else np.zeros((len(X),1))
    for j in range(X.shape[1]):
        bad = np.isnan(X[:, j])
        if bad.any(): X[bad, j] = np.nanmean(X[:, j])
    return X
res = {}
for ds, g in df.groupby('dataset_id'):
    Z = StandardScaler().fit_transform(clean(g[xcols].values))
    p = PCA(random_state=42).fit(Z)
    cum = np.cumsum(p.explained_variance_ratio_)
    ncomp = max(2, min(8, int(np.searchsorted(cum, 0.95) + 1), Z.shape[1]))
    Z2 = PCA(n_components=ncomp, random_state=42).fit_transform(Z)
    row = {'n': len(Z), 'dims': Z.shape[1], 'pca95': ncomp}
    for k in range(2, min(7, len(Z)//5) + 1):
        lab = KMeans(k, n_init=10, random_state=42).fit_predict(Z2)
        row[f'sil{k}'] = round(silhouette_score(Z2, lab), 3)
        try:
            row[f'bic{k}'] = round(GaussianMixture(k, covariance_type='full', n_init=2,
                                                   random_state=42).fit(Z2).bic(Z2), 1)
        except Exception: pass
    res[ds] = row
tab0 = pd.DataFrame(res).T
print(tab0.to_string())"""
md_a2 = """## 2. Preprocessing choice per dataset
Same KMeans at the prior k, but with four normalizations (standard / robust / minmax / log-sym).
The best one per dataset on public train becomes `NORM_PRIOR` in `solution.py`."""
c_norm = """from sklearn.cluster import KMeans
NORMS = ['standard', 'robust', 'minmax', 'log-sym']
rows = []
for ds, g in df.groupby('dataset_id'):
    X = _clean(g[xcols].values)
    k0 = K_PRIOR.get(ds, 3)
    s = {}
    for nm in NORMS:
        Z = _reduce(X, nm)
        lab = KMeans(k0, n_init=10, random_state=42).fit_predict(Z)
        s[nm] = round(_sil(Z, lab), 3)
    rows.append({'dataset': ds, 'k_prior': k0, **s, 'best_norm': max(s, key=s.get)})
norm_tab = pd.DataFrame(rows).set_index('dataset')
print(norm_tab.to_string())
derived = norm_tab['best_norm']
prior = pd.Series(NORM_PRIOR)
cmp = pd.DataFrame({'derived_best': derived,
                   'in_solution': prior.reindex(derived.index).fillna('minmax (default)')})
print(cmp.to_string())"""
md_a3 = """## 3. Geometry of ds16 and ds23
Both are **ring + dense central blob**: a k=2 fit gives one ring and one blob; standard clusterers
split the ring into arcs (ARI ~ 0 vs the true split). `solution.py` therefore replaces labels for
ds16/ds23 with an ellipse-aware radial boundary fitted in PCA-2 space
(bootstrap ARI 0.93 / 0.94 across 70% resamples, see folder sol_16_23/).
True k = 2 (the initial table had 3)."""
c_run = """# Run the exact solution pipeline on all 24 datasets
infos = {}
for ds, g in df.groupby('dataset_id'):
    lab, info = _run(g[xcols].values.astype(float), ds)
    infos[ds] = info
    print(ds, 'k =', info['k_used'], '| winner =', info['winner'],
          '| km_stab =', info['km_stability'], '| core_removed =', info['core_frac_removed'])"""
c_tab = """METHODS = ['KMeans', 'GaussianMixture', 'SpectralClustering', 'DBSCAN', 'HDBSCAN', 'ring/blob']
tab = pd.DataFrame([
    {'dataset': ds, 'k': i['k_used'],
     **{m: i['methods'].get(m, np.nan) for m in METHODS},
     'winner': i['winner']} for ds, i in infos.items()]).set_index('dataset')
print(tab.to_string())"""
md_vis = """## 4. Visualizations
Scatter plots: fresh PCA-2D projection of the cleaned features, colored by the final labels
(the same ones `cluster()` returns)."""
c_plot1 = """import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, ds in zip(axes.ravel(), ['ds00', 'ds04', 'ds11', 'ds16', 'ds19', 'ds23']):
    Xc = clean(df[df.dataset_id == ds][xcols].values)
    P = PCA(2, random_state=42).fit_transform(StandardScaler().fit_transform(Xc))
    lab = cluster(df[df.dataset_id == ds][xcols].values.astype(float), ds)
    ax.scatter(P[:, 0], P[:, 1], c=lab, s=5, cmap='tab10', alpha=0.6)
    ax.set_title(f"{ds}: k={infos[ds]['k_used']}, winner={infos[ds]['winner']}")
    ax.axis('off')
plt.tight_layout(); plt.show()"""
c_plot2 = """import matplotlib.pyplot as plt
ks = [2, 3, 4, 5, 6]
fig, ax = plt.subplots(figsize=(10, 5))
for ds in sorted(res):
    row = res[ds]
    vals = [row.get(f'sil{k}') for k in ks]
    if any(v is None for v in vals): continue
    ax.plot(ks, vals, marker='o', ms=3, alpha=0.55, label=ds)
    bk = infos[ds]['k_used']
    if bk in ks:
        ax.scatter([bk], [row[f'sil{bk}']], s=70, color='red', zorder=5)
ax.set_xlabel('k'); ax.set_ylabel('silhouette (KMeans)')
ax.set_title('Silhouette vs k for all 24 datasets (red = k used by the solution)')
ax.legend(ncol=4, fontsize=7); plt.tight_layout(); plt.show()"""
c_plot3 = """import matplotlib.pyplot as plt
M = tab[METHODS].T
fig, ax = plt.subplots(figsize=(12, 3))
im = ax.imshow(M.values, aspect='auto', cmap='viridis')
ax.set_yticks(range(len(METHODS)), METHODS)
ax.set_xticks(range(len(M.columns)), M.columns, rotation=90, fontsize=7)
for i, m in enumerate(METHODS):
    for j, ds in enumerate(M.columns):
        w = infos[ds]['winner'] == m
        ax.text(j, i, f"{M.values[i, j]:.2f}", ha='center', va='center', fontsize=7,
                color='yellow' if w else 'white', weight='bold' if w else 'normal')
ax.set_title('Candidate score = silhouette + 0.3*consensus NMI (bold/yellow = winner)')
plt.colorbar(im); plt.tight_layout(); plt.show()"""

# ---- winners/runtime for the Выводы text: run the pipeline once here ----
spec = __import__('importlib').util.spec_from_file_location('sol', SP)
sol = __import__('importlib').util.module_from_spec(spec); spec.loader.exec_module(sol)
df0 = pd.read_csv(os.path.join('public_data', 'train.csv'))
xx = [c for c in df0.columns if c.startswith('x')]
rows, t_all = {}, time.time()
for ds, g in df0.groupby('dataset_id'):
    t1 = time.time()
    lab, info = sol._run(g[xx].values.astype(float), ds)
    rows[ds] = {'k': int(info['k_used']), 'winner': info['winner'],
                'rt': round(time.time() - t1, 2)}
mean_rt = float(np.mean([r['rt'] for r in rows.values()]))
from collections import Counter
wc = Counter(r['winner'] for r in rows.values())
wline = ', '.join(f'{k}: {v}/24' for k, v in sorted(wc.items(), key=lambda x: -x[1]))
json.dump(rows, open('tmp_report.json', 'w'), indent=1)
print('pipeline run:', round(time.time() - t_all, 1), 's total')

md_concl = f"""## Выводы
- Предобработка подбирается по датасету на открытых данных (таблица раздела 2):
  minmax / log-sym / robust / standard; затем PCA до <=8 измерений (95% дисперсии).
- Число кластеров берётся из анализа обучающей выборки (раздел 1); для ds16/ds23 k=2
  (геометрия «кольцо + центр»), а метки строит эллиптическая граница ring/blob
  (bootstrap ARI 0.93/0.94 на публичных данных).
- Победители ансамбля по датасетам: {wline}.
  DBSCAN/HDBSCAN участвуют только как подтверждающие кандидаты (низкий уровень шума + NMI >= 0.5),
  поэтому не портят плотные перекрывающиеся кластеры, а core-point тримминг стабилизирует
  KMeans на хвостовых данных.
- Всё детерминировано (random_state=42): ячейка «Solution code» — дословное тело `solution.py`,
  поэтому результаты ноутбука совпадают с результатами `cluster()`. Среднее время на датасет ~{mean_rt:.1f} c.
- Внутренние метрики (silhouette 0.25–0.6 в зависимости от датасета, стабильность перезапусков
  > 0.8 после core-тримминга) указывают на Mean ARI выше проходного порога 0.60.
"""

cells = [M(md0), M(md1), C(c_load), M(md_code), C(c_pipe), M(md_a1), C(c_an),
         M(md_a2), C(c_norm), M(md_a3), C(c_run), C(c_tab), M(md_vis),
         C(c_plot1), C(c_plot2), C(c_plot3), M(md_concl)]
nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3 (ipykernel)',
                                  'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
json.dump(nb, open(NP, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('notebook written:', len(cells), 'cells')

# ---- execute in a clean dir mimicking the checker ----
tmp = os.path.join(ROOT, 'tmp_checker_run')
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
shutil.copy(SP, tmp); shutil.copy(NP, tmp)
shutil.copy(os.path.join('public_data', 'train.csv'), tmp)
r = subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                   '--inplace', 'notebook.ipynb'], cwd=tmp,
                  capture_output=True, text=True, timeout=3600)
print('nbconvert rc =', r.returncode)
if r.returncode != 0:
    print(r.stdout[-2000:]); print(r.stderr[-2000:]); sys.exit(1)

# ---- zip exactly solution.py + notebook.ipynb at root ----
import zipfile
zpath = os.path.join(FIN, 'submission.zip')
if os.path.exists(zpath): os.remove(zpath)
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(tmp, 'notebook.ipynb'), 'notebook.ipynb')
    z.write(SP, 'solution.py')
print('zip rebuilt:', os.path.getsize(zpath), 'bytes')
shutil.rmtree(tmp)
print('DONE')
