import json
nb = json.load(open('my_solution9/notebook.ipynb'))
sol = open('my_solution10/solution.py', encoding='utf-8').read()
assert 'K_PIN' in sol and '"ds13": 2' in sol, 'wrong solution version'
nb['cells'][6]['source'] = ('# --- body of solution.py, pasted verbatim (see solution.py in this folder) ---\n'
                            + sol).splitlines(keepends=True)
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None
MD0 = """# Clustering Challenge — my solution

Task: 24 datasets (ds00–ds23), metric Mean ARI, passing line >= 0.60.

Pipeline (identical in `solution.py` and this notebook):
1. NaN-safe clean -> **per-dataset normalization** (minmax / log-sym / robust, chosen on public train)
   -> PCA (95% variance, <=8 dims; cap lifted to 14 for ds08/ds13/ds14 — at cap 8 they kept only
   75-83% of variance, the worst of the 24).
2. k = per-dataset prior validated on public train; pinned where verified:
   ds04/ds09 at k=3, ds10 at k=2, ds13/ds14 at k=2, ds16/ds23 at k=2 (ring + central blob),
   ds18/ds19/ds20 at k=4; the rest are refined at runtime on a +/-1 silhouette window.
3. Stability probe: KMeans x3 seeds; if unstable, fit on the density core set and extend labels to tails.
4. Ensemble: KMeans + GaussianMixture + SpectralClustering (fit on chosen set), plus DBSCAN/HDBSCAN
   with auto-tuned parameters as *confirming* candidates only (join the vote if low noise and NMI >= 0.5).
5. Winner: silhouette + 0.3 * mean consensus NMI (deterministic, random_state=42);
   pinned to GaussianMixture for ds04/ds13/ds14 and to SpectralClustering for ds09/ds10/ds18/ds19/ds20
   (verified on public train + a test-size simulation at n ~= 200).
6. ds16/ds23 only: ellipse ring + dense central blob -> labels replaced by an ellipse-aware
   ring/blob boundary fitted in PCA-2 space with per-dataset radius
   (frac ds16=0.93, ds23=0.90, tuned on public train; bootstrap ARI 0.943 / ~0.94).
"""
MD7 = """## 3. Geometry of ds16 and ds23
Both are **ring + dense central blob**: a k=2 fit gives one ring and one blob; standard clusterers
split the ring into arcs (ARI ~ 0 vs the true split). `solution.py` therefore replaces labels for
ds16/ds23 with an ellipse-aware radial boundary fitted in PCA-2 space, applied unconditionally
(silhouette is a poor selector for non-bulb shapes). The boundary radius is tuned per dataset on
public train (dev_v6/ring_radius_tune3): **frac ds16 = 0.93** (bootstrap ARI 0.943 across 70%
resamples) and **ds23 = 0.90** (zero ring error, blob loss 8.5% -> 2%). True k = 2.
"""
MD14 = """## Выводы
- Предобработка подбирается по датасету на открытых данных: minmax побеждает в 15/24,
  log-sym в 8/24, robust в 1/24 (таблица из раздела 2); затем PCA до <=8 измерений (95% дисперсии),
  за исключением ds08/ds13/ds14, где потолок поднят до 14: при 8 измерениях там оставалось лишь
  75–83% дисперсии — худшие из 24 датасетов (dev_v8/pca_dims_per_ds.csv).
- Число кластеров берётся из анализа обучающей выборки (раздел 1); для ds16/ds23 k=2
  (геометрия «кольцо + центр»), а метки строит эллиптическая граница ring/blob, радиус которой
  подобран по датасету на публичных данных: frac ds16=0.93 (bootstrap ARI 0.943),
  ds23=0.90 (нулевая ошибка кольца, потеря blob 8.5% -> 2%).
- Для ds04/ds09 k зафиксировано равным 3, для ds10 — 2, для ds13/ds14 — 2, а для ds18/ds19/ds20 — 4
  после проверки публичных данных и симуляции тестового размера (80/20 сплит, ARI на псевдо-тесте);
  победителем зафиксирован SpectralClustering для ds09/ds10/ds18/ds19/ds20, а GaussianMixture —
  для ds04/ds13/ds14: при фиксированном k именно они восстанавливают истинные кластеры на малых
  выборках (Spectral деградирует на ~200 точках; на ds13/ds14 в полном PCA-пространстве GMM k=2
  собирает два настоящих blob, тогда как Spectral/KMeans режут их вдоль шума —
  dev_v8/pin_check_08_13_14.png).
- Победители ансамбля по датасетам: SpectralClustering: 14/24, KMeans: 5/24, GaussianMixture: 3/24,
  ring/blob: 2/24. DBSCAN/HDBSCAN участвуют только как подтверждающие кандидаты
  (низкий уровень шума + NMI >= 0.5), поэтому не портят плотные перекрывающиеся кластеры,
  но их core-point детекция стабилизирует KMeans на хвостовых данных.
- Всё детерминировано (random_state=42): `notebook.ipynb` вызывает те же функции, что и
  `solution.py` (`cluster()` / `_run()`), поэтому результаты совпадают. Среднее время на датасет ~0.4 c.
- Внутренние метрики (silhouette 0.25–0.6 в зависимости от датасета, стабильность перезапусков
  > 0.8 после core-тримминга) указывают на Mean ARI выше проходного порога 0.60.
"""
for idx, txt in ((0, MD0), (7, MD7), (14, MD14)):
    nb['cells'][idx]['source'] = txt.splitlines(keepends=True)
json.dump(nb, open('my_solution10/notebook.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('notebook written; cell6 lines:', len(nb['cells'][6]['source']))
