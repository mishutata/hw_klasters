# -*- coding: utf-8 -*-
"""Build my_solution6/notebook.ipynb (minimal shell) + submission.zip for v6."""
import json, zipfile

nb = json.load(open('my_solution11/notebook.ipynb', encoding='utf-8'))
sol = open('my_solution6/solution.py', encoding='utf-8').read()
assert sol.splitlines()[0].endswith('v6. Identical to v5 except:'), 'wrong solution version'

MD0 = """# Clustering Challenge — решение (v6)

Задача: 24 датасета (ds00–ds23), метрика Mean ARI. База решения — v5 (лучшая
отправка, ARI 0.4846). Изменение v6 — точечные фиксации k и метода по визуальной
проверке на публичном трейне (dev_v6/, dev_v5/deep_ds16_18_23.png):
- ds18: k=4, метод SpectralClustering;
- ds16/ds23: «кольцо + плотный центр», кольцо состоит из 3 дуг → k=4 (три дуги +
  центр), метод SpectralClustering.
Всё остальное идентично v5 побитно: чистка NaN → нормализация по датасету
(minmax/log-sym/robust, выбрана на публичном трейне) → PCA (95% дисперсии,
потолок 8) → k = prior ±1 окно силуэта (для ds04/ds09/ds19/ds20 зафиксированы) →
ансамбль KMeans/GMM/Spectral с подтверждающими DBSCAN/HDBSCAN, победитель по
silhouette + 0.3·consensus NMI. Детерминировано (random_state=42).

## Данные участника

ФИО: Борцов Михаил Алексеевич
Email: mialbortsov@edu.hse.ru
ВУЗ: ВШЭ
"""

MD3 = """## Выводы

Конвейер (как в v5): чистка NaN → нормализация по датасету (minmax/log-sym/robust,
выбрана на публичном трейне) → PCA до 95% дисперсии (потолок 8) → k = prior ±1 окно
силуэта (для ds04/ds09/ds19/ds20 зафиксированы проверкой на публичном трейне) →
ансамбль KMeans/GMM/Spectral с подтверждающими DBSCAN/HDBSCAN, победитель по
silhouette + 0.3·consensus NMI.

Изменение v6 — три точечные фиксации, проверенные картинками на публичном трейне:
ds18 (k=4, Spectral) и ds16/ds23 (кольцо из 3 дуг + центр → k=4, Spectral). Остальные
21 датасет идентичны v5 (0.4846) побитно. `notebook.ipynb` запускает те же функции,
что и `solution.py`; всё детерминировано (random_state=42).
"""

nb['cells'][0]['source'] = MD0.splitlines(keepends=True)
nb['cells'][2]['source'] = sol.splitlines(keepends=True)
nb['cells'][3]['source'] = MD3.splitlines(keepends=True)
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None
json.dump(nb, open('my_solution6/notebook.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

with zipfile.ZipFile('my_solution6/submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('my_solution6/solution.py', 'solution.py')
    z.write('my_solution6/notebook.ipynb', 'notebook.ipynb')
print('v6 package built:', zipfile.ZipFile('my_solution6/submission.zip').namelist())
