# -*- coding: utf-8 -*-
"""Build my_solution12/notebook.ipynb from the v11 shell:
participant data (cell 0) + verbatim solution.py body (cell 2) + Выводы (cell 3).
Usage: python build_nb12.py <tau>   (tau = chosen MEDIAN_TAU, e.g. 0.2)
"""
import json
import sys

tau = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
nb = json.load(open('my_solution11/notebook.ipynb', encoding='utf-8'))
sol = open('my_solution12/solution.py', encoding='utf-8').read()
assert 'MEDIAN_TAU' in sol and 'v12' in sol, 'wrong solution version'

MD0 = f"""# Clustering Challenge — решение (v12)

Задача: 24 датасета (ds00–ds23), метрика Mean ARI. База решения — v6 (лучшая отправка,
ARI 0.4846); v12 отличается только правилом выбора числа компонент PCA: вместо жёсткого
потолка в 8 измерений и правила «95% дисперсии» (средняя/дисперсия — чувствительны к
выбросам) компоненты сохраняются, пока **медианная** по точкам ошибка восстановления в
Z-пространстве превышает {tau} × **медианное** попарное расстояние между точками;
потолка нет. Мотивация: при потолке 8 худшие датасеты (ds08/ds13/ds14) сохраняли лишь
75–83% дисперсии, а у них же почти нулевая разность скоростей победителя ансамбля и
высокая нестабильность выбора на тестовом размере (dev_v11/diag_v12.py).

## Данные участника

ФИО: Борцов Михаил Алексеевич
Email: mialbortsov@edu.hse.ru
ВУЗ: ВШЭ
"""

MD3 = f"""## Выводы

Конвейер (как в v6): чистка NaN → нормализация по датасету (minmax/log-sym/robust, выбрана на
публичном трейне) → PCA → выбор k (prior + окно ±1 по силуэту; для ds04/ds09/ds16/ds18/ds19/ds20/ds23
k и метод зафиксированы по проверке на публичном трейне) → ансамбль KMeans/GMM/Spectral с
подтверждением DBSCAN/HDBSCAN, победитель по silhouette + 0.3·consensus NMI.

Изменение v12 — выбор размерности PCA без жёсткого потолка: компоненты сохраняются, пока
медианная ошибка восстановления не опустится ниже {tau} × медианного попарного расстояния в
Z-пространстве (медианы вместо средних — устойчивость к выбросам). Это снимает потерю дисперсии
на ds08/ds13/ds14 (75–83% при старом потолке 8), не меняя правила для остальных датасетов сильнее,
чем требует их собственная геометрия. Всё детерминировано (random_state=42): `notebook.ipynb`
запускает те же функции, что и `solution.py`.
"""

nb['cells'][0]['source'] = MD0.splitlines(keepends=True)
nb['cells'][2]['source'] = sol.splitlines(keepends=True)
nb['cells'][3]['source'] = MD3.splitlines(keepends=True)
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None
json.dump(nb, open('my_solution12/notebook.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('notebook written, tau =', tau)
