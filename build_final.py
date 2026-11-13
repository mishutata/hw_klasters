# -*- coding: utf-8 -*-
"""Build my_solution_final/notebook.ipynb + submission.zip for v6 (best, ARI 0.4846)."""
import json, zipfile

nb = json.load(open('my_solution11/notebook.ipynb', encoding='utf-8'))
sol = open('my_solution6/solution.py', encoding='utf-8').read()
assert sol.splitlines()[0].startswith('"""Clustering solution FINAL — v6'), 'wrong version'

MD0 = """# Clustering Challenge — финальное решение (v6, Mean ARI 0.4846)

Задача: 24 датасета (ds00–ds23), метрика Mean ARI. Это лучшая из всех отправленных версий
(0.4846); поздние эксперименты v12/v13 (правило выбора размерности PCA на медианной ошибке
восстановления, глобальное и точечное) дали 0.4655 и 0.4662 — оба хуже, поэтому финальным
осталось правило v6: PCA до 95% дисперсии с потолком 8 компонент.

## Данные участника

ФИО: Борцов Михаил Алексеевич
Email: mialbortsov@edu.hse.ru
ВУЗ: ВШЭ
"""

MD3 = """## Выводы

Конвейер (v6): чистка NaN → нормализация по датасету (minmax/log-sym/robust, выбрана на
публичном трейне) → PCA (95% дисперсии, потолок 8 компонент) → выбор k (prior + окно ±1 по
силуэту; для ds04/ds09/ds16/ds18/ds19/ds20/ds23 k и метод зафиксированы по проверке на
публичном трейне) → ансамбль KMeans/GMM/Spectral с подтверждением DBSCAN/HDBSCAN, победитель
по silhouette + 0.3·consensus NMI.

Почему остановились на v6: единственная незакрытая гипотеза — что потолок PCA в 8 измерений
вредит ds08/ds09/ds13/ds14 (там d95 = 12/9/11/13, при потолке теряется до ~25% дисперсии).
Проверено двумя отправками: глобальное медианное правило без потолка (v12) — 0.4655, и то же
правило только для этих четырёх датасетов (v13, остальные 20 побитно совпадают с v6) — 0.4662.
Оба хуже 0.4846, причём почти на одинаково: потеря целиком объясняется этими четырьмя
датасетами, т.е. дополнительные зашумлённые компоненты ухудшают кластеризацию, а не улучшают.
Правило v6 оставлено без изменений. Всё детерминировано (random_state=42): `notebook.ipynb`
запускает те же функции, что и `solution.py`.
"""

nb['cells'][0]['source'] = MD0.splitlines(keepends=True)
nb['cells'][2]['source'] = sol.splitlines(keepends=True)
nb['cells'][3]['source'] = MD3.splitlines(keepends=True)
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None
json.dump(nb, open('my_solution_final/notebook.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

import shutil
shutil.copyfile('my_solution6/solution.py', 'my_solution_final/solution.py')
with zipfile.ZipFile('my_solution_final/submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('my_solution_final/solution.py', arcname='solution.py')
    z.write('my_solution_final/notebook.ipynb', arcname='notebook.ipynb')

z = zipfile.ZipFile('my_solution_final/submission.zip')
print('final zip:', z.namelist())
chk = json.loads(z.read('notebook.ipynb'))
print('cell0 ok:', all(s in ''.join(chk['cells'][0]['source']) for s in ('Борцов Михаил Алексеевич','mialbortsov@edu.hse.ru','ВШЭ')))
print('sol header:', z.read('solution.py').decode().splitlines()[0])
