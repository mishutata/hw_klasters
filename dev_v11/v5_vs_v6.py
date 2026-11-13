# -*- coding: utf-8 -*-
"""Verify v6 differs from v5 ONLY on ds16/ds18/ds23 (full train)."""
import importlib.util
import numpy as np
import pandas as pd

def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

v5m = load('v5m', 'my_solution5/solution.py')
v6m = load('v6m', 'my_solution6/solution.py')
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
same, diff = [], []
for ds, g in df.groupby('dataset_id'):
    X = g[xcols].values.astype(float)
    a = v5m.cluster(X, ds)
    b = v6m.cluster(X, ds)
    (same if np.array_equal(a, b) else diff).append(ds)
print('identical to v5:', len(same), same)
print('differ from v5 :', len(diff), diff)
