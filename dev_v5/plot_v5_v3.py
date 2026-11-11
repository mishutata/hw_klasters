# -*- coding: utf-8 -*-
"""Side-by-side v3 vs v5 final labels, fresh PCA-2 per dataset."""
import importlib.util
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

sol3 = load('analysis2/v3_solution.py', 'sol3')
sol5 = load('my_solution5/solution.py', 'sol5')
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
ds_list = sorted(df.dataset_id.unique())

fig, axes = plt.subplots(24, 2, figsize=(9, 10 * 12))
for r, ds in enumerate(ds_list):
    X = df[df.dataset_id == ds][xcols].values.astype(float)
    P = PCA(2, random_state=42).fit_transform(StandardScaler().fit_transform(sol5._clean(X)))
    for c, (sol, tag) in enumerate(((sol3, 'v3'), (sol5, 'v5'))):
        lab, info = sol._run(X, ds)
        ax = axes[r, c]
        ax.scatter(P[:, 0], P[:, 1], c=lab, s=4, cmap='tab10', alpha=0.6)
        ax.set_title(f'{ds} {tag}: k={info["k_used"]}, {info["winner"]}', fontsize=8)
        ax.axis('off')
plt.tight_layout()
plt.savefig('v5_vs_v3.png', dpi=100)
print('saved v5_vs_v3.png')
