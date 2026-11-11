"""Pair plots for every dataset: all C(p,2) feature pairs (x_i vs x_j),
colored by the current pipeline's cluster labels (my_solver/solution.py,
full public train).

Outputs: pair_plots/dsXX.png  (one figure per dataset)
"""
import importlib.util
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOL = 'my_solver/solution.py'
OUT = 'pair_plots'
os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location('sol', SOL)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

df = pd.read_csv('public_data/train.csv')
ids = [c for c in df.columns if c.startswith('x')]
MAXPTS = None         # None = all points
NCOLS = 7             # subplots per row

rng = np.random.RandomState(42)
summary = []
for ds, g in df.groupby('dataset_id'):
    Xraw = g[ids].to_numpy(float)
    keep = ~np.all(np.isnan(Xraw), axis=0)
    Xc = sol._clean(Xraw)          # same cleaning the pipeline uses
    p = Xc.shape[1]
    lab = np.unique(np.asarray(sol.cluster(Xraw, ds)), return_inverse=True)[1]
    k = int(lab.max() + 1)

    if MAXPTS is not None and len(Xc) > MAXPTS:
        idx = rng.choice(len(Xc), size=MAXPTS, replace=False)
        Xs, ls = Xc[idx], lab[idx]
    else:
        Xs, ls = Xc, lab

    pairs = list(combinations(range(p), 2))   # (i, j): x_i vs x_j
    nplots = len(pairs)
    nrows = int(np.ceil(nplots / NCOLS))
    fig, axes = plt.subplots(
        nrows, NCOLS,
        figsize=(NCOLS * 2.5, nrows * 2.5 + 1.4),
        squeeze=False,
    )
    try:
        cmap = plt.colormaps['tab20']
    except AttributeError:  # older matplotlib
        cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i / 20) for i in range(20)]   # discrete tab20 entries
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_linewidth(0.4)

    for n, (i, j) in enumerate(pairs):
        ax = axes[n // NCOLS, n % NCOLS]
        m = np.isfinite(Xs[:, i]) & np.isfinite(Xs[:, j])
        for c in range(k):
            mm = m & (ls == c)
            if mm.any():
                ax.scatter(Xs[mm, i], Xs[mm, j], s=5, alpha=0.6,
                           linewidths=0, color=colors[c % 20], rasterized=True)
        row, col = n // NCOLS, n % NCOLS
        if row == nrows - 1:
            ax.set_xlabel(f'x{i+1}', fontsize=7)
        if col == 0:
            ax.set_ylabel(f'x{j+1}', fontsize=7, rotation=0, labelpad=2)
    # hide leftover empty subplots
    for n in range(nplots, nrows * NCOLS):
        axes[n // NCOLS, n % NCOLS].axis('off')

    sizes = ', '.join(f'C{c}:{int((lab == c).sum())}' for c in range(k))
    fig.suptitle(f'{ds}: p={p} features, k={k} clusters [{sizes}] — all {nplots} feature pairs'
                 f'  (labels: my_solver/solution.py on full train)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(OUT, f'{ds}.png')
    fig.savefig(out, dpi=100)
    plt.close(fig)
    summary.append((ds, p, k, nplots, out))
    print('saved', out)

print('\nds   p   k  npairs')
for ds, p, k, nplots, out in summary:
    print(f'{ds}  {p:2d}  {k}  {nplots}')
