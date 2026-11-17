"""Plot the EXACT parabolas that solution.py fits for ds19.

Runs the real code path (sol.cluster), then reads sol.DS19_LAST_FIT to draw
the two fitted curves — the very ones used for point assignment. A self-check
asserts that 'nearest plotted curve' reproduces the labels point-for-point,
so the picture is provably what the code computes.

Usage:  python plot_ds19.py        (writes ds19_parabolas.png next to itself)
"""
import importlib.util
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # remove for interactive use
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sol", os.path.join(HERE, "solution.py"))
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

# --- run the actual solution on the train ds19 sample ---
df = pd.read_csv(os.path.join(HERE, '..', 'public_data', 'train.csv'))
d19 = df[df.dataset_id == 'ds19']
Xraw = d19[[c for c in d19.columns if c.startswith('x')]].values.astype(float)
lab = sol.cluster(Xraw, 'ds19')

fit = sol.DS19_LAST_FIT
assert fit and np.array_equal(lab, fit['labels']), "cluster() did not take the ds19 parabola path"
Q, R = fit['Q'], fit['R']
curves = [cv @ R for cv in fit['curves_rot']]   # rotated back to the (x6s, x7s) frame

# self-check: nearest plotted curve must reproduce the labels point-for-point
Dm = np.stack([np.linalg.norm(Q[:, None, :] - cv[None, :, :], axis=2).min(axis=1)
               for cv in curves], axis=1)
assert np.array_equal(Dm.argmin(1), lab), "plotted curves do NOT reproduce the labels!"

# --- draw ---
fig, ax = plt.subplots(figsize=(8, 8))
for g, col in ((0, '#f4c430'), (1, '#e63946')):
    ax.scatter(Q[lab == g, 0], Q[lab == g, 1], s=9, c=col, edgecolors='none',
               label=f'cluster {g} (n={int((lab == g).sum())})')
for cv, col in zip(curves, ('#f4c430', '#e63946')):
    ax.plot(cv[:, 0], cv[:, 1], color=col, lw=2.8, alpha=.95)
ax.set_xlabel('x6 (std)')
ax.set_ylabel('x7 (std)')
ax.set_title('ds19: exact parabolas fitted by solution.py\n'
             f'+{sol.DS19_ROT_DEG:g} deg rotated frame, two-parabola EM, point -> nearest curve')
ax.legend()
plt.tight_layout()
out = os.path.join(HERE, 'ds19_parabolas.png')
plt.savefig(out, dpi=130)
print('saved', out)
print('cluster sizes:', np.bincount(lab))
