"""Plot the EXACT parabolas that probes/ds19friend/solution.py uses for ds19.

Runs the real code path (sol._assign_ds19_parabolas), then draws the two
hardcoded friend-parabolas with a self-check that recomputing the same
distance matrix reproduces the labels point-for-point.

Usage:  python plot_ds19.py   (writes ds19_friend_parabolas.png next to itself)
"""
import importlib.util
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sol", os.path.join(HERE, "solution.py"))
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

df = pd.read_csv(os.path.join(HERE, '..', '..', 'public_data', 'train.csv'))
d19 = df[df.dataset_id == 'ds19']
Xraw = d19[[c for c in d19.columns if c.startswith('x')]].values.astype(float)

lab = sol._assign_ds19_parabolas(Xraw)

# normalized frame the friend's code works in
x = (Xraw[:, 5] - sol.DS19_MEAN_X6) / sol.DS19_SCALE_X6
y = (Xraw[:, 6] - sol.DS19_MEAN_X7) / sol.DS19_SCALE_X7

# self-check: independent recomputation of the exact distance matrix
dists = np.zeros((len(x), 2))
for c, (theta, coef) in enumerate(sol.DS19_PARABOLA_PARAMS):
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    dists[:, c] = np.abs(yr - np.polyval(coef, xr))
assert np.array_equal(dists.argmin(1), lab), "recomputed labels != solution labels!"

# draw each parabola: sample xr over the data's range (with margin), unrotate back
fig, ax = plt.subplots(figsize=(8, 8))
for g, col in ((0, '#f4c430'), (1, '#e63946')):
    ax.scatter(x[lab == g], y[lab == g], s=9, c=col, edgecolors='none',
               label=f'cluster {g} (n={int((lab == g).sum())})')
for theta, coef in sol.DS19_PARABOLA_PARAMS:
    xr_d = x * np.cos(theta) + y * np.sin(theta)
    lo, hi = np.quantile(xr_d, 0.01), np.quantile(xr_d, 0.99)
    m = 0.15
    xs = np.linspace(lo - m, hi + m, 300)
    ys = np.polyval(coef, xs)
    Xp = xs * np.cos(theta) - ys * np.sin(theta)
    Yp = xs * np.sin(theta) + ys * np.cos(theta)
    ax.plot(Xp, Yp, color='#222', lw=2.8, alpha=.95)
ax.set_xlabel('x6 (friend normalization)')
ax.set_ylabel('x7 (friend normalization)')
ax.set_title('ds19: friend\'s two hardcoded parabolas (probes/ds19friend)\n'
             'point -> smaller |y_rot - f(x_rot)|, black = fitted curves')
ax.legend()
plt.tight_layout()
out = os.path.join(HERE, 'ds19_friend_parabolas.png')
plt.savefig(out, dpi=130)
print('saved', out)
print('cluster sizes:', np.bincount(lab))
for i, (theta, coef) in enumerate(sol.DS19_PARABOLA_PARAMS):
    print(f'parabola {i}: theta={np.degrees(theta):.1f} deg, y = {coef[0]:.3f} x^2 + {coef[1]:.3f} x + {coef[2]:.3f}')
