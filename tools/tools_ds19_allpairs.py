"""ds19 — ALL 28 axis combinations (C(8,2)) in one grid.

Same preprocessing as the other ds19 views: _clean (drop all-NaN cols,
mean-impute), StandardScaler. Points colored by the CANONICAL para2
membership computed in (x1s, x5s) — so each panel shows where the two
arcs land when projected onto that coordinate pair.
Output: dev/v10/ds19_allpairs.png  (7 cols x 4 rows)
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('public_data/train.csv')
d19 = df[df.dataset_id == 'ds19']
Xraw = d19[[c for c in d19.columns if c.startswith('x')]].values.astype(float)
cols_all = [c for c in d19.columns if c.startswith('x')]
mask = ~np.all(np.isnan(Xraw), axis=0)
cols = [c for c, m in zip(cols_all, mask) if m]
X = Xraw[:, mask].copy()
ni = np.where(np.isnan(X))
if len(ni[0]):
    X[ni] = np.take(np.nanmean(X, axis=0), ni[1])
Xs = StandardScaler().fit_transform(X)
p = Xs.shape[1]
print(f"{Xs.shape[0]} points, {p} features: {cols}")

# canonical membership from (x1s, x5s) — same EM as the probe
i1, i5 = cols.index('x1'), cols.index('x5')
Qc = Xs[:, [i1, i5]]
def em_two_quadratics(Q):
    best = None
    for s in (0, 1, 2, 3, 4):
        lab = KMeans(2, n_init=5, random_state=s).fit_predict(Q)
        small = False
        for _ in range(60):
            cs = []
            for i in (0, 1):
                m = lab == i
                if m.sum() < 8:
                    small = True; break
                A = np.vstack([Q[m, 0]**2, Q[m, 0], np.ones(m.sum())]).T
                c, *_ = np.linalg.lstsq(A, Q[m, 1], rcond=None)
                cs.append(c)
            if small: break
            r = np.stack([np.abs(Q[:, 1] - (cs[i][0]*Q[:, 0]**2 + cs[i][1]*Q[:, 0] + cs[i][2])) for i in (0, 1)])
            nl = r.argmin(0)
            if (nl == lab).all(): break
            lab = nl
        if small: continue
        rms = 0.0
        for i in (0, 1):
            m = lab == i
            A = np.vstack([Q[m, 0]**2, Q[m, 0], np.ones(m.sum())]).T
            c, *_ = np.linalg.lstsq(A, Q[m, 1], rcond=None)
            rms += float(np.sqrt(np.mean((Q[m, 1] - A@c)**2)))
        bal = min(float((lab == 0).mean()), float((lab == 1).mean()))
        if best is None or (rms, -bal) < best[0]:
            best = ((rms, -bal), lab.copy())
    return best[1]
can_lab = em_two_quadratics(Qc)
n0, n1 = int((can_lab == 0).sum()), int((can_lab == 1).sum())

C0, C1 = '#f4c430', '#6a3d9a'
pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
ncols = 7
nrows = (len(pairs) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows))
fig.suptitle(f'ds19 — all {len(pairs)} axis combinations (standardized), colored by canonical para2 membership from (x1,x5)   [arc0={n0}, arc1={n1}]', fontsize=14)
for ax, (i, j) in zip(axes.ravel(), pairs):
    Q = Xs[:, [i, j]]
    ax.scatter(Q[can_lab == 0, 0], Q[can_lab == 0, 1], s=9, c=C0, alpha=0.75, edgecolors='none')
    ax.scatter(Q[can_lab == 1, 0], Q[can_lab == 1, 1], s=9, c=C1, alpha=0.75, edgecolors='none')
    ax.set_title(f'{cols[i]} vs {cols[j]}', fontsize=11)
    ax.set_xlabel(cols[i]); ax.set_ylabel(cols[j])
for ax in axes.ravel()[len(pairs):]:
    ax.axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('dev/v10/ds19_allpairs.png', dpi=150)
print("saved dev/v10/ds19_allpairs.png")
