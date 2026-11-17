"""ds19 parabola theory — full-story visualization.

Panels:
  A. The truth: two interleaved parabolic arcs in (x1s, x5s), EM membership + fits
  B. What the champion does: spectral k=2 on PCA-4 -> positional cut (wrong)
  C. Fit quality: residuals of each arc vs its quadratic fit (tight bands)
  D. Membership along the curve: fraction para0 in x1 bins (no single flip point)

Uses the EXACT probe procedure (probes/ds19parab/solution.py _parabola2_labels):
StandardScaler -> Q = (x1s, x5s) -> KMeans(2,n_init=5) seeds 0..4 -> quadratic EM,
best (rms_sum, -balance). Champion path: StandardScaler -> PCA(4) -> spectral knn nn=20.

Output: dev/v10/ds19_parabola_story.png
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

df = pd.read_csv('public_data/train.csv')
d19 = df[df.dataset_id == 'ds19']
X = d19[[c for c in d19.columns if c.startswith('x')]].values.astype(float)

def _clean(X):
    """Exact copy of _clean from probes/ds19parab/solution.py."""
    X = np.asarray(X, dtype=float)
    mask = ~np.all(np.isnan(X), axis=0)
    X = X[:, mask]
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = np.take(col_mean, nan_idx[1])
    return X

X = _clean(X)
Xs = StandardScaler().fit_transform(X)
Q = Xs[:, [0, 4]]          # (x1s, x5s) — where the parabolas live

def parabola2_labels(Q):
    """Exact copy of _parabola2_labels from probes/ds19parab/solution.py."""
    best = None
    for s in (0, 1, 2, 3, 4):
        lab = KMeans(2, n_init=5, random_state=s).fit_predict(Q)
        small = False
        for _ in range(60):
            cs = []
            for i in (0, 1):
                m = lab == i
                if m.sum() < 8:
                    small = True
                    break
                A = np.vstack([Q[m, 0]**2, Q[m, 0], np.ones(m.sum())]).T
                c, *_ = np.linalg.lstsq(A, Q[m, 1], rcond=None)
                cs.append(c)
            if small:
                break
            r = np.stack([np.abs(Q[:, 1] - (cs[i][0]*Q[:, 0]**2 + cs[i][1]*Q[:, 0] + cs[i][2])) for i in (0, 1)])
            nl = r.argmin(0)
            if (nl == lab).all():
                break
            lab = nl
        if small:
            continue
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

para_lab = parabola2_labels(Q)

# champion path: PCA(4) -> spectral knn nn=20, k=2 (seed 42, as in cluster())
Z4 = PCA(4, random_state=42).fit_transform(Xs)
champ_lab = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=20,
                               random_state=42).fit_predict(Z4)

# final quadratic fits per arc (for drawing + residuals)
fits, rms = [], []
for i in (0, 1):
    m = para_lab == i
    A = np.vstack([Q[m, 0]**2, Q[m, 0], np.ones(m.sum())]).T
    c, *_ = np.linalg.lstsq(A, Q[m, 1], rcond=None)
    fits.append(c)
    rms.append(float(np.sqrt(np.mean((Q[m, 1] - (c[0]*Q[m,0]**2 + c[1]*Q[m,0] + c[2]))**2))))

# crossing of the two arcs
A2 = np.poly(fits[0][0]-fits[1][0], fits[0][1]-fits[1][1], fits[0][2]-fits[1][2]) if False else None
d2, d1, d0 = fits[0][0]-fits[1][0], fits[0][1]-fits[1][1], fits[0][2]-fits[1][2]
roots = np.roots([d2, d1, d0]) if abs(d2) > 1e-9 else [-d0/d1]
cross = [r.real for r in roots if np.isreal(r) and Q[:, 0].min() <= r.real <= Q[:, 0].max()]

ari = adjusted_rand_score(champ_lab, para_lab)
n0, n1 = int((para_lab == 0).sum()), int((para_lab == 1).sum())
print(f"para0: a={fits[0][0]:+.3f} b={fits[0][1]:+.3f} c={fits[0][2]:+.3f} n={n0} rms={rms[0]:.3f}")
print(f"para1: a={fits[1][0]:+.3f} b={fits[1][1]:+.3f} c={fits[1][2]:+.3f} n={n1} rms={rms[1]:.3f}")
print(f"crossings in range: {[round(c,3) for c in cross]}")
print(f"ARI(champion spectral k=2, para2) = {ari:.3f}")

C0, C1 = '#f4c430', '#6a3d9a'   # gold / purple (as in earlier figures)
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('ds19 — two interleaved parabolas: the truth vs what the champion does', fontsize=15)

# ---- A. truth -------------------------------------------------------------
ax = axes[0][0]
for i, (lab_i, col) in enumerate(((0, C0), (1, C1))):
    m = para_lab == lab_i
    ax.scatter(Q[m, 0], Q[m, 1], s=9, c=col, alpha=0.75, edgecolors='none',
               label=f'parabola {lab_i}  (n={m.sum()}, a={fits[lab_i][0]:+.3f}, rms={rms[lab_i]:.2f})')
    xs = np.linspace(Q[:, 0].min(), Q[:, 0].max(), 400)
    ax.plot(xs, fits[lab_i][0]*xs**2 + fits[lab_i][1]*xs + fits[lab_i][2], '-', lw=2.2, c=col)
for cx in cross:
    ax.axvline(cx, color='red', ls=':', lw=1.6)
    ax.annotate(f'crossing\nx={cx:.2f}', xy=(cx, Q[:, 1].max()*0.98),
                xytext=(cx+0.15, Q[:, 1].max()*0.93), color='red', fontsize=9)
ax.set_title(f'A. The actual structure: EM of two quadratics on (x1s, x5s) — ARI vs champion = {ari:.3f}')
ax.set_xlabel('x1 (std)'); ax.set_ylabel('x5 (std)')
ax.legend(loc='lower left', fontsize=9)

# ---- B. champion ------------------------------------------------------------
ax = axes[0][1]
cmap_b = plt.cm.Blues(np.linspace(0.35, 0.9, 2))
for i in (0, 1):
    m = champ_lab == i
    ax.scatter(Q[m, 0], Q[m, 1], s=9, c=cmap_b[i], alpha=0.75, edgecolors='none',
               label=f'cluster {i} (n={m.sum()})')
ax.set_title('B. Champion: spectral k=2 on PCA-4 — cuts the curve positionally (left vs right)')
ax.set_xlabel('x1 (std)'); ax.set_ylabel('x5 (std)')
ax.legend(loc='lower left', fontsize=9)

# ---- C. residuals ------------------------------------------------------------
ax = axes[1][0]
for i, col in ((0, C0), (1, C1)):
    m = para_lab == i
    res = Q[m, 1] - (fits[i][0]*Q[m, 0]**2 + fits[i][1]*Q[m, 0] + fits[i][2])
    ax.scatter(Q[m, 0], res, s=9, c=col, alpha=0.6, edgecolors='none', label=f'arc {i} residuals')
ax.axhline(0, color='k', lw=0.8)
ax.set_title(f'C. Residuals vs quadratic fit — tight bands (rms {rms[0]:.2f} / {rms[1]:.2f})')
ax.set_xlabel('x1 (std)'); ax.set_ylabel('residual (std)')
ax.legend(loc='lower left', fontsize=9)

# ---- D. membership along the curve -------------------------------------------
ax = axes[1][1]
bins = np.linspace(Q[:, 0].min(), Q[:, 0].max(), 11)
fractions, centers = [], []
for b in range(10):
    m = (Q[:, 0] >= bins[b]) & (Q[:, 0] < bins[b+1])
    if m.sum() > 0:
        fractions.append((para_lab[m] == 0).mean())
        centers.append(0.5*(bins[b]+bins[b+1]))
ax.plot(centers, fractions, '-o', c=C0, lw=2)
for cx in cross:
    ax.axvline(cx, color='red', ls=':', lw=1.6)
ax.set_title('D. Fraction of arc 0 along the curve — no single flip point (arcs interleave)')
ax.set_xlabel('x1 (std)'); ax.set_ylabel('fraction arc 0')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = 'dev/v10/ds19_parabola_story.png'
plt.savefig(out, dpi=150)
print(f"saved {out}")
