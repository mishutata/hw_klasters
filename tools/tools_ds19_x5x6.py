"""ds19 in the (x5, x6) plane — requested view.

Left : points colored by the CANONICAL para2 membership (from (x1,x5)) +
        per-arc quadratic fits IN THIS PLANE (least squares, no EM)
Right: same points colored by the EM split found directly in (x5,x6)
        (both orientations, best rms) + its fits; ARI vs canonical printed.
Output: dev/v10/ds19_x5x6.png
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
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
i5, i6 = cols.index('x5'), cols.index('x6')
Q = Xs[:, [i5, i6]]          # (x5s, x6s)

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
            best = ((rms, -bal), lab.copy(), cs)
    return best[1], best[2], best[0][0]

# canonical membership from (x1,x5)
can_lab, _, can_rms = em_two_quadratics(Xs[:, [0, 4]])
print(f"canonical (x1,x5) rms_sum={can_rms:.3f}")

# EM directly in (x5,x6), both orientations
lab_yx, fits_yx, rms_yx = em_two_quadratics(Q)
lab_xy, fits_xy, rms_xy = em_two_quadratics(Q[:, ::-1])
print(f"EM in (x5,x6) y~f(x): rms_sum={rms_yx:.3f}  ARI vs canonical={adjusted_rand_score(lab_yx, can_lab):.3f}")
print(f"EM in (x5,x6) x~f(y): rms_sum={rms_xy:.3f}  ARI vs canonical={adjusted_rand_score(lab_xy, can_lab):.3f}")

# per-arc quadratic fits IN THIS PLANE using the CANONICAL labels
fits_can = []
for i in (0, 1):
    m = can_lab == i
    A = np.vstack([Q[m, 0]**2, Q[m, 0], np.ones(m.sum())]).T
    c, *_ = np.linalg.lstsq(A, Q[m, 1], rcond=None)
    fits_can.append(c)
    r = Q[m, 1] - (c[0]*Q[m,0]**2 + c[1]*Q[m,0] + c[2])
    print(f"canonical arc{i} fit in (x5,x6): a={c[0]:+.3f} b={c[1]:+.3f} c={c[2]:+.3f} rms={np.sqrt(np.mean(r**2)):.3f}")

# union-curve quality: x6 ~ f(x5) on all points
A = np.vstack([Q[:,0]**2, Q[:,0], np.ones(len(Q))]).T
cu, *_ = np.linalg.lstsq(A, Q[:,1], rcond=None)
r2 = 1 - np.sum((Q[:,1]-A@cu)**2)/np.sum((Q[:,1]-Q[:,1].mean())**2)
print(f"union curve x6~f(x5): R2={r2:.3f}")

C0, C1 = '#f4c430', '#6a3d9a'
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes[0]
for i in (0, 1):
    m = can_lab == i
    ax.scatter(Q[m, 0], Q[m, 1], s=9, c=(C0, C1)[i], alpha=0.75, edgecolors='none',
               label=f'arc {i} (canonical, n={m.sum()})')
    xs = np.linspace(Q[:, 0].min(), Q[:, 0].max(), 400)
    ax.plot(xs, fits_can[i][0]*xs**2 + fits_can[i][1]*xs + fits_can[i][2], '-', lw=2.2, c=(C0, C1)[i])
ax.set_title('ds19 (x5s, x6s): colored by the TRUE membership from (x1,x5)\n+ per-arc quadratic fits in this plane')
ax.set_xlabel('x5 (std)'); ax.set_ylabel('x6 (std)')
ax.legend(loc='upper left', fontsize=9)

lab, fits = (lab_yx, fits_yx) if rms_yx <= rms_xy else (lab_xy, fits_xy)
orient = 'y~f(x)' if rms_yx <= rms_xy else 'x~f(y)'
ax = axes[1]
for i in (0, 1):
    m = lab == i
    ax.scatter(Q[m, 0], Q[m, 1], s=9, c=(C0, C1)[i], alpha=0.75, edgecolors='none',
               label=f'EM cluster {i} (n={m.sum()})')
    if orient == 'y~f(x)':
        xs = np.linspace(Q[:, 0].min(), Q[:, 0].max(), 400)
        ax.plot(xs, fits[i][0]*xs**2 + fits[i][1]*xs + fits[i][2], '-', lw=2.2, c=(C0, C1)[i])
    else:
        ys = np.linspace(Q[:, 1].min(), Q[:, 1].max(), 400)
        ax.plot(fits[i][0]*ys**2 + fits[i][1]*ys + fits[i][2], ys, '-', lw=2.2, c=(C0, C1)[i])
ax.set_title(f'EM of two quadratics run directly in (x5,x6) [{orient}]\nARI vs canonical = {adjusted_rand_score(lab, can_lab):.3f}')
ax.set_xlabel('x5 (std)'); ax.set_ylabel('x6 (std)')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('dev/v10/ds19_x5x6.png', dpi=150)
print("saved dev/v10/ds19_x5x6.png")
