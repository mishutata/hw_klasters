"""ds19 v2: two parabolas, robust (trimmed) fit per arm, each arm picks best orientation
(PCA-frame diagonal / y=f(x) / x=g(y)); every point assigned to nearest parabola. No spectral."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score

df = pd.read_csv('public_data/train.csv')
d19 = df[df.dataset_id == 'ds19']
Xraw = d19[[c for c in d19.columns if c.startswith('x')]].values.astype(float)
mask = ~np.all(np.isnan(Xraw), axis=0)
cols = [c for c, m in zip([c for c in d19.columns if c.startswith('x')], mask) if m]
X = Xraw[:, mask].copy()
ni = np.where(np.isnan(X))
if len(ni[0]): X[ni] = np.take(np.nanmean(X, axis=0), ni[1])
mu = X.mean(axis=0); sd = X.std(axis=0)
Xs = (X - mu) / sd
ix6, ix7 = cols.index('x6'), cols.index('x7')
Q = Xs[:, [ix6, ix7]]

Z4 = PCA(4, random_state=42).fit_transform(Xs)
lab_champ = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=20,
                               assign_labels='kmeans', random_state=42).fit_predict(Z4)

def make_curve(form, P):
    if form == 'PCA':
        m = P.mean(axis=0); Pc = P - m
        w, V = np.linalg.eigh(np.cov(Pc.T))
        u_ax = V[:, np.argmax(w)]; v_ax = np.array([-u_ax[1], u_ax[0]])
        u = Pc @ u_ax; v = Pc @ v_ax
        a, b, c = np.polyfit(u, v, 2)
        lo, hi = u.min(), u.max()
        def sample(n=400):
            uu = np.linspace(lo, hi, n); q = a*uu**2 + b*uu + c
            return m + np.outer(uu, u_ax) + np.outer(q, v_ax)
        rms = float(np.sqrt(np.mean((v - (a*u**2 + b*u + c))**2)))
        return sample, rms, f'PCA-frame  v={a:+.4f}u^2 {b:+.4f}u {c:+.4f}'
    if form == 'YX':   # y = f(x)
        a, b, c = np.polyfit(P[:, 0], P[:, 1], 2)
        lo, hi = P[:, 0].min(), P[:, 0].max()
        def sample(n=400):
            xx = np.linspace(lo, hi, n); return np.stack([xx, a*xx**2 + b*xx + c], axis=1)
        rms = float(np.sqrt(np.mean((P[:, 1] - (a*P[:, 0]**2 + b*P[:, 0] + c))**2)))
        return sample, rms, f'y=f(x)  {a:+.4f}x^2 {b:+.4f}x {c:+.4f}'
    a, b, c = np.polyfit(P[:, 1], P[:, 0], 2)   # XY: x = g(y)
    lo, hi = P[:, 1].min(), P[:, 1].max()
    def sample(n=400):
        yy = np.linspace(lo, hi, n); return np.stack([a*yy**2 + b*yy + c, yy], axis=1)
    rms = float(np.sqrt(np.mean((P[:, 0] - (a*P[:, 1]**2 + b*P[:, 1] + c))**2)))
    return sample, rms, f'x=g(y)  {a:+.4f}y^2 {b:+.4f}y {c:+.4f}'

def dist_to(Pm, sample):
    cv = sample()
    return np.linalg.norm(Pm[:, None, :] - cv[None, :, :], axis=2).min(axis=1)

def fit_robust(P):
    best = None
    for form in ('PCA', 'YX', 'XY'):
        sample, rms, desc = make_curve(form, P)
        for _ in range(3):                       # trim + refit
            d = dist_to(P, sample)
            keep = d <= max(np.quantile(d, 0.6), 0.35)
            if keep.sum() < 8: break
            sample, rms, desc = make_curve(form, P[keep])
        d = dist_to(P, sample)
        thr = max(np.quantile(d, 0.6), 0.35); keep = d <= thr
        stat = (float(d[keep].mean()), int(keep.sum()))
        if best is None or stat[0] < best[1]:
            best = (sample, stat[0], desc + f'  [inliers {stat[1]}/{len(P)}, mean d {stat[0]:.3f}]')
    return best

def assign_nearest(Qp, samples):
    D = np.stack([dist_to(Qp, s) for s in samples], axis=1)
    return D.argmin(axis=1), D

lab = KMeans(2, n_init=10, random_state=42).fit_predict(Q)
for it in range(20):
    fits = [fit_robust(Q[lab == g]) for g in (0, 1)]
    lab_new, D = assign_nearest(Q, [f[0] for f in fits])
    line = f"iter {it:2d}: sizes {(lab==0).sum():4d}/{(lab==1).sum():4d} | " + \
           " | ".join(f"g{g}: {fits[g][2]}" for g in (0, 1))
    print(line)
    if np.array_equal(lab_new, lab):
        print("  stable")
        break
    lab = lab_new

cens = [Q[lab == g].mean(axis=0) for g in (0, 1)]
gY = int(np.argmin([c[0] - c[1] for c in cens])); gR = 1 - gY
print(f"\nYELLOW arm: group {gY}, n={(lab==gY).sum()}")
print(f"RED    arm: group {gR}, n={(lab==gR).sum()}")
print(f"ARI(nearest-parabola v2, champion spectral):", round(adjusted_rand_score(lab_champ, lab), 4))

fig, ax = plt.subplots(figsize=(7.5, 7))
for name, g, col in (('YELLOW', gY, '#f4c430'), ('RED', gR, '#e63946')):
    cv = fits[g][0]()
    ax.plot(cv[:, 0], cv[:, 1], color=col, lw=2.6, alpha=.95, zorder=5, label=f'{name} parabola')
for name, g, col in (('YELLOW', gY, '#f4c430'), ('RED', gR, '#e63946')):
    Pm = Q[lab == g]
    ax.scatter(Pm[:, 0], Pm[:, 1], s=9, c=col, alpha=.85, edgecolors='none', zorder=4,
               label=f'{name} points (n={len(Pm)})')
ax.set_xlabel('x6 (std)'); ax.set_ylabel('x7 (std)')
ax.legend(fontsize=8, loc='upper right')
ax.set_title('ds19 v2: robust two-parabola fit; each point -> nearest parabola (no spectral)')
plt.tight_layout()
plt.savefig('dev/v10/ds19_red_yellow_formulas.png', dpi=130)
print('saved dev/v10/ds19_red_yellow_formulas.png')
