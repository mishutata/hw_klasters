"""ds19 v5: yellow FROZEN. Red = v4 curve (v3 spine rotated +15 deg left about red
centroid), then TIGHTENED (x-compressed 10% about the same pivot) and nudged +0.10 right.
Affine image of a parabola -> still an exact parabola; formulas via exact t-fit.
Every point -> nearest of {frozen yellow, new red}. No spectral."""
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
    if form == 'YX':
        a, b, c = np.polyfit(P[:, 0], P[:, 1], 2)
        lo, hi = P[:, 0].min(), P[:, 0].max()
        def sample(n=400):
            xx = np.linspace(lo, hi, n); return np.stack([xx, a*xx**2 + b*xx + c], axis=1)
        rms = float(np.sqrt(np.mean((P[:, 1] - (a*P[:, 0]**2 + b*P[:, 0] + c))**2)))
        return sample, rms, f'y=f(x)  {a:+.4f}x^2 {b:+.4f}x {c:+.4f}'
    a, b, c = np.polyfit(P[:, 1], P[:, 0], 2)
    lo, hi = P[:, 1].min(), P[:, 1].max()
    def sample(n=400):
        yy = np.linspace(lo, hi, n); return np.stack([a*yy**2 + b*yy + c, yy], axis=1)
    rms = float(np.sqrt(np.mean((P[:, 0] - (a*P[:, 1]**2 + b*P[:, 1] + c))**2)))
    return sample, rms, f'x=g(y)  {a:+.4f}y^2 {b:+.4f}y {c:+.4f}'

def dist_to(Pm, cv):
    return np.linalg.norm(Pm[:, None, :] - cv[None, :, :], axis=2).min(axis=1)

def fit_robust(P):
    best = None
    for form in ('PCA', 'YX', 'XY'):
        sample, rms, desc = make_curve(form, P)
        for _ in range(3):
            d = dist_to(P, sample())
            keep = d <= max(np.quantile(d, 0.6), 0.35)
            if keep.sum() < 8: break
            sample, rms, desc = make_curve(form, P[keep])
        d = dist_to(P, sample())
        thr = max(np.quantile(d, 0.6), 0.35); keep = d <= thr
        stat = (float(d[keep].mean()), int(keep.sum()))
        if best is None or stat[0] < best[1]:
            best = (sample, stat[0], desc + f'  [inliers {stat[1]}/{len(P)}, mean d {stat[0]:.3f}]')
    return best

lab = KMeans(2, n_init=10, random_state=42).fit_predict(Q)
for it in range(20):
    fits = [fit_robust(Q[lab == g]) for g in (0, 1)]
    Dm = np.stack([dist_to(Q, f[0]()) for f in fits], axis=1)
    lab_new = Dm.argmin(axis=1)
    if np.array_equal(lab_new, lab): break
    lab = lab_new
cens = [Q[lab == g].mean(axis=0) for g in (0, 1)]
gY = int(np.argmin([c[0] - c[1] for c in cens])); gR = 1 - gY
yellow_cv = fits[gY][0]()            # FROZEN — perfect per user
red_P = Q[lab == gR]

def ridge_xgY(P, bw=0.25):
    y = P[:, 1]
    edges = np.arange(y.min(), y.max() + 1e-9, bw)
    yc, xc, wts = [], [], []
    for i in range(len(edges) - 1):
        m = (y >= edges[i]) & (y < edges[i+1])
        if m.sum() < 3: continue
        yc.append(y[m].mean()); xc.append(np.median(P[m, 0])); wts.append(m.sum())
    c3 = np.polyfit(np.array(yc), np.array(xc), 2, w=np.array(wts))
    lo, hi = y.min(), y.max()
    return lambda n=400: np.stack([np.polyval(c3, np.linspace(lo, hi, n)), np.linspace(lo, hi, n)], axis=1)

def rotate(cv, ang_deg, pivot):
    th = np.radians(ang_deg); c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return (cv - pivot) @ R.T + pivot

pivot = red_P.mean(axis=0)
v4_red = rotate(ridge_xgY(red_P)(), +15, pivot)        # v4: rotated left

# ---------- v5: tighten (x-compress 10% about pivot) + nudge +0.10 right ----------
S, DX = 0.90, +0.10
new_red_cv = np.column_stack([pivot[0] + S * (v4_red[:, 0] - pivot[0]) + DX, v4_red[:, 1]])

d_old = dist_to(red_P, v4_red); k_old = d_old <= max(np.quantile(d_old, 0.6), 0.35)
d_new = dist_to(red_P, new_red_cv); k_new = d_new <= max(np.quantile(d_new, 0.6), 0.35)
print(f"pivot (red centroid): {pivot}")
print(f"v4 red:  inliers {k_old.sum()}/{len(red_P)}, mean d {d_old[k_old].mean():.3f}, "
      f"x [{v4_red[:,0].min():+.3f},{v4_red[:,0].max():+.3f}]")
print(f"v5 red:  inliers {k_new.sum()}/{len(red_P)}, mean d {d_new[k_new].mean():.3f}, "
      f"x [{new_red_cv[:,0].min():+.3f},{new_red_cv[:,0].max():+.3f}]")

Dm = np.stack([dist_to(Q, yellow_cv), dist_to(Q, new_red_cv)], axis=1)
lab_new = Dm.argmin(axis=1)
print(f"\nnew division: yellow n={int((lab_new==0).sum())}, red n={int((lab_new==1).sum())}")
print("ARI(v5 nearest-parabola, champion spectral):",
      round(adjusted_rand_score(lab_champ, np.where(lab_new == 0, gY, gR)), 4))

# ---------- exact formulas (affine t in [0,1], bottom tip -> top tip) ----------
t = np.linspace(0, 1, len(new_red_cv))
cx_ = np.polyfit(t, new_red_cv[:, 0], 2)
cy_ = np.polyfit(t, new_red_cv[:, 1], 2)
print("exact t-fit residuals:", np.abs(np.polyval(cx_, t)-new_red_cv[:,0]).max(),
      np.abs(np.polyval(cy_, t)-new_red_cv[:,1]).max())
def fmt3(c0, c1, c2):
    return f"{c0:+.5f} {c1:+.5f}t {c2:+.5f}t^2"
print("\nRED parabola v5 (t: bottom tip -> top tip):")
print(f"  x6s(t) = {fmt3(*cx_[::-1])}")
print(f"  x7s(t) = {fmt3(*cy_[::-1])}")
for nm, k, co in (("x6", ix6, cx_), ("x7", ix7, cy_)):
    c2, c1, c0 = co
    print(f"  {nm}(t) = {fmt3(mu[k] + sd[k]*c0, sd[k]*c1, sd[k]*c2)}   [raw]")

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(7.5, 7))
ax.plot(v4_red[:, 0], v4_red[:, 1], color='#e63946', lw=1.2, ls='--', alpha=.55,
        label='red before (v4: rotated 15 deg left)')
ax.plot(yellow_cv[:, 0], yellow_cv[:, 1], color='#f4c430', lw=2.6, alpha=.95, zorder=5,
        label='YELLOW parabola (frozen)')
ax.plot(new_red_cv[:, 0], new_red_cv[:, 1], color='#e63946', lw=2.6, alpha=.95, zorder=5,
        label='RED parabola (tighter + right)')
for g, col, nm in ((0, '#f4c430', 'YELLOW'), (1, '#e63946', 'RED')):
    Pm = Q[lab_new == g]
    ax.scatter(Pm[:, 0], Pm[:, 1], s=9, c=col, alpha=.85, edgecolors='none', zorder=4,
               label=f'{nm} points (n={len(Pm)})')
ax.set_xlabel('x6 (std)'); ax.set_ylabel('x7 (std)')
ax.legend(fontsize=8, loc='upper right')
ax.set_title('ds19 v5: red hook tightened 10% and nudged right; yellow frozen; points -> nearest parabola')
plt.tight_layout()
plt.savefig('dev/v10/ds19_red_yellow_formulas.png', dpi=130)
print('\nsaved dev/v10/ds19_red_yellow_formulas.png')
