"""ds19 v3: freeze the yellow parabola (perfect per user), TIGHTEN the red hook:
fit x = g(y) through binned-median spine of the red points (weighted quadratic).
Every point assigned to nearest of {frozen yellow, tightened red}. No spectral."""
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

# ---------- reproduce v2 exactly (same deterministic EM) ----------
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

def dist_to(Pm, sample):
    cv = sample()
    return np.linalg.norm(Pm[:, None, :] - cv[None, :, :], axis=2).min(axis=1)

def fit_robust(P):
    best = None
    for form in ('PCA', 'YX', 'XY'):
        sample, rms, desc = make_curve(form, P)
        for _ in range(3):
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

lab = KMeans(2, n_init=10, random_state=42).fit_predict(Q)
for it in range(20):
    fits = [fit_robust(Q[lab == g]) for g in (0, 1)]
    lab_new, D = assign_nearest = None, None
    Dm = np.stack([dist_to(Q, f[0]) for f in fits], axis=1)
    lab_new = Dm.argmin(axis=1)
    if np.array_equal(lab_new, lab): break
    lab = lab_new
cens = [Q[lab == g].mean(axis=0) for g in (0, 1)]
gY = int(np.argmin([c[0] - c[1] for c in cens])); gR = 1 - gY
yellow_sample = fits[gY][0]          # FROZEN — user says perfect
red_P = Q[lab == gR]
old_red = fits[gR][0]()
print(f"v2 recap: yellow n={(lab==gY).sum()}, red n={(lab==gR).sum()}")
print(f"old red curve: x-range [{old_red[:,0].min():.3f}, {old_red[:,0].max():.3f}], "
      f"y-range [{old_red[:,1].min():.3f}, {old_red[:,1].max():.3f}]")

# ---------- tighten red: spine (binned medians) + weighted quadratic x = g(y) ----------
def ridge_xgY(P, bw=0.25):
    y = P[:, 1]
    edges = np.arange(y.min(), y.max() + 1e-9, bw)
    yc, xc, wts = [], [], []
    for i in range(len(edges) - 1):
        m = (y >= edges[i]) & (y < edges[i+1])
        if m.sum() < 3: continue
        yc.append(y[m].mean()); xc.append(np.median(P[m, 0])); wts.append(m.sum())
    yc, xc, wts = np.array(yc), np.array(xc), np.array(wts)
    c3 = np.polyfit(yc, xc, 2, w=wts)
    lo, hi = y.min(), y.max()
    def sample(n=400):
        yy = np.linspace(lo, hi, n); return np.stack([np.polyval(c3, yy), yy], axis=1)
    d = dist_to(P, sample)
    keep = d <= max(np.quantile(d, 0.6), 0.35)
    print(f"ridge spine: {len(yc)} bins; x = {c3[0]:+.4f}y^2 {c3[1]:+.4f}y {c3[2]:+.4f}; "
          f"inliers {keep.sum()}/{len(P)}, mean d {d[keep].mean():.3f}")
    return sample, c3

new_red_sample, c3r = ridge_xgY(red_P)
nr = new_red_sample()
print(f"new red curve: x-range [{nr[:,0].min():.3f}, {nr[:,0].max():.3f}], "
      f"y-range [{nr[:,1].min():.3f}, {nr[:,1].max():.3f}]")

# ---------- reassign every point to nearest of {frozen yellow, tightened red} ----------
Dm = np.stack([dist_to(Q, yellow_sample), dist_to(Q, new_red_sample)], axis=1)
lab_new = Dm.argmin(axis=1)
nY = int((lab_new == (gY if True else 0)).sum())  # careful: map below
# lab_new index 0 = yellow curve, 1 = red curve
print(f"new division: yellow n={int((lab_new==0).sum())}, red n={int((lab_new==1).sum())}")
print(f"ARI(v3 nearest-parabola, champion spectral):", round(adjusted_rand_score(lab_champ,
      np.where(lab_new == 0, gY, gR)), 4))

# formulas for the tightened red (std + raw), t in [0,1] from top tip to bottom tip
y_lo, y_hi = nr[:, 1].min(), nr[:, 1].max()
a_, b_, c_ = c3r
def fmt_poly(coefs):
    terms = []
    for d2, aa in enumerate(coefs):
        if abs(aa) < 5e-4 and d2 > 0: continue
        tv = {0: '', 1: 't', 2: 't^2'}[d2]
        terms.append(f"{aa:+.5f}{tv}" if d2 else f"{aa:+.5f}")
    return " ".join(terms)
L = y_hi - y_lo
Aq, Bq, Cq = a_*L**2, 2*a_*y_lo*L + b_*L, a_*y_lo**2 + b_*y_lo + c_
print("\nRED parabola v3 (t: top tip -> bottom tip):")
print(f"  x6s(t) = {fmt_poly([Cq, Bq, Aq])}")
print(f"  x7s(t) = {fmt_poly([y_lo, L, 0.0])}")
for j, feat, k in ((0, 'x6', ix6), (1, 'x7', ix7)):
    coefs = [mu[k] + sd[k]*Cq, sd[k]*Bq, sd[k]*Aq] if j == 0 else [mu[k] + sd[k]*y_lo, sd[k]*L, 0.0]
    print(f"  {feat}(t) = {fmt_poly(coefs)}   [raw]")

# ---------- figures ----------
fig, ax = plt.subplots(figsize=(7.5, 7))
ax.plot(old_red[:, 0], old_red[:, 1], color='#e63946', lw=1.2, ls='--', alpha=.6, label='old red (v2)')
cvY = yellow_sample()
ax.plot(cvY[:, 0], cvY[:, 1], color='#f4c430', lw=2.6, alpha=.95, zorder=5, label='YELLOW parabola (frozen)')
ax.plot(nr[:, 0], nr[:, 1], color='#e63946', lw=2.6, alpha=.95, zorder=5, label='RED parabola (tightened)')
for g, col, nm in ((0, '#f4c430', 'YELLOW'), (1, '#e63946', 'RED')):
    Pm = Q[lab_new == g]
    ax.scatter(Pm[:, 0], Pm[:, 1], s=9, c=col, alpha=.85, edgecolors='none', zorder=4,
               label=f'{nm} points (n={len(Pm)})')
ax.set_xlabel('x6 (std)'); ax.set_ylabel('x7 (std)')
ax.legend(fontsize=8, loc='upper right')
ax.set_title('ds19 v3: yellow frozen; red hook tightened to its spine; points -> nearest parabola')
plt.tight_layout()
plt.savefig('dev/v10/ds19_red_yellow_formulas.png', dpi=130)
print('\nsaved dev/v10/ds19_red_yellow_formulas.png')
