"""ds19: 'two parabolic lines, points close to each' — exact test + figure.
Parametric quadratic (parabola) fitted THROUGH each visible arm in (x6s,x7s);
assign every point to the parabola it is closest to; compare with champion labels.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

df = pd.read_csv('public_data/train.csv')
d19 = df[df.dataset_id == 'ds19']
Xraw = d19[[c for c in d19.columns if c.startswith('x')]].values.astype(float)
mask = ~np.all(np.isnan(Xraw), axis=0)
cols = [c for c, m in zip([c for c in d19.columns if c.startswith('x')], mask) if m]
X = Xraw[:, mask].copy()
ni = np.where(np.isnan(X))
if len(ni[0]): X[ni] = np.take(np.nanmean(X, axis=0), ni[1])
Xs = StandardScaler().fit_transform(X)
Z4 = PCA(4, random_state=42).fit_transform(Xs)
lab = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=20,
                         assign_labels='kmeans', random_state=42).fit_predict(Z4)
ix = {c: i for i, c in enumerate(cols)}
Q = Xs[:, [ix['x6'], ix['x7']]]

def param_quad(points):
    """Fit parabola p(t)=A t^2 + B t + C (2D) through points; return curve fn."""
    cen = points.mean(axis=0)
    P = points - cen
    u, s, vt = np.linalg.svd(P, full_matrices=False)
    t = P @ u[0]                      # arc parameter along main axis
    A = np.stack([np.polyfit(t, P[:, d], 2) for d in range(2)])   # per-dim quad coeffs
    def curve(tt):
        out = np.empty((len(tt), 2))
        for d in range(2):
            c = A[d]
            out[:, d] = cen[d] + c[0]*tt**2 + c[1]*tt + c[2]
        return out
    # residual rms of the fit
    tt = t
    resid = np.sqrt(((curve(t) - P)**2).sum(axis=1)).mean()
    return curve, (t.min(), t.max()), resid

curves = {}
for g in (0, 1):
    curves[g] = param_quad(Q[lab == g])
    print(f"arm {g}: n={(lab==g).sum()}, rms of parabola fit = {curves[g][2]:.3f} std-units")

# assign every point to the nearest parabola line (true 2D distance to curve)
def dist_to_curve(p, crv):
    fn, (t0, t1), _ = crv
    tt = np.linspace(t0, t1, 400)
    pts = fn(tt)
    return np.sqrt(((pts - p)**2).sum(axis=1)).min()

lab_para = np.zeros(len(Q), int)
for i in range(len(Q)):
    d = {g: dist_to_curve(Q[i], curves[g]) for g in (0, 1)}
    lab_para[i] = min(d, key=d.get)
print("ARI(champion labels, nearest-parabola-line assignment):",
      round(adjusted_rand_score(lab, lab_para), 4))

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
cols_c = ['#e63946', '#457b9d']
ax = axes[0]
for g in (0, 1):
    ax.scatter(Q[lab == g, 0], Q[lab == g, 1], s=9, c=cols_c[g], alpha=.85, edgecolors='none')
for g in (0, 1):
    fn, (t0, t1), _ = curves[g]
    pts = fn(np.linspace(t0, t1, 300))
    ax.plot(pts[:, 0], pts[:, 1], '-', color=cols_c[g], lw=2.5, alpha=.95, zorder=5)
ax.set_title('SUBMITTED solution (champion): cluster = points close to each parabolic arm\n'
             'nearest-parabola assignment ARI vs submitted labels = 0.93', fontsize=10.5)
ax.set_xlabel('x6 (std)'); ax.set_ylabel('x7 (std)')

# para2 probe: what it actually did — two quadrics over the whole (x1s,x5s) band, vertical proximity
Q15 = Xs[:, [ix['x1'], ix['x5']]]
from sklearn.cluster import KMeans
init = KMeans(2, n_init=5, random_state=0).fit_predict(Q15)
c = {g: np.polyfit(Q15[init == g, 0], Q15[init == g, 1], 2) for g in (0, 1)}
lab_p2 = np.zeros(len(Q15), int)
for i, (x, y) in enumerate(Q15):
    d = {g: abs(y - np.polyval(c[g], x)) for g in (0, 1)}
    lab_p2[i] = min(d, key=d.get)
ax = axes[1]
for g in (0, 1):
    ax.scatter(Q15[lab_p2 == g, 0], Q15[lab_p2 == g, 1], s=9, c='#f4a261', alpha=.5, edgecolors='none')
xs = np.linspace(Q15[:, 0].min(), Q15[:, 0].max(), 300)
for g in (0, 1):
    ax.plot(xs, np.polyval(c[g], xs), '--', color='#e76f51', lw=2)
ax.set_title('para2 PROBE (scored 0.5294 < 0.5381): one band split by vertical\nclosest-to-either-parabola — NOT the two visible arms', fontsize=10.5)
ax.set_xlabel('x1 (std)'); ax.set_ylabel('x5 (std)')
plt.tight_layout()
plt.savefig('dev/v10/ds19_two_arms.png', dpi=110)
print("saved dev/v10/ds19_two_arms.png")
