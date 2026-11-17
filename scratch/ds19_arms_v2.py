import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
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

def smooth_curve(points, deg=4):
    cen = points.mean(axis=0)
    P = points - cen
    u, s, vt = np.linalg.svd(P, full_matrices=False)
    t = P @ u[0]
    C = [np.polyfit(t, P[:, d], deg) for d in range(2)]
    def curve(tt):
        out = np.empty((len(tt), 2))
        for d in range(2):
            out[:, d] = cen[d] + np.polyval(C[d], tt)
        return out
    resid = np.sqrt(((curve(t) - P)**2).sum(axis=1)).mean()
    return curve, (t.min(), t.max()), resid

curves = {g: smooth_curve(Q[lab == g]) for g in (0, 1)}
for g in (0, 1):
    print(f"arm {g}: n={(lab==g).sum()}, rms deg-4 fit = {curves[g][2]:.3f}")

def dist_to_curve(p, crv):
    fn, (t0, t1), _ = crv
    pts = fn(np.linspace(t0, t1, 600))
    return np.sqrt(((pts - p)**2).sum(axis=1)).min()

lab_line = np.zeros(len(Q), int)
for i in range(len(Q)):
    d = {g: dist_to_curve(Q[i], curves[g]) for g in (0, 1)}
    lab_line[i] = min(d, key=d.get)
print("ARI(champion, nearest-line assignment, deg-4):", round(adjusted_rand_score(lab, lab_line), 4))

fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
cols_c = ['#e63946', '#457b9d']
ax = axes[0]
for g in (0, 1):
    ax.scatter(Q[lab == g, 0], Q[lab == g, 1], s=9, c=cols_c[g], alpha=.85, edgecolors='none',
               label=f'cluster {g} (n={(lab==g).sum()})')
for g in (0, 1):
    fn, (t0, t1), _ = curves[g]
    pts = fn(np.linspace(t0, t1, 300))
    ax.plot(pts[:, 0], pts[:, 1], '-', color=cols_c[g], lw=2.5, alpha=.95, zorder=5)
ax.set_title('SUBMITTED champion labels: two arms, points close to each line\n'
             f'(nearest-line agreement ARI = {adjusted_rand_score(lab, lab_line):.2f})', fontsize=10.5)
ax.set_xlabel('x6 (std)'); ax.set_ylabel('x7 (std)'); ax.legend(fontsize=8)

Q15 = Xs[:, [ix['x1'], ix['x5']]]
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
ax.set_title('para2 PROBE (submitted, scored 0.5294 < 0.5381): split by vertical\nclosest-to-either-parabola inside ONE band — not the two arms', fontsize=10.5)
ax.set_xlabel('x1 (std)'); ax.set_ylabel('x5 (std)')
plt.tight_layout()
plt.savefig('dev/v10/ds19_two_arms.png', dpi=110)
print("saved dev/v10/ds19_two_arms.png")
