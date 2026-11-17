"""ds19: is the champion's k=2 partition == 'two parabolic arms, points close to each'?
Plane of interest: (x6s, x7s) = row4/col5 panel with two visible lobes.
1) champion labels (exact test-time path)
2) fit one quadratic per lobe in that plane; assign each point to nearest arm -> ARI vs champion
3) figure: scatter colored by champion labels + the two fitted parabolas overlaid
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

# exact champion path for ds19 (local, not pooled)
Xs = StandardScaler().fit_transform(X)
Z4 = PCA(4, random_state=42).fit_transform(Xs)
lab = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=20,
                         assign_labels='kmeans', random_state=42).fit_predict(Z4)

ix = {c: i for i, c in enumerate(cols)}
Q = Xs[:, [ix['x6'], ix['x7']]]   # the (x6s, x7s) plane

# fit a quadratic per lobe in this plane, then assign each point to nearest arm
def quad_fit(xy):  # x -> y parabola
    c = np.polyfit(xy[:, 0], xy[:, 1], 2)
    return c
arms = {}
for g in (0, 1):
    arms[g] = quad_fit(Q[lab == g])

def assign_nearest_arm(Qp):
    out = np.zeros(len(Qp), int)
    for i, (x, y) in enumerate(Qp):
        d = {g: abs(y - np.polyval(c, x)) / max(np.std(Q[:, 1]), 1e-9) for g, c in arms.items()}
        out[i] = min(d, key=d.get)
    return out

lab_nearest = assign_nearest_arm(Q)
print("ARI(nearest-arm assignment in (x6,x7) plane, champion labels):",
      round(adjusted_rand_score(lab, lab_nearest), 4))

# also: is the split positional along x6? boundary location
order = np.argsort(Q[:, 0])
srt = lab[order]
ch = np.where(srt[1:] != srt[:-1])[0] + 1
print(f"transitions in x6-sorted order: {len(ch)}, boundary at x6*={Q[order,0][int(np.median(ch))]:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
cols_c = ['#e63946', '#457b9d']
ax = axes[0]
for g in (0, 1):
    ax.scatter(Q[lab == g, 0], Q[lab == g, 1], s=9, c=cols_c[g], alpha=.8, edgecolors='none', label=f'cluster {g}')
xs = np.linspace(Q[:, 0].min(), Q[:, 0].max(), 300)
for g in (0, 1):
    ax.plot(xs, np.polyval(arms[g], xs), '-', color=cols_c[g], lw=2.2, alpha=.9, label=f'parabola {g} fit')
ax.set_title('CHAMPION labels = two parabolic arms;\npoints near each arm form one cluster', fontsize=11)
ax.legend(fontsize=8); ax.set_xlabel('x6 (std)'); ax.set_ylabel('x7 (std)')

# para2 style: vertical-proximity split of the whole band by 2 quadratics (what the probe did)
from sklearn.cluster import KMeans
Qp = Q.copy()
init = KMeans(2, n_init=5, random_state=0).fit_predict(Qp[:, :1] * 0 + np.stack([Qp[:,0], Qp[:,1]], axis=1))
c0 = quad_fit(Q[init == 0]); c1 = quad_fit(Q[init == 1])
lab_para2 = assign_nearest_arm.__wrapped__ if False else None
out = np.zeros(len(Q), int)
for i, (x, y) in enumerate(Q):
    d = {0: abs(y - np.polyval(c0, x)), 1: abs(y - np.polyval(c1, x))}
    out[i] = min(d, key=d.get)
ax = axes[1]
for g in (0, 1):
    ax.scatter(Q[out == g, 0], Q[out == g, 1], s=9, c='#f4a261', alpha=.5, edgecolors='none')
xs = np.linspace(Q[:, 0].min(), Q[:, 0].max(), 300)
for g, c in zip((0, 1), (c0, c1)):
    ax.plot(xs, np.polyval(c, xs), '--', color='#e76f51', lw=2)
ax.set_title('para2 probe: 2 quadrics over the WHOLE band,\npoints assigned by vertical proximity (scored LOWER)', fontsize=11)
ax.set_xlabel('x6 (std)'); ax.set_ylabel('x7 (std)')
plt.tight_layout()
plt.savefig('dev/v10/ds19_two_arms.png', dpi=110)
print("saved dev/v10/ds19_two_arms.png")
