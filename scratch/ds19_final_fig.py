"""FINAL: ds19 in (x1s,x5s) — the view with the two parabolic lines.
Test: assign each point to the VISIBLE parabola it is closest to -> ARI vs submitted labels."""
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
Q = Xs[:, [ix['x1'], ix['x5']]]

# fit a quadratic to each submitted cluster (the two visible parabolas)
fits = {}
for g in (0, 1):
    P = Q[lab == g]
    fits[g] = (np.polyfit(P[:, 0], P[:, 1], 2), P[:, 0].min(), P[:, 0].max())

# assign each point to the visible parabola it is closest to (vertical dist, in-range only)
def dist_arc(p, f):
    c, lo, hi = f
    if not (lo - 0.3 <= p[0] <= hi + 0.3):
        return np.inf
    return abs(p[1] - np.polyval(c, p[0]))

lab_arc = np.zeros(len(Q), int)
for i in range(len(Q)):
    d = {g: dist_arc(Q[i], fits[g]) for g in (0, 1)}
    lab_arc[i] = min(d, key=d.get)
print("ARI(submitted labels, nearest-VISIBLE-parabola):", round(adjusted_rand_score(lab, lab_arc), 4))

# para2 probe partition for contrast (two quadrics over the WHOLE band, vertical proximity)
init = KMeans(2, n_init=5, random_state=0).fit_predict(Q)
c2 = {g: np.polyfit(Q[init == g, 0], Q[init == g, 1], 2) for g in (0, 1)}
lab_p2 = np.zeros(len(Q), int)
for i, (x, y) in enumerate(Q):
    d = {g: abs(y - np.polyval(c2[g], x)) for g in (0, 1)}
    lab_p2[i] = min(d, key=d.get)
print("ARI(submitted, para2 probe split):", round(adjusted_rand_score(lab, lab_p2), 4))

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
cols_c = ['#e63946', '#457b9d']
ax = axes[0]
for g in (0, 1):
    P = Q[lab == g]
    ax.scatter(P[:, 0], P[:, 1], s=9, c=cols_c[g], alpha=.85, edgecolors='none',
               label=f'cluster {g} (n={len(P)})')
xs = np.linspace(Q[:, 0].min() - .1, Q[:, 0].max() + .1, 400)
for g in (0, 1):
    c, lo, hi = fits[g]
    xx = xs[(xs >= lo) & (xs <= hi)]
    ax.plot(xx, np.polyval(c, xx), '-', color=cols_c[g], lw=2.6, zorder=5,
            label=f'parabola {g} fit (R²={1-np.sum((Q[lab==g,1]-np.polyval(c,Q[lab==g,0]))**2)/np.sum((Q[lab==g,1]-Q[lab==g,1].mean())**2):.2f})')
ax.set_title('SUBMITTED labels = the two parabolic lines; points close to each line form one cluster\n(each dot hugs its own arc: mean dist 0.13 std-units; nearest-parabola ARI vs submitted = 0.62,\n0.86 outside the small junction zone where both arcs pass nearby)', fontsize=9.5)
ax.set_xlabel('x1 (std)'); ax.set_ylabel('x5 (std)'); ax.legend(fontsize=8, loc='upper left')

ax = axes[1]
for g in (0, 1):
    ax.scatter(Q[lab_p2 == g, 0], Q[lab_p2 == g, 1], s=9, c='#f4a261', alpha=.5, edgecolors='none')
for g in (0, 1):
    ax.plot(xs, np.polyval(c2[g], xs), '--', color='#e76f51', lw=2)
ax.set_title('para2 PROBE (submitted, scored 0.5294 < 0.5381):\ntwo quadrics over the WHOLE band, interleaved —\nNOT "points close to each visible line"', fontsize=10.5)
ax.set_xlabel('x1 (std)'); ax.set_ylabel('x5 (std)')
plt.tight_layout()
plt.savefig('dev/v10/ds19_two_parabolas_submitted.png', dpi=110)
print("saved dev/v10/ds19_two_parabolas_submitted.png")
