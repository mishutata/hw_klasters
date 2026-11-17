import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

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

fits = {}
for g in (0, 1):
    P = Q[lab == g]
    fits[g] = (np.polyfit(P[:, 0], P[:, 1], 2), P[:, 0].min(), P[:, 0].max())

d0 = np.array([abs(Q[i,1]-np.polyval(fits[0][0], Q[i,0])) if fits[0][1]-0.3 <= Q[i,0] <= fits[0][2]+0.3 else np.inf for i in range(len(Q))])
d1 = np.array([abs(Q[i,1]-np.polyval(fits[1][0], Q[i,0])) if fits[1][1]-0.3 <= Q[i,0] <= fits[1][2]+0.3 else np.inf for i in range(len(Q))])
lab_arc = (d1 < d0).astype(int)

# ambiguous zone: both distances finite and close
amb = (np.isfinite(d0) & np.isfinite(d1) & (np.abs(d0-d1) < 0.15))
print(f"points in junction/ambiguous zone: {amb.sum()} / {len(Q)} ({100*amb.mean():.0f}%)")
ok = ~amb
print("ARI outside ambiguous zone:", round(adjusted_rand_score(lab[ok], lab_arc[ok]), 4))

# how far is each point from its own arc? (tightness of 'dots close to the line')
d_own = np.where(lab == 0, d0, d1)
print(f"distance to own parabola: mean={d_own.mean():.3f}, p95={np.percentile(d_own,95):.3f} std-units")

# and the gap between arcs at the junction: min distance between an arc-0 point and arc-1's curve
print("\nx1-ranges: cluster0", fits[0][1:].round(2), "cluster1", fits[1][1:].round(2))
