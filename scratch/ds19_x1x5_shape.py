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

for g in (0, 1):
    P = Q[lab == g]
    print(f"cluster {g}: n={len(P)}, x1 [{P[:,0].min():.2f},{P[:,0].max():.2f}], "
          f"x5 [{P[:,1].min():.2f},{P[:,1].max():.2f}]")

# positional cut along x1?
cut = (Q[:, 0] < np.median(Q[:, 0])).astype(int)
print("ARI(champion, median-x1 cut):", round(adjusted_rand_score(lab, cut), 4))
# where's the boundary in x1-sorted order?
order = np.argsort(Q[:, 0]); srt = lab[order]
ch = np.where(srt[1:] != srt[:-1])[0] + 1
print(f"transitions in x1-sorted: {len(ch)}, boundary x1*={Q[order,0][int(np.median(ch))]:.2f}")

# overlap of the two clusters' x1 ranges
lo = max(Q[lab==0,0].min(), Q[lab==1,0].min()); hi = min(Q[lab==0,0].max(), Q[lab==1,0].max())
print(f"x1 overlap: [{lo:.2f},{hi:.2f}] of total span {Q[:,0].min():.2f}..{Q[:,0].max():.2f}")

# is each cluster a function x5=f(x1)? (thin arc check already passed) — fit quad per cluster
for g in (0, 1):
    P = Q[lab == g]
    c = np.polyfit(P[:, 0], P[:, 1], 2)
    r = 1 - np.sum((P[:,1]-np.polyval(c,P[:,0]))**2)/np.sum((P[:,1]-P[:,1].mean())**2)
    print(f"cluster {g}: quad x5~x1 R2={r:.3f}, coeffs a={c[0]:+.3f}")
