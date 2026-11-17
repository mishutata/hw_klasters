"""Scan all 28 planes: in which one do the champion's 2 clusters look like 2 THIN arcs?
Metric per plane: for each cluster, rms distance to its best parametric quad curve,
normalized by cluster std. Low = thin line."""
import numpy as np, pandas as pd, itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering

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

def thinness(P):
    """rms distance to best parametric quad through P, normalized by overall std"""
    cen = P.mean(axis=0); Pc = P - cen
    u, s, vt = np.linalg.svd(Pc, full_matrices=False)
    t = Pc @ u[0]
    C = [np.polyfit(t, Pc[:, d], 2) for d in range(2)]
    pts = np.stack([cen[d] + np.polyval(C[d], t) for d in range(2)], axis=1)
    rms = np.sqrt(((pts - P)**2).sum(axis=1)).mean()
    return rms / (np.sqrt((P**2).mean()) + 1e-9)

res = []
for i, j in itertools.combinations(range(len(cols)), 2):
    Q = Xs[:, [i, j]]
    th = [thinness(Q[lab == g]) for g in (0, 1)]
    res.append((max(th), cols[i], cols[j], th[0], th[1]))
res.sort()
print(f"{'pair':>8}  {'worst thinness':>14}   per-cluster")
for w, a, b, t0, t1 in res[:12]:
    print(f"{a+'-'+b:>8}  {w:14.3f}   {t0:.3f} / {t1:.3f}")
