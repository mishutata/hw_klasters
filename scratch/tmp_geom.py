import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
print(f"{'ds':5} {'n':>5} {'d':>2} {'pc1':>5} {'top3%':>6} {'sil k=2..8'}")
for ds, g in df.groupby('dataset_id'):
    X = np.asarray(g[xcols].values, dtype=float)
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep]
    for j in range(X.shape[1]):
        b = np.isnan(X[:,j]); X[b,j] = np.nanmean(X[:,j]) if b.any() else 0
    Z = StandardScaler().fit_transform(X)
    p = PCA(random_state=42).fit(Z)
    evr = p.explained_variance_ratio_
    top3 = evr[:min(3,len(evr))].sum()
    sils = []
    for k in range(2, 9):
        lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z).labels_
        sils.append(round(silhouette_score(Z, lab), 3))
    print(f"{ds:5} {len(X):>5} {X.shape[1]:>2} {evr[0]:>5.2f} {top3:>6.2f} {' '.join(f'{s:.2f}' for s in sils)}")
