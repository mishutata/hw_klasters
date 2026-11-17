import sys
sys.path.insert(0, 'my_solver')
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from solution import _clean, _reduce, cluster, K_PRIOR, NORM_BEST

df = pd.read_csv('public_data/train.csv')
print(f"{'ds':5s} {'k':>2s} {'winner':18s} {'sizes':>14s} {'sil':>6s}  sil-by-dim@k (2..8)")
for ds, g in sorted(df.groupby('dataset_id')):
    X = g.drop(columns=['id', 'dataset_id']).values.astype(float)
    lab = cluster(X, ds)
    # reduced space for silhouette
    Xc = _clean(X)
    norm = NORM_BEST[ds]['method']
    if norm == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        Z0 = MinMaxScaler().fit_transform(Xc)
    elif norm == 'robust':
        from sklearn.preprocessing import RobustScaler
        Z0 = RobustScaler().fit_transform(Xc)
    else:
        Z0 = StandardScaler().fit_transform(np.sign(Xc) * (np.log1p(np.abs(Xc))))
    evr_full = PCA(n_components=min(8, Xc.shape[1]), svd_solver='full').fit(Z0).explained_variance_ratio_
    sils = []
    for d in range(2, min(9, len(evr_full))):
        Z = PCA(n_components=d, random_state=42).fit_transform(Z0)
        km = KMeans(K_PRIOR[ds], n_init=15, random_state=42).fit(Z)
        sils.append(silhouette_score(Z, km.labels_))
    sizes = np.bincount(lab)
    print(f"{ds:5s} {K_PRIOR[ds]:2d} {'?':18s} {str(sizes):>14s} " + '  '.join(f'{s:4.2f}' for s in sils))
