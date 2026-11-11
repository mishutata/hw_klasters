import numpy as np, pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import hdbscan

K_PRIOR = {"ds00":2,"ds01":3,"ds02":2,"ds03":2,"ds04":3,"ds05":3,"ds06":3,"ds07":3,
           "ds08":2,"ds09":3,"ds10":3,"ds11":4,"ds12":2,"ds13":2,"ds14":2,"ds15":4,
           "ds16":2,"ds17":2,"ds18":2,"ds19":4,"ds20":4,"ds21":3,"ds22":2,"ds23":2}

def clean(X):
    X = np.asarray(X, float)
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep] if keep.any() else np.zeros((len(X),1))
    for j in range(X.shape[1]):
        c = X[:,j]; b = np.isnan(c)
        if b.any(): X[b,j] = np.nanmean(c)
    return X

def reduce(X):
    Z = StandardScaler().fit_transform(X)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    if X.shape[1] > 3:
        p = PCA(random_state=42).fit(Z)
        cum = np.cumsum(p.explained_variance_ratio_)
        n = max(2, min(8, int(np.searchsorted(cum, 0.95)+1), X.shape[1]))
        Z = PCA(n_components=n, random_state=42).fit_transform(Z)
    return Z

def eps_knee(Z, k):
    d, _ = cKDTree(Z).query(Z, k=k+1)
    kd = np.sort(d[:, k])
    x = np.linspace(0, 1, len(kd))
    y = (kd - kd[0]) / max(kd[-1]-kd[0], 1e-9)
    return float(kd[int(np.argmax(y - x))])

def main():
    df = pd.read_csv('public_data/train.csv')
    rows = []
    for ds in sorted(df['dataset_id'].unique()):
        d = df[df['dataset_id']==ds]
        cols = [c for c in d.columns if c.startswith('x')]
        X = clean(d[cols].values); Z = reduce(X)
        n = len(Z); k0 = K_PRIOR[ds]

        km = KMeans(n_clusters=k0, n_init=15, random_state=42).fit(Z)
        lab_km = km.labels_
        cands = {'KM': lab_km}
        try:
            sp = SpectralClustering(n_clusters=k0, affinity='nearest_neighbors',
                                    n_neighbors=min(n-1, max(10,int(np.sqrt(n)))),
                                    random_state=42).fit_predict(Z)
            if len(np.unique(sp))==k0: cands['SP'] = sp
        except Exception: pass
        try:
            gm = GaussianMixture(n_components=k0, covariance_type='full', n_init=3,
                                 random_state=42).fit(Z)
            lab = gm.predict(Z)
            if len(np.unique(lab))==k0: cands['GM'] = lab
        except Exception: pass

        # DBSCAN: eps from k-distance knee, min_samples=5 and 10
        for ms in (5, 10):
            eps = eps_knee(Z, ms)
            cands[f'DB{ms}'] = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)

        # HDBSCAN with small min_cluster_size so k0 clusters survive
        mcs = max(10, n // (4*k0))
        try:
            cands['HD'] = hdbscan.HDBSCAN(min_cluster_size=mcs,
                                          cluster_selection_method='eom').fit(Z).labels_
        except Exception: pass

        row = {'ds': ds, 'n': n, 'k0': k0, 'dim': Z.shape[1]}
        for name, lab in cands.items():
            u = np.unique(lab)
            ncl = int(len(u[u>=0])) if (lab<0).any() else len(u)
            noise = float((lab<0).mean()) if (lab<0).any() else 0.0
            try:
                vv = lab[lab>=0] if noise>0 else lab
                sil = silhouette_score(Z, lab) if len(np.unique(vv))>=2 else np.nan
            except Exception:
                sil = np.nan
            row[f'{name}'] = f"ncl={ncl},noise={noise:.2f},sil={sil:.3f}" if not np.isnan(sil) else f"ncl={ncl},noise={noise:.2f},sil=nan"
            if name != 'KM':
                row[f'{name}@KM'] = round(adjusted_rand_score(lab_km, lab), 3)
        rows.append(row)
    out = pd.DataFrame(rows).set_index('ds')
    for c in out.columns: print(f"{c:<10}", end=" | ")
    print()
    for ds, r in out.iterrows():
        print(f"{ds:<10}", end="")
        for c in out.columns: print(f" {str(r[c]):<18}", end="")
        print()
    out.to_csv('analysis/dbscan_research2.csv')

main()
