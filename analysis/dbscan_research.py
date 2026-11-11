"""Research: where can density-based clustering (DBSCAN/HDBSCAN) improve the
KMeans/Spectral/GMM ensemble? Run on public 80% split, compare candidates.

Metrics (no labels available):
- ARI between candidate labelings (agreement proxy)
- silhouette of each candidate
- noise fraction for density methods
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
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

def eps_from_kdist(Z, k):
    """k-distance plot: sort NN distances at rank k, pick knee by max-curvature."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(Z).query(Z, k=k+1)
    kd = np.sort(d[:, k])
    # normalize to [0,1] box and find point farthest from the diagonal (kneedle-ish)
    x = np.linspace(0, 1, len(kd))
    y = (kd - kd[0]) / max(kd[-1]-kd[0], 1e-9)
    dist = y - x          # distance above the line from (0,0)->(1,1)
    i = int(np.argmax(dist))
    return float(kd[i])

def run():
    df = pd.read_csv('public_data/train.csv')
    rows = []
    for ds in sorted(df['dataset_id'].unique()):
        d = df[df['dataset_id']==ds]
        cols = [c for c in d.columns if c.startswith('x')]
        X = clean(d[cols].values)
        Z = reduce(X)
        n = len(Z); k0 = K_PRIOR[ds]

        # current ensemble winner (simplified: best of KM/Spectral/GMM at k0 by sil+consensus)
        cands = {}
        km = KMeans(n_clusters=k0, n_init=15, random_state=42).fit(Z)
        cands['KMeans'] = km.labels_
        try:
            sp = SpectralClustering(n_clusters=k0, affinity='nearest_neighbors',
                                    n_neighbors=min(n-1, max(10,int(np.sqrt(n)))),
                                    random_state=42).fit_predict(Z)
            if len(np.unique(sp))==k0: cands['Spectral'] = sp
        except Exception: pass
        try:
            gm = GaussianMixture(n_components=k0, covariance_type='full', n_init=3,
                                 random_state=42).fit(Z)
            lab = gm.predict(Z)
            if len(np.unique(lab))==k0: cands['GMM'] = lab
        except Exception: pass

        # DBSCAN with auto eps (k-distance knee), min_samples in {5,10}
        for kmin in (5, 10):
            eps = eps_from_kdist(Z, kmin)
            lab = DBSCAN(eps=eps, min_samples=kmin).fit_predict(Z)
            cands[f'DB{kmin}_e{eps:.2f}'] = lab

        # HDBSCAN: min_cluster_size scaled so ~k0 clusters are plausible
        mcs = max(15, n // (3*k0))
        try:
            h = hdbscan.HDBSCAN(min_cluster_size=max(10, n//(4*k0)),
                                cluster_selection_epsilon=0.0).fit(Z)
            lab = h.labels_
            cands[f'HDB_{n//(4*k0)}'] = lab
        except Exception as e:
            cands['HDB_err'] = np.zeros(n, int)

        # pairwise ARI vs KMeans (anchor) + own silhouette
        anchor = cands['KMeans']
        row = {'ds': ds, 'n': n, 'k0': k0, 'dim': Z.shape[1]}
        for name, lab in cands.items():
            u = np.unique(lab[lab>=0]) if (lab<0).any() else np.unique(lab)
            row[f'{name}_ari_kmeans'] = round(adjusted_rand_score(anchor, lab), 3)
            try:
                v = lab.copy(); 
                # silhouette needs >=2 clusters and no singleton-only
                vv = v[v>=0] if (v<0).any() else v
                row[f'{name}_sil'] = round(silhouette_score(Z, v), 3) if len(np.unique(vv))>=2 else np.nan
            except Exception:
                row[f'{name}_sil'] = np.nan
            row[f'{name}_noise'] = round(float((lab<0).mean()), 3)
            row[f'{name}_ncl'] = int(len(np.unique(lab[lab>=0])))
        rows.append(row)
    out = pd.DataFrame(rows).set_index('ds')
    cols = [c for c in out.columns if not c.endswith('_sil')]
    return out

if __name__ == '__main__':
    out = run()
    print(out.to_string())
    out.to_csv('analysis/dbscan_research.csv')
