"""Round 2: (a) outlier tail diagnostics via NN distances;
(b) robust DBSCAN config search anchored on k_prior; (c) does it beat current winner?"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
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
        c = X[:,j]; b = np.nan != c  # placeholder
        b = np.isnan(c)
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

def reassign_noise(Z, lab):
    """map -1 -> nearest cluster centroid (never emit -1)"""
    if not (lab < 0).any(): return lab.astype(int), 0.0
    out = lab.copy()
    uniq = np.unique(lab[lab>=0])
    cents = np.array([Z[lab==c].mean(0) for c in uniq])
    noise_idx = np.where(lab<0)[0]
    d2 = ((Z[noise_idx][:,None,:] - cents[None,:,:])**2).sum(-1)
    out[noise_idx] = uniq[np.argmin(d2, axis=1)]
    return out.astype(int), float(len(noise_idx)/len(lab))

df = pd.read_csv('public_data/train.csv')
print(f"{'ds':<6}{'k0':>3} {'d5_p99/p50':>11} | DBSCAN-robust: eps,ms,ncl,noise,sil@KM  | winner-vs-KM ARI")
for ds in sorted(df['dataset_id'].unique()):
    d = df[df['dataset_id']==ds]
    cols = [c for c in d.columns if c.startswith('x')]
    X = clean(d[cols].values); Z = reduce(X)
    n = len(Z); k0 = K_PRIOR[ds]

    # outlier tail: 5th NN distance quantiles
    d5, _ = cKDTree(Z).query(Z, k=6)
    d5 = d5[:,5]
    tail = f"{np.quantile(d5,0.99)/max(np.quantile(d5,0.5),1e-9):.2f}x"

    # current winner (KM/SP/GM at k0, sil+consensus)
    cands = {}
    km = KMeans(n_clusters=k0, n_init=15, random_state=42).fit(Z)
    cands['KM'] = km.labels_
    try:
        sp = SpectralClustering(n_clusters=k0, affinity='nearest_neighbors',
                                n_neighbors=min(n-1,max(10,int(np.sqrt(n)))),
                                random_state=42).fit_predict(Z)
        if len(np.unique(sp))==k0: cands['SP']=sp
    except Exception: pass
    try:
        gm = GaussianMixture(n_components=k0, covariance_type='full', n_init=3, random_state=42).fit(Z)
        lab = gm.predict(Z)
        if len(np.unique(lab))==k0: cands['GM']=lab
    except Exception: pass

    # robust DBSCAN search: eps grid from NN-distance percentiles x knee
    kdist5, _ = cKDTree(Z).query(Z, k=6)
    base = float(kdist5[:,5].mean())
    best = None
    for ms in (5, 10):
        for f in (0.8, 1.0, 1.5, 2.0, 3.0):
            eps = base * f
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)
            ncl = len(np.unique(lab[lab>=0])) if (lab<0).any() else len(np.unique(lab))
            noise = float((lab<0).mean())
            if ncl < 2 or noise > 0.15: continue
            lab_r, _ = reassign_noise(Z, lab)
            key = (abs(ncl-k0), -noise)
            if best is None or key < best[0]:
                best = (key, eps, ms, ncl, noise, lab_r)
    if best:
        _, eps, ms, ncl, noise, lab_db = best
        ari = round(adjusted_rand_score(cands['KM'], lab_db),3)
        sil = round(silhouette_score(Z, lab_db),3)
        # current winner by sil+consensus
        sc = np.array([silhouette_score(Z,c) for c in cands.values()]) + 0.3*np.array(
            [np.mean([normalized_mutual_info_score(a,b) for b in cands.values() if a is not b]) for a in cands.values()])
        wname = list(cands.keys())[int(np.argmax(sc))]
        print(f"{ds:<6}{k0:>3} {tail:>11} | eps={eps:.2f} ms={ms} ncl={ncl} noise={noise:.2f} sil={sil} ARI@KM={ari} | win={wname}")
    else:
        print(f"{ds:<6}{k0:>3} {tail:>11} | (no robust DBSCAN config found)")
