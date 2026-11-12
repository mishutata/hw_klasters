"""v11 (rewritten): unsupervised k-evidence for all 24 datasets. No label proxy, no self-referential ARI."""
import sys, importlib.util
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

spec = importlib.util.spec_from_file_location("s6", "my_solution6/solution.py")
s6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s6)
raw = pd.read_csv("public_data/train.csv")

out = []
for ds in sorted(raw["dataset_id"].unique()):
    X = raw[raw["dataset_id"] == ds].drop(columns=["id", "dataset_id"]).values.astype(float)
    n = len(X); prior = s6.K_PRIOR.get(ds, 2)
    Z = s6._reduce(s6._clean(X), s6.NORM_PRIOR.get(ds, "std"))
    ks = [k for k in range(2, min(7, max(3, n // 40)) + 1)]
    sils = {}
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z).labels_
        sils[k] = round(float(silhouette_score(Z, lab)), 3)
    bics = {}
    for k in ks:
        try:
            gm = GaussianMixture(n_components=k, covariance_type="full", n_init=2, random_state=42).fit(Z)
            bics[k] = round(float(gm.bic(Z)), 0)
        except Exception:
            bics[k] = None
    # bootstrap stability at prior and neighbors: 3x 60% subsamples, KMeans each, pairwise ARI on row intersection
    stabs = {}
    for k in [x for x in (prior - 1, prior, prior + 1) if x >= 2]:
        pairs = []
        sub = {}
        for s in (42, 7, 123):
            idx = np.sort(np.random.RandomState(s).choice(n, size=int(0.6 * n), replace=False))
            sub[s] = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Z[idx]).labels_
        for i, si in enumerate((42, 7, 123)):
            for sj in (42, 7, 123)[i + 1:]:
                inter = np.intersect1d(np.sort(np.random.RandomState(si).choice(n, size=int(0.6 * n), replace=False)),
                                       np.sort(np.random.RandomState(sj).choice(n, size=int(0.6 * n), replace=False)))
                a1 = {r: l for r, l in zip(np.sort(np.random.RandomState(si).choice(n, size=int(0.6 * n), replace=False)), sub[si])}
                a2 = {r: l for r, l in zip(np.sort(np.random.RandomState(sj).choice(n, size=int(0.6 * n), replace=False)), sub[sj])}
                pairs.append(adjusted_rand_score([a1[r] for r in inter], [a2[r] for r in inter]))
        stabs[k] = round(float(np.mean(pairs)), 3) if pairs else None
    # eigengap on kNN graph
    gap_k, gaps = None, []
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=15).fit(Z)
        dist, ind = nn.kneighbors(Z)
        g = np.zeros((n, n))
        for i in range(n):
            g[i, ind[i]] = 1.0 / (dist[i] + 1e-9)
        g[np.diag_indices_from(g)] = 0.0
        L = laplacian(g, normed=True)
        kmax = min(prior + 3, n - 2)
        w = np.sort(eigsh(L, k=kmax, which="SM", return_eigenvectors=False))
        gaps = [round(float(x), 4) for x in np.diff(w)]
        gap_k = int(np.argmax(gaps)) + 1
    except Exception:
        pass
    # cross-algorithm agreement at prior k
    km = KMeans(n_clusters=prior, n_init=15, random_state=42).fit(Z).labels_
    agree = {}
    try:
        gm = GaussianMixture(n_components=prior, covariance_type="full", n_init=3, random_state=42).fit(Z)
        agree["KM-GM"] = round(float(normalized_mutual_info_score(km, gm.predict(Z))), 3)
    except Exception:
        pass
    try:
        sp = SpectralClustering(n_clusters=prior, affinity="nearest_neighbors",
                                n_neighbors=min(n - 1, max(10, int(np.sqrt(n)))), random_state=42).fit_predict(Z)
        agree["KM-SP"] = round(float(normalized_mutual_info_score(km, sp)), 3)
    except Exception:
        pass
    row = {"ds": ds, "n": n, "prior": prior, "sil": sils,
           "bic_min_k": min(bics, key=lambda k: (bics[k] is None, bics[k])) if any(v is not None for v in bics.values()) else None,
           "stab": stabs, "gap_k": gap_k, "agree": agree}
    out.append(row)
    print(f"{ds}: n={n} prior={prior} sil={sils} bicmin={row['bic_min_k']} stab={stabs} gapk={gap_k} agree={agree}", flush=True)

pd.DataFrame(out).to_json("dev_v11/k_evidence_table.json", orient="records", indent=1)
print("saved dev_v11/k_evidence_table.json")
