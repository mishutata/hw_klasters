"""Probe ds13/ds14 at test size WITH the pipeline's real PCA cap (14):
reference-free silhouette + cross-sample-size stability per algorithm."""
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

def load(p):
    spec = importlib.util.spec_from_file_location("m", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

v9, v10 = load("my_solution9/solution.py"), load("my_solution10/solution.py")
df = pd.read_csv("public_data/train.csv")

def fit_alg(nm, Z):
    if nm == "KMeans":
        return KMeans(n_clusters=2, n_init=15, random_state=42).fit(Z).labels_
    if nm == "GMM":
        return GaussianMixture(n_components=2, covariance_type="full",
                               n_init=3, random_state=42).fit(Z).predict(Z)
    return SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                              n_neighbors=max(10, int(np.sqrt(len(Z)))),
                              random_state=42).fit_predict(Z)

out = []
for ds in ("ds13", "ds14"):
    g = df[df["dataset_id"] == ds]
    X = g[[c for c in g.columns if c.startswith("x")]].to_numpy(float)
    rng = np.random.RandomState(42)
    ref9 = v9.cluster(X, ds)  # v9 full-train winner labels
    tidx = []
    for c in np.unique(ref9):
        idx = np.where(ref9 == c)[0]
        tidx.append(rng.choice(idx, size=max(1, int(round(len(idx) * 0.2))), replace=False))
    ti = np.sort(np.concatenate(tidx))
    T = X[ti]
    cap = v10.PCA_CAP.get(ds, 8)
    Zt = v10._reduce(T, v10.NORM_PRIOR.get(ds, "minmax"), cap)
    Zf = v10._reduce(X, v10.NORM_PRIOR.get(ds, "minmax"), cap)
    out.append(f"===== {ds} (PCA dims test={Zt.shape[1]} full={Zf.shape[1]}) =====")
    for nm in ("KMeans", "GMM", "Spectral"):
        lab_t = fit_alg(nm, Zt)
        lab_f = fit_alg(nm, Zf)
        sil = silhouette_score(Zt, lab_t)
        a_same = adjusted_rand_score(lab_f[ti], lab_t)
        a_v9ref = adjusted_rand_score(ref9[ti], lab_t)
        out.append(f"  {nm:9s} sil_test={sil:.3f} samealg_full_vs_test={a_same:.3f} vs_v9full_ref={a_v9ref:.3f}")
    a_kg = adjusted_rand_score(fit_alg("KMeans", Zt), fit_alg("GMM", Zt))
    a_ks = adjusted_rand_score(fit_alg("KMeans", Zt), fit_alg("Spectral", Zt))
    a_gs = adjusted_rand_score(fit_alg("GMM", Zt), fit_alg("Spectral", Zt))
    out.append(f"  pairwise test: KM-GM {a_kg:.3f} KM-SP {a_ks:.3f} GM-SP {a_gs:.3f}")
print(chr(10).join(out))
