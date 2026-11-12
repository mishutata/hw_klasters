"""ds10: per-method seed stability at k in {2,3,4}, full + test-size samples.
Z-space built exactly like the pipeline (clean -> log-sym norm -> PCA95<=8)."""
import importlib.util
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

spec = importlib.util.spec_from_file_location("s6", "my_solution6/solution.py")
s6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s6)

df = pd.read_csv("public_data/train.csv").drop(columns=["dataset_id"])
ds10 = df[df["dataset_id"] == "ds10"].drop(columns=["dataset_id"]).values if "dataset_id" in df.columns else None
raw = pd.read_csv("public_data/train.csv")
print("cols:", list(raw.columns)[:8], "... n_ds10 =", (raw["dataset_id"]=="ds10").sum())
X = raw[raw["dataset_id"] == "ds10"].drop(columns=["id", "dataset_id"]).values.astype(float)

def stability(Xs, k):
    Z = s6._reduce(s6._clean(Xs), "log-sym")
    out = {}
    labs = {m: [] for m in ("KM", "GM", "SP")}
    seeds = (42, 7, 123)
    for s in seeds:
        labs["KM"].append(KMeans(n_clusters=k, n_init=15, random_state=s).fit(Z).labels_)
        try:
            g = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=s).fit(Z)
            labs["GM"].append(g.predict(Z))
        except Exception:
            labs["GM"].append(None)
        try:
            sp = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                    n_neighbors=min(len(Z)-1, max(10, int(np.sqrt(len(Z))))),
                                    random_state=s).fit_predict(Z)
            if len(np.unique(sp)) == k: labs["SP"].append(sp)
            else: labs["SP"].append(None)
        except Exception:
            labs["SP"].append(None)
    for m in ("KM", "GM", "SP"):
        L = [l for l in labs[m] if l is not None]
        if len(L) >= 2:
            out[m] = round(float(np.mean([adjusted_rand_score(L[i], L[j])
                                          for i, j in ((0,1),(0,2),(1,2))])), 3)
    return Z.shape[1], out

for tag, Xs in (("FULL", X),):
    n_test = int(len(X) * 0.25)
    print(f"\n== ds10 {tag} n={len(X)} (test~{n_test}) ==")
    for k in (2, 3, 4):
        d, st = stability(X, k)
        print(f"k={k} dim={d} seedARI: " + "  ".join(f"{m}={v}" for m, v in st.items()))
    rng = np.random.RandomState(42)
    for t in range(4):
        idx = rng.choice(len(X), n_test, replace=False)
        for k in (2, 3, 4):
            d, st = stability(X[idx], k)
            print(f"test#{t} n={n_test} k={k}: " + "  ".join(f"{m}={v}" for m, v in st.items()))
