"""Simulate grading protocol: stratified 80/20 split per dataset (by our
reference labels as proxy for true clusters), run v7 pipeline on the 20%
pseudo-test only, measure ARI vs reference restricted to those rows.
Low ARI_sim = fragile on small samples = likely low hidden test ARI."""
import importlib.util
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

spec = importlib.util.spec_from_file_location("sol", "my_solution10/solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

df = pd.read_csv("public_data/train.csv")
X_all, ref_all = {}, {}
for ds, g in df.groupby("dataset_id"):
    X = g[[c for c in g.columns if c.startswith("x")]].to_numpy(float)
    X_all[ds] = X
    ref_all[ds] = sol.cluster(X, ds)

if "--probe" not in sys.argv:
    rows = []
    for ds in sorted(X_all):
        X, ref = X_all[ds], ref_all[ds]
        rng = np.random.RandomState(42)
        tidx = []
        for c in np.unique(ref):
            idx = np.where(ref == c)[0]
            tidx.append(rng.choice(idx, size=max(1, int(round(len(idx) * 0.2))), replace=False))
        ti = np.sort(np.concatenate(tidx))
        T, refT = X[ti], ref[ti]
        base = sol.cluster(T, ds)
        rows.append({"ds": ds, "n_test": len(T),
                     "k_used_train": int(len(np.unique(ref))),
                     "ari_sim_base": round(adjusted_rand_score(refT, base), 3)})

    out = pd.DataFrame(rows).sort_values("ari_sim_base")
    out.to_csv("dev_v10/sim_test_eval.csv", index=False)
    print(out.to_string(index=False))
    print("\nmean ari_sim (v10):", round(out["ari_sim_base"].mean(), 3))

# ---------- probe: ds04 / ds10 (+ weak ring datasets) at test size ----------
if "--probe" in sys.argv:
    from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
    from sklearn.mixture import GaussianMixture
    PROBE = [a for a in sys.argv if a.startswith("ds")]
    for ds in PROBE:
        X, ref = X_all[ds], ref_all[ds]
        rng = np.random.RandomState(42)
        tidx = []
        for c in np.unique(ref):
            idx = np.where(ref == c)[0]
            tidx.append(rng.choice(idx, size=max(1, int(round(len(idx) * 0.2))), replace=False))
        ti = np.sort(np.concatenate(tidx))
        T, refT = X[ti], ref[ti]
        Zt = sol._reduce(T, sol.NORM_PRIOR.get(ds, "minmax"))
        print(f"\n=== {ds}  n_test={len(T)} ===")
        for k in (2, 3, 4):
            labs = {}
            km = [KMeans(n_clusters=k, n_init=15, random_state=s).fit(Zt).labels_
                  for s in (42, 7, 123)]
            stab_km = float(np.mean([adjusted_rand_score(km[i], km[j])
                                     for i, j in ((0, 1), (0, 2), (1, 2))]))
            labs["KMeans"] = (km[0], stab_km)
            try:
                g = GaussianMixture(n_components=k, covariance_type="full",
                                    n_init=3, random_state=42).fit(Zt)
                gl = [GaussianMixture(n_components=k, covariance_type="full",
                                      n_init=3, random_state=s).fit(Zt).predict(Zt)
                      for s in (42, 7)]
                sg = float(adjusted_rand_score(gl[0], gl[1]))
                labs["GMM"] = (gl[0], sg)
            except Exception:
                pass
            try:
                sp = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                        n_neighbors=max(10, int(np.sqrt(len(Zt)))),
                                        random_state=42).fit_predict(Zt)
                labs["Spectral"] = (sp, float("nan"))
            except Exception:
                pass
            for nm, (lab, st) in labs.items():
                if len(np.unique(lab)) == k:
                    print(f"  k={k} {nm:9s} ARI_vs_ref={adjusted_rand_score(refT, lab):.3f} "
                          f"stab={st:.3f}")
