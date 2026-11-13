# -*- coding: utf-8 -*-
"""Cross-slice stability analysis for the v14 model (base: proven v5, ARI 0.4846).

Idea
----
At inference time `cluster(X, dataset_id)` fits everything on the hidden TEST
points alone (~25% of each dataset's size, since test = 20% of an 80/20 split).
So the question per dataset is: "which (norm, dim, k, algorithm) recovers the
true partition when fitted on a small fresh sample?" We have no labels, but we
can measure transferability between independent samples from the same
distribution:

  M1 (predictive cross-slice ARI) - for KMeans / GaussianMixture:
      take R random 25% slices s_1..s_R of public train; fit the model on
      slice i, predict slice j (i != j); score = mean ARI(pred_j, selffit_j).
      High M1 = the partition found in one sample reproduces on another sample.

  M2 (subsample consistency) - for non-predictive methods (Spectral, ring/blob):
      fit on full train -> L_full; fit on each slice -> L_s;
      score = mean ARI(L_s, L_full restricted to s).

v5's actual per-dataset decision (k_used, winner) is obtained by running
my_solution5.solution.explain() and scored with the same metrics, so every
candidate is compared on an equal footing.
"""
import importlib.util, json, os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

spec = importlib.util.spec_from_file_location("sol5", "my_solution5/solution.py")
sol5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol5)

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

R_SLICES = 8
SLICE_FRAC = 0.25
NORMS = ["minmax", "log-sym", "robust", "standard"]
KS = [2, 3, 4, 5, 6]
DIMS_ALT = [6, 10]          # alternative PCA dims tested in stage B (besides v5-auto)
SEED = 42

# ---------------------------------------------------------------- data
df = pd.read_csv("public_data/train.csv")
XS = [c for c in df.columns if c.startswith("x")]
DATA = {}
for ds, g in df.groupby("dataset_id"):
    X = sol5._clean(g[XS].values)
    DATA[ds] = X
DS_LIST = sorted(DATA)

def slices_of(n, r=R_SLICES, frac=SLICE_FRAC):
    rng = np.random.RandomState(SEED)
    m = max(int(frac * n), 20)
    return [rng.choice(n, m, replace=False) for _ in range(r)]

def reduce_v5(X, norm):
    Z = sol5._apply_norm(X, norm)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    if X.shape[1] > 3:
        p = PCA(random_state=42).fit(Z)
        cum = np.cumsum(p.explained_variance_ratio_)
        ncomp = max(2, min(8, int(np.searchsorted(cum, 0.95) + 1), X.shape[1]))
        Z = PCA(n_components=ncomp, random_state=42).fit_transform(Z)
    return Z

def reduce_dim(X, norm, d):
    Z = sol5._apply_norm(X, norm)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    if X.shape[1] > 3:
        d = min(d, X.shape[1])
        Z = PCA(n_components=d, random_state=42).fit_transform(Z)
    return Z

def m1_predictive(Z, k, alg):
    """M1 for KMeans/GMM: fit on slice i, predict slice j, ARI vs self-fit of j."""
    n = len(Z); idxs = slices_of(n)
    fits = []
    ok = True
    for ix in idxs:
        Zs = Z[ix]
        try:
            if alg == "KMeans":
                m = KMeans(k, n_init=15, random_state=42).fit(Zs)
            else:
                m = GaussianMixture(k, covariance_type="full", n_init=3,
                                    random_state=42).fit(Zs)
                if len(np.unique(m.predict(Zs))) != k:
                    ok = False; break
        except Exception:
            ok = False; break
        fits.append((ix, m))
    if not ok:
        return np.nan, None
    aris = []
    for i, (ixi, mi) in enumerate(fits):
        for j, (ixj, mj) in enumerate(fits):
            if i == j: continue
            try:
                lab_j = mj.predict(Z[ixj])
            except Exception:
                ok = False; break
            aris.append(adjusted_rand_score(mi.predict(Z[ixj]), lab_j))
        if not ok: return np.nan, None
    return float(np.mean(aris)), idxs

def m2_consistency(Z, k, alg):
    """M2 for Spectral / ring-blob-like deterministic labelers."""
    n = len(Z); idxs = slices_of(n)
    def fit(z):
        if alg == "Spectral":
            nn = min(len(z) - 1, max(10, int(np.sqrt(len(z)))))
            lab = SpectralClustering(k, affinity="nearest_neighbors",
                                     n_neighbors=nn, random_state=42).fit_predict(z)
        else:
            raise ValueError(alg)
        return lab
    try:
        Lfull = fit(Z)
    except Exception:
        return np.nan
    aris = []
    for ix in idxs:
        try:
            Ls = fit(Z[ix])
        except Exception:
            continue
        a, b = Ls, Lfull[ix]
        if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
            aris.append(0.0); continue
        aris.append(adjusted_rand_score(a, b))
    return float(np.mean(aris)) if aris else np.nan

def sil(Z, lab):
    u = np.unique(lab)
    if len(u) < 2 or len(u) >= len(Z): return -1.0
    try: return float(silhouette_score(Z, lab))
    except Exception: return -1.0

def fit_full(Z, k, alg):
    if alg == "KMeans":
        return KMeans(k, n_init=15, random_state=42).fit(Z).labels_
    if alg == "GaussianMixture":
        try:
            m = GaussianMixture(k, covariance_type="full", n_init=3,
                                random_state=42).fit(Z)
            lab = m.predict(Z)
            return lab if len(np.unique(lab)) == k else None
        except Exception:
            return None
    if alg == "Spectral":
        try:
            nn = min(len(Z) - 1, max(10, int(np.sqrt(len(Z)))))
            return SpectralClustering(k, affinity="nearest_neighbors",
                                      n_neighbors=nn, random_state=42).fit_predict(Z)
        except Exception:
            return None
    return None

def analyze_dataset(ds, X, stage="AB"):
    """Return list of candidate dicts for one dataset."""
    rows = []
    Zcache = {}
    def getZ(norm, dim):
        key = (norm, dim)
        if key not in Zcache:
            Zcache[key] = reduce_v5(X, norm) if dim == "auto" else reduce_dim(X, norm, dim)
        return Zcache[key]

    # ---- stage A: all norms x k, KMeans+GMM M1 at v5-auto dim
    topA = []
    for norm in NORMS:
        Z = getZ(norm, "auto")
        for k in KS:
            for alg in ("KMeans", "GaussianMixture"):
                t0 = time.time()
                s1, _ = m1_predictive(Z, k, alg)
                labf = fit_full(Z, k, alg)
                sf = sil(Z, labf) if labf is not None else np.nan
                row = dict(ds=ds, norm=norm, dim="auto", k=k, alg=alg,
                           M1=s1, sil_full=sf, t=round(time.time() - t0, 2))
                rows.append(row)
                topA.append((s1 if s1 == s1 else -9, row))
    topA = [r for s, r in sorted(topA, key=lambda t: -t[0]) if s > -9]

    # ---- stage B: spectral M2 + alt dims for the best (norm,k) combos
    seen = set()
    stageB = []
    for r in topA[:6]:                      # top-6 (norm,k,alg) by KMeans/GMM M1
        key = (r["norm"], r["k"])
        if key in seen: continue
        seen.add(key); stageB.append(key)
    for norm, k in stageB:
        for dim in ("auto",) + tuple(DIMS_ALT):
            Z = getZ(norm, dim)
            s2 = m2_consistency(Z, k, "Spectral")
            labf = fit_full(Z, k, "Spectral")
            rows.append(dict(ds=ds, norm=norm, dim=str(dim), k=k, alg="Spectral",
                             M1=np.nan, M2=s2,
                             sil_full=(sil(Z, labf) if labf is not None else np.nan)))
            # re-score KMeans/GMM at this dim too (dim can change their ranking)
            for alg in ("KMeans", "GaussianMixture"):
                s1, _ = m1_predictive(Z, k, alg)
                labf2 = fit_full(Z, k, alg)
                rows.append(dict(ds=ds, norm=norm, dim=str(dim), k=k, alg=alg,
                                 M1=s1,
                                 sil_full=(sil(Z, labf2) if labf2 is not None else np.nan)))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    t0 = time.time()
    os.makedirs("dev_v14", exist_ok=True)
    csv_path = "dev_v14/stability_results.csv"
    prev = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    done = set(prev["ds"].unique()) if prev is not None else set()
    frames = [prev] if prev is not None else []
    for ds in DS_LIST:
        if only and ds != only: continue
        if ds in done:
            print(f"{ds}: already done, skip", flush=True)
            continue
        X = DATA[ds]
        t1 = time.time()
        fr = analyze_dataset(ds, X)
        frames.append(fr)
        pd.concat(frames).to_csv(csv_path, index=False)   # incremental save
        print(f"{ds}: {len(fr)} candidates in {time.time()-t1:.1f}s", flush=True)
    print(f"TOTAL {time.time()-t0:.1f}s -> dev_v14/stability_results.csv") 
