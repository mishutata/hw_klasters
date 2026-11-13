# -*- coding: utf-8 -*-
"""Tier-1: end-to-end self-consistency of the v5 pipeline per dataset.
For each dataset: run the EXACT v5 pipeline on R random 25% slices of public
train (same size regime as the hidden test), compare each slice's partition to
the full-data partition restricted to that slice (ARI, permutation invariant).
High = v5's current choices are stable at test size -> keep. Low = unstable ->
the dataset is a likely ARI loser; look up alternatives in stability_results.csv.
Also scores v5's exact (norm,k) config with M1/M2 and completes the Spectral M2
grid so every candidate is comparable.
"""
import importlib.util, os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sol5", "my_solution5/solution.py")
sol5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol5)
sys.path.insert(0, os.path.join(ROOT, "dev_v14"))
from stability_analysis import (DATA, DS_LIST, NORMS, KS, R_SLICES, SLICE_FRAC,
                                SEED, slices_of, reduce_v5, m1_predictive,
                                m2_consistency, fit_full, sil)

df = pd.read_csv("public_data/train.csv")
XS = [c for c in df.columns if c.startswith("x")]

def v5_pipeline_stability(ds, X, r=R_SLICES):
    lab_full = sol5.cluster(X, ds)
    idxs = slices_of(len(X), r=r)
    aris, ks_used, winners = [], [], []
    for ix in idxs:
        Xi = X[ix]
        try:
            lab_i, info = sol5._run(Xi, ds)
            a = float(np.mean(lab_i == lab_full[ix]))  # raw agreement (sanity)
            from sklearn.metrics import adjusted_rand_score
            aris.append(adjusted_rand_score(lab_i, lab_full[ix]))
            ks_used.append(info['k_used']); winners.append(info['winner'])
        except Exception as e:
            aris.append(np.nan); ks_used.append(None); winners.append(None)
    return (float(np.mean([a for a in aris if a == a])) if any(a == a for a in aris) else np.nan,
            len([a for a in aris if a == a]), ks_used, winners)

def ring_blob_labeler(Xc):
    lab = sol5._ring_blob_labels(Xc)
    return None if lab is None else lab.astype(int)

def dbscan_v5_labeler(Z, k0):
    """Replicates v5's auto-DBSCAN branch exactly (deterministic given Z)."""
    from sklearn.cluster import DBSCAN
    n = len(Z)
    d5 = sol5._nn_dist(Z, 5 if n >= 200 else 3)
    best = None
    for eps in np.unique(np.quantile(d5, [0.60, 0.75, 0.90, 0.95])):
        for ms in (max(3, n // 100), 5):
            lab = DBSCAN(eps=float(eps), min_samples=ms).fit_predict(Z)
            ncl = len(np.unique(lab)) - (1 if (lab < 0).any() else 0)
            noise = float((lab < 0).mean())
            if 2 <= ncl <= max(3, k0 + 3) and noise <= 0.15:
                lab_r = sol5._reassign_noise(Z, lab)
                if best is None or sil(Z, lab_r) > sil(Z, best):
                    best = lab_r
    return best

rows = []
for ds in DS_LIST:
    X = DATA[ds]
    t0 = time.time()
    stab, n_ok, ks, wins = v5_pipeline_stability(ds, X)
    # v5's exact config scores
    k0 = sol5.K_PIN.get(ds, sol5.K_PRIOR.get(ds, 3))
    norm = sol5.NORM_PRIOR.get(ds, "minmax")
    Z = reduce_v5(X, norm)
    s1k, _ = m1_predictive(Z, k0, "KMeans")
    s1g, _ = m1_predictive(Z, k0, "GaussianMixture")
    s2sp = m2_consistency(Z, k0, "Spectral")
    extra = {}
    if ds in sol5.RING_DS:
        labf = ring_blob_labeler(X)
        if labf is not None:
            idxs = slices_of(len(X))
            aris = []
            for ix in idxs:
                li = ring_blob_labeler(X[ix])
                if li is None: continue
                aris.append(float(np.mean(li == labf[ix])))
            extra["ringblob_agree"] = float(np.mean(aris)) if aris else np.nan
    if ds == "ds10":
        labf = dbscan_v5_labeler(Z, k0)
        if labf is not None:
            idxs = slices_of(len(X))
            aris = []
            for ix in idxs:
                li = dbscan_v5_labeler(Z[ix], k0)
                if li is None: continue
                from sklearn.metrics import adjusted_rand_score
                aris.append(adjusted_rand_score(li, labf[ix]))
            extra["dbscan_m2"] = float(np.mean(aris)) if aris else np.nan
    rows.append(dict(ds=ds, v5_k=k0, v5_norm=norm, pipe_stab=stab, n_ok=n_ok,
                     M1_KMeans=s1k, M1_GMM=s1g, M2_Spectral=s2sp,
                     **extra, t=round(time.time() - t0, 1)))
    print(f"{ds}: pipe_stab={stab:.3f} (n_ok={n_ok}) KM={s1k:.3f} GMM={s1g:.3f} "
          f"SP={s2sp:.3f} {extra} [{time.time()-t0:.1f}s]", flush=True)

out = pd.DataFrame(rows)
out.to_csv("dev_v14/tier1_pipeline_stability.csv", index=False)
print("saved dev_v14/tier1_pipeline_stability.csv")
