"""v10 pinning analysis (improvement plan A+B+C).

For every NON-pinned dataset of the v8 pipeline:
  - full-size evidence for k=2..6: silhouette, KMeans seed-stability (5 seeds),
    GMM BIC (full/tied/diag covariance);
  - bootstrap stability: NBOOT resamples of train, mean pairwise ARI between
    refits per (k, algorithm)  [method A/B];
  - test-size check: fit on a ~200-row subsample, ARI vs the full-data
    partition of the same (k, algorithm);
  - recommendation = argmax over (k in top-sil set + prior k, alg in {KM,GMM})
    of boot_stability + 0.25*sil - 0.01*max(0, BIC_k - best_BIC)  [method C].
Outputs: dev_v8/pin_analysis.csv, dev_v8/pin_check_08_13_14.png, compact table.
"""
import importlib.util
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

spec = importlib.util.spec_from_file_location("sol8", "my_solution8/solution.py")
sol8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol8)

data = pd.read_csv("public_data/train.csv")
rng = np.random.RandomState(42)
NBOOT, TEST_N = 30, 200
SEEDS5 = (42, 7, 123, 999, 2024)

PINNED = set(sol8.K_PIN)
DS = sorted(set(data["dataset_id"]) - PINNED)

def sil(Z, lab):
    u = np.unique(lab)
    if len(u) < 2 or len(u) >= len(Z):
        return -1.0
    try:
        return float(silhouette_score(Z, lab))
    except Exception:
        return -1.0

rows, recs = [], {}
Zstore, Xstore = {}, {}
for ds in DS:
    X = (data[data["dataset_id"] == ds]
     .drop(columns=[c for c in ("id", "dataset_id") if c in data.columns])
     .values.astype(float))
    Xc = sol8._clean(X)
    norm = sol8.NORM_PRIOR.get(ds, "minmax")
    Z = sol8._reduce(Xc, norm)
    n = len(Z)
    Zstore[ds], Xstore[ds] = Z, X
    k0 = sol8.K_PRIOR.get(ds, 3)

    # --- full-size evidence per k ---
    ev = {}
    for k in range(2, 7):
        labs = [KMeans(n_clusters=k, n_init=3, random_state=s).fit(Z).labels_
                for s in SEEDS5]
        stab_km = float(np.mean([adjusted_rand_score(labs[i], labs[j])
                                 for i, j in combinations(range(5), 2)]))
        gm_f = GaussianMixture(n_components=k, covariance_type="full",
                               n_init=10, random_state=42).fit(Z)
        gm_t = GaussianMixture(n_components=k, covariance_type="tied",
                               n_init=5, random_state=42).fit(Z)
        gm_d = GaussianMixture(n_components=k, covariance_type="diag",
                               n_init=5, random_state=42).fit(Z)
        ev[k] = dict(sil=sil(Z, labs[0]), stab_km=stab_km,
                     bic=float(gm_f.bic(Z)), bic_tied=float(gm_t.bic(Z)),
                     bic_diag=float(gm_d.bic(Z)))

    # --- bootstrap stability for top-k set (sil top-3 + prior k) ---
    topks = sorted(set(sorted(range(2, 7), key=lambda k: -ev[k]["sil"])[:3]) | {k0})
    # bootstrap stability: refit on resample, PREDICT ALL points, ARI vs the
    # full-data partition of the same (k, alg) -- items are identical here.
    boot = {}
    idxs = [rng.choice(n, n, replace=True) for _ in range(NBOOT)]
    for k in topks:
        ref_km = KMeans(n_clusters=k, n_init=15, random_state=42).fit(Z).labels_
        ref_gm = (GaussianMixture(n_components=k, covariance_type="full",
                                  n_init=10, random_state=42).fit(Z).predict(Z))
        a_km, a_gm = [], []
        for ix in idxs:
            Zb = Z[ix]
            a_km.append(adjusted_rand_score(
                KMeans(n_clusters=k, n_init=1, random_state=42).fit(Zb).predict(Z), ref_km))
            a_gm.append(adjusted_rand_score(
                GaussianMixture(n_components=k, covariance_type="full",
                                n_init=1, random_state=42).fit(Zb).predict(Z), ref_gm))
        boot[(k, "KMeans")] = float(np.mean(a_km))
        boot[(k, "GaussianMixture")] = float(np.mean(a_gm))

    # --- test-size check: does the partition hold at n~200? ---
    def test_ari(k, alg):
        sub = rng.choice(n, TEST_N, replace=False)
        Zs = Z[sub]
        if alg == "KMeans":
            lab = KMeans(n_clusters=k, n_init=15, random_state=42).fit_predict(Zs)
            ref = KMeans(n_clusters=k, n_init=15, random_state=42).fit(Z).labels_[sub]
        elif alg == "SpectralClustering":
            lab = (SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                      n_neighbors=min(len(Zs) - 1,
                                                      max(10, int(np.sqrt(len(Zs))))),
                                      random_state=42).fit_predict(Zs))
            ref = (SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                      n_neighbors=min(n - 1, max(10, int(np.sqrt(n)))),
                                      random_state=42).fit_predict(Z)[sub])
        else:
            lab = (GaussianMixture(n_components=k, covariance_type="full",
                                   n_init=3, random_state=42).fit(Zs).predict(Zs))
            ref = (GaussianMixture(n_components=k, covariance_type="full",
                                   n_init=10, random_state=42).fit(Z).predict(Z)[sub])
        return float(adjusted_rand_score(lab, ref))

    # --- current v8 behaviour on this ds (train size) ---
    cur_lab = sol8.cluster(X, ds)
    cur_info = sol8.explain(X, ds)
    cur_k, cur_win = int(cur_info["k_used"]), cur_info["winner"]

    # --- recommendation ---
    best_bic = min(ev[k]["bic"] for k in topks)
    cands = []
    for k in topks:
        for alg in ("KMeans", "GaussianMixture"):
            s = boot[(k, alg)] + 0.25 * ev[k]["sil"] \
                - 0.01 * max(0.0, ev[k]["bic"] - best_bic)
            cands.append((s, k, alg))
    cands.sort(reverse=True)
    s_rec, k_rec, alg_rec = cands[0]
    t_rec = test_ari(k_rec, alg_rec)
    t_cur = test_ari(cur_k, cur_win) if cur_win in ("KMeans", "GaussianMixture",
                                                    "SpectralClustering") else float("nan")
    recs[ds] = dict(k=cur_k, win=cur_win, k_rec=k_rec, alg_rec=alg_rec,
                    s_rec=round(s_rec, 3), t_cur=t_cur, t_rec=t_rec)
    for k in topks:
        for alg in ("KMeans", "GaussianMixture"):
            rows.append(dict(ds=ds, n=n, k=k, alg=alg,
                             sil=round(ev[k]["sil"], 3),
                             stab_full=round(ev[k]["stab_km"], 3) if alg == "KMeans" else np.nan,
                             boot=round(boot[(k, alg)], 3),
                             bic=round(ev[k]["bic"], 1)))
    print(f"{ds} n={n} cur=k{cur_k}/{cur_win}(tAri={t_cur:.2f}) | " +
          " ".join(f"k{k}:{boot[(k,'KMeans')]:.2f}/{boot[(k,'GaussianMixture')]:.2f}"
                   f"(sil{ev[k]['sil']:.2f})" for k in topks) +
          f" | REC k={k_rec} {alg_rec} s={s_rec:.3f} tAri={t_rec:.2f}")

pd.DataFrame(rows).to_csv("dev_v8/pin_analysis.csv", index=False)

# --- visual check for ds08/13/14: current vs recommended partition ---
fig, axes = plt.subplots(3, 2, figsize=(10, 13))
for r, ds in enumerate(("ds08", "ds13", "ds14")):
    Z, X = Zstore[ds], Xstore[ds]
    cur_lab = sol8.cluster(X, ds)
    rc = recs[ds]
    if rc["alg_rec"] == "KMeans":
        rec_lab = KMeans(n_clusters=rc["k_rec"], n_init=15,
                         random_state=42).fit(Z).labels_
    else:
        rec_lab = (GaussianMixture(n_components=rc["k_rec"], covariance_type="full",
                                   n_init=10, random_state=42).fit(Z).predict(Z))
    axes[r, 0].scatter(Z[:, 0], Z[:, 1], c=cur_lab, s=6)
    axes[r, 0].set_title(f"{ds} CURRENT k={rc['k']} {rc['win']}")
    axes[r, 1].scatter(Z[:, 0], Z[:, 1], c=rec_lab, s=6)
    axes[r, 1].set_title(f"{ds} RECOMMENDED k={rc['k_rec']} {rc['alg_rec']}")
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig("dev_v8/pin_check_08_13_14.png", dpi=90)
print("saved dev_v8/pin_analysis.csv + pin_check_08_13_14.png")
