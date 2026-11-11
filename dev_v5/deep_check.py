# -*- coding: utf-8 -*-
"""Arbitrate visual read vs BIC/stability for ds09 (2 vs 4) and ds15 (4 vs 5).
Uses the exact v3 pipeline (clean + reduce) from analysis2/v3_solution.py."""
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sol3", "analysis2/v3_solution.py")
sol3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol3)

df = pd.read_csv("public_data/train.csv")

for ds, ks in (("ds09", [2, 3, 4]), ("ds15", [3, 4, 5])):
    Xs = df[df["dataset_id"] == ds].drop(columns=["id", "dataset_id"])
    Xc = sol3._clean(Xs.to_numpy(float))
    Z = sol3._reduce(Xc, sol3.NORM_PRIOR.get(ds, "minmax"))
    print(f"== {ds} n={len(Z)} dims={Z.shape[1]} norm={sol3.NORM_PRIOR.get(ds,'minmax')}")
    for mcs in (10, 20, 30):
        lab = HDBSCAN(min_cluster_size=mcs).fit_predict(Z)
        nc = len(set(lab)) - (1 if -1 in set(lab) else 0)
        print(f"   HDBSCAN mcs={mcs}: {nc} clusters, noise={(lab == -1).sum()}")
    for k in ks:
        labels = [KMeans(k, n_init=1, random_state=i).fit(Z).labels_ for i in range(8)]
        init_stab = np.mean([adjusted_rand_score(a, b)
                             for a in labels for b in labels if id(a) != id(b)])
        gmm = GaussianMixture(k, covariance_type="full", n_init=3,
                              random_state=42).fit(Z)
        print(f"   k={k}: init-stab={init_stab:.3f} sil={silhouette_score(Z, labels[0]):.3f}")
        print(f"        GMM BIC={gmm.bic(Z):.1f}")
    # figure: density contours + KMeans at the two candidate k
    fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4))
    if isinstance(axes, np.ndarray) is False:
        axes = [axes]
    ax0 = axes[0]
    for ax, k in zip(axes, ks):
        km = KMeans(k, n_init=10, random_state=42).fit(Z)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=km.labels_, cmap="tab10",
                        s=6, alpha=0.8, edgecolor="none")
        ax.set_title(f"{ds} k={k}")
    fig.tight_layout()
    fn = f"dev_v5/deep_{ds}.png"
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    print("   saved", fn)
