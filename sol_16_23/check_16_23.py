"""Check ds16 & ds23: are they 2 clusters (ellipse/ring + central blob)?

Runs my_solver/solution.py pipeline on the public train data, draws the
k=2 result and quantifies how good it is:
  - silhouette of k=2 vs k=3,4
  - NMI between solver labels and a radius-based reference split
    (blob = innermost points by distance to dataset centre)
Saves PNGs into this folder.
"""
import sys, os
sys.path.insert(0, r"..\my_solver")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score

from solution import _clean, _reduce, cluster, explain

df = pd.read_csv(r"..\public_data\train.csv")
plt.rcParams.update({"figure.dpi": 110})

for ds in ("ds16", "ds23"):
    X = df[df.dataset_id == ds].drop(columns=["id", "dataset_id"]).values.astype(float)
    n = len(X)
    Z2 = _reduce(X)                       # PCA space used by the solver
    p2 = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(_clean(X)))

    rep = explain(X, ds)
    lab = rep["labels"]
    sils = {k: round(float(silhouette_score(Z2, KMeans(k, n_init=15, random_state=42).fit(Z2).labels_)), 3)
            for k in (2, 3, 4)}

    # radius-based reference: blob = innermost 30% by dist to centre
    r = np.linalg.norm(p2 - p2.mean(0), axis=1)
    thr = np.quantile(r, 0.30)
    ref = (r <= thr).astype(int)          # 1 = central blob

    nmi = normalized_mutual_info_score(ref, lab)
    sizes = np.bincount(lab)
    frac_blob_per_cluster = [round((ref[lab == c] == 1).mean(), 3) for c in range(len(sizes))]

    print(f"== {ds}: n={n}  k_used={rep['k_used']} winner={rep['winner']} sil_by_k={sils}")
    print(f"   cluster sizes={sizes}, blob-fraction per cluster={frac_blob_per_cluster}, NMI vs radius-ref={nmi:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    sc = ax[0].scatter(p2[:, 0], p2[:, 1], c=lab, cmap="viridis", s=4, alpha=0.7)
    ax[0].set_title(f"{ds} solver labels (k={rep['k_used']}, winner {rep['winner']})\nsil k=2/3/4 = {sils[2]}/{sils[3]}/{sils[4]}")
    ax[1].scatter(p2[:, 0], p2[:, 1], c=np.where(ref == 1, 1, 0), cmap="viridis", s=4, alpha=0.7)
    ax[1].set_title(f"radius reference: inner 30% = blob\nNMI(solver, ref) = {nmi:.3f}")
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(f"{ds}_check.png", bbox_inches="tight")
    plt.close(fig)

    # zoom into the centre to see the blob clearly
    c = p2.mean(0, keepdims=True)
    d = np.linalg.norm(p2 - c, axis=1)
    m = d <= np.quantile(d, 0.55)
    fig, a = plt.subplots(figsize=(6, 5.4))
    a.scatter(p2[m, 0], p2[m, 1], c=lab[m], cmap="viridis", s=6, alpha=0.8)
    a.set_title(f"{ds} centre zoom (inner 55% of points)")
    fig.tight_layout()
    fig.savefig(f"{ds}_center_zoom.png", bbox_inches="tight")
    plt.close(fig)
print("saved to sol_16_23/")
