"""Does ds09 look like k=2? Candidates (KMeans/GMM/Spectral) at k=2 and k=3,
silhouette + cross-seed stability, side-by-side figure."""
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

spec = importlib.util.spec_from_file_location("sol9", "my_solution9/solution.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

X = pd.read_csv("public_data/train.csv")
df = X[X.dataset_id == "ds09"]
Xc = M._clean(df.drop(columns=["dataset_id", "id"] if "id" in df else ["dataset_id"]))
Z = M._reduce(Xc, M.NORM_PRIOR.get("ds09", "minmax"), M.PCA_CAP.get("ds09", 8))
n = len(Z)
print(f"ds09: n={n}, Z shape {Z.shape}")

PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for k in (2, 3):
    print(f"\n=== k={k} ===")
    cands = {}
    km_seeds = [KMeans(n_clusters=k, n_init=15, random_state=s).fit(Z) for s in (42, 7, 123)]
    stab_km = float(np.mean([adjusted_rand_score(km_seeds[i].labels_, km_seeds[j].labels_)
                             for i, j in ((0, 1), (0, 2), (1, 2))]))
    cands["KMeans"] = km_seeds[0].labels_
    try:
        gm = GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                             random_state=42).fit(Z)
        lab = gm.predict(Z)
        if len(np.unique(lab)) == k:
            cands["GMM"] = lab
    except Exception as e:
        print("  GMM failed:", e)
    try:
        sp = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                n_neighbors=min(n - 1, max(10, int(np.sqrt(n)))),
                                random_state=42).fit_predict(Z)
        if len(np.unique(sp)) == k:
            cands["Spectral"] = sp
    except Exception as e:
        print("  Spectral failed:", e)
    for nm, lab in cands.items():
        sil = silhouette_score(Z, lab)
        sizes = np.bincount(lab)
        extra = f" km_stab={stab_km:.3f}" if nm == "KMeans" else ""
        print(f"  {nm:9s} sil={sil:+.4f} sizes={list(sizes)}{extra}")

# side-by-side figure: k=2 candidates vs current k=3 winner (Spectral, from v9)
fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
_r = M.cluster(Xc, "ds09")
labs9 = _r[0] if isinstance(_r, tuple) else _r   # current v9 labels (k=3 Spectral)
title9 = f"v9 k=3 winner | sizes={np.bincount(labs9).tolist()}"
panels = [("KMeans k=2", KMeans(n_clusters=2, n_init=15, random_state=42).fit(Z).labels_),
          ("GMM k=2", GaussianMixture(n_components=2, covariance_type="full",
                                      n_init=3, random_state=42).fit(Z).predict(Z)),
          ("Spectral k=2", SpectralClustering(
              n_clusters=2, affinity="nearest_neighbors",
              n_neighbors=min(n - 1, max(10, int(np.sqrt(n)))),
              random_state=42).fit_predict(Z)),
          (title9, labs9)]
# pick two informative projections: x4-x5 style from pair plot -> use Z dims with most spread
spreads = np.std(Z, axis=0)
top = np.argsort(spreads)[::-1][:3]
projs = [(top[0], top[1]), (top[0], top[2])]
for ax, (nm, lab) in zip(axes, panels):
    a, b = projs[0]
    ax.scatter(Z[:, a], Z[:, b], c=lab, cmap="tab10", s=4, alpha=0.7)
    ax.set_title(f"{nm}\nsil={silhouette_score(Z, lab):+.3f}", fontsize=9)
axes[4].axis("off")
axes[4].text(0.02, 0.5, "Top-3 spread dims: " + ", ".join(f"z{d}" for d in top),
             fontsize=10, va="center")
plt.suptitle("ds09: k=2 candidates vs current v9 (k=3)")
plt.tight_layout()
plt.savefig("analysis2/ds09_k2_vs_k3.png", dpi=110)
print("\nsaved analysis2/ds09_k2_vs_k3.png")
