"""ds16 & ds23: is the right answer {ellipse/ring, central blob}?  (k=2)

For each dataset:
  1. locate the ring boundary with an ellipse-aware radial profile
     (per-angle median radius of the outer shell, circularly smoothed);
  2. reference labels: blob = inside the boundary, ring = on/outside it;
  3. score every candidate k=2 clustering against that reference (ARI/NMI)
     and by silhouette in the solver's reduced space:
        - KMeans / Spectral / GMM       (what solution.py currently runs)
        - KMeans on [Z, radius]         (radius-aware variant)
        - radius split itself           (oracle for this geometry)
  4. stability of the reference split across bootstrap resamples;
  5. save figures into this folder.
"""
import sys
sys.path.insert(0, r"..\my_solver")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from solution import _clean, _reduce, cluster

df = pd.read_csv(r"..\public_data\train.csv")
plt.rcParams.update({"figure.dpi": 110})


def ellipse_boundary(p):
    """Ellipse-aware ring boundary: per-angle median radius of the outer shell."""
    c0 = p.mean(0)
    d0 = np.linalg.norm(p - c0, axis=1)
    outer = d0 >= np.quantile(d0, 0.45)          # keep the outer shell only
    ang = np.arctan2(p[:, 1] - c0[1], p[:, 0] - c0[0])
    nb = 36
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    Rb = np.full(nb, np.nan)
    for i, (e, f) in enumerate(zip(edges[:-1], edges[1:])):
        m = (ang >= e) & (ang < f) & outer
        if m.any():
            Rb[i] = np.median(d0[m])             # median = robust to blob leaks
    for i in range(nb):                          # fill any empty bins
        if np.isnan(Rb[i]):
            j = (i + 1) % nb
            while np.isnan(Rb[j]):
                j = (j + 1) % nb
            Rb[i] = Rb[j]
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    sm = np.array([sum(k[j] * Rb[(i - j) % nb] for j in range(5)) for i in range(nb)])
    th = (edges[:-1] + edges[1:]) / 2
    return c0, th, sm


def inside_boundary(p, c, th, R, frac=0.80):
    """True if point is clearly inside the fitted boundary."""
    d = np.linalg.norm(p - c, axis=1)
    a = (np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]) + np.pi) % (2 * np.pi) - np.pi
    t_ext = np.concatenate([th - 2*np.pi, th, th + 2*np.pi])
    R_ext = np.concatenate([R, R, R])
    Rloc = np.interp(a, t_ext, R_ext)
    return d < frac * Rloc


for ds in ("ds16", "ds23"):
    X = df[df.dataset_id == ds].drop(columns=["id", "dataset_id"]).values.astype(float)
    n = len(X)
    Zr = _reduce(X)                                   # solver space (k=2 chosen there)
    sc = StandardScaler().fit_transform(_clean(X))
    p2 = PCA(n_components=2, random_state=42).fit_transform(sc)   # draw space

    c, th, R = ellipse_boundary(p2)
    ref = inside_boundary(p2, c, th, R).astype(int)  # 1=blob, 0=ring
    frac_blob = ref.mean()
    print(f"== {ds}: n={n}  centre=({c[0]:.2f},{c[1]:.2f})  blob fraction={frac_blob:.3f}")

    # --- candidates (labels in solver space) ---
    km = KMeans(2, n_init=15, random_state=42).fit(Zr).labels_
    sp = SpectralClustering(2, affinity="nearest_neighbors",
                            n_neighbors=max(10, int(np.sqrt(n))),
                            random_state=42).fit_predict(Zr)
    gm = GaussianMixture(2, covariance_type="full", n_init=3,
                         random_state=42).fit(Zr).predict(Zr)
    r_raw = np.linalg.norm(_clean(X) - _clean(X).mean(0), axis=1)
    kmaug = KMeans(2, n_init=15, random_state=42).fit(np.c_[Zr, r_raw]).labels_
    sol_lab = cluster(X, ds)

    cands = {"current solver (Spectral)": sp, "KMeans k=2": km,
             "GMM k=2": gm, "KMeans [Z+r] k=2": kmaug,
             "radius split (oracle)": ref}
    print(f"   {'candidate':28s} {'ARI vs ref':>10s} {'NMI vs ref':>10s} {'sil(Z)':>8s}")
    for name, lab in cands.items():
        a = adjusted_rand_score(ref, lab)
        m = normalized_mutual_info_score(ref, lab)
        s = silhouette_score(Zr, lab) if len(np.unique(lab)) == 2 else -1.0
        print(f"   {name:28s} {a:10.3f} {m:10.3f} {s:8.3f}")
    s_ref = silhouette_score(Zr, ref)
    print(f"   reference silhouette in solver space: {s_ref:.3f}")

    # --- bootstrap stability of the radius split (resample 70%, refit) ---
    aris = []
    rng = np.random.default_rng(42)
    for t in range(50):
        idx = rng.choice(n, size=int(0.7 * n), replace=False)
        sub = p2[idx]
        c_, th_, R_ = ellipse_boundary(sub)
        lab_sub = inside_boundary(sub, c_, th_, R_).astype(int)
        aris.append(adjusted_rand_score(ref[idx], lab_sub))
    print(f"   bootstrap ARI of radius split (50x70% resamples): "
          f"mean={np.mean(aris):.3f} sd={np.std(aris):.3f}")

    # --- figures ---
    fig, ax = plt.subplots(1, 3, figsize=(16.8, 5.4))
    a = ax[0]
    a.scatter(p2[:, 0], p2[:, 1], c=sol_lab, cmap="viridis", s=4, alpha=0.7)
    a.set_title(f"current solver k=2 (Spectral):\narc-vs-arc split, ARI vs ref = "
                f"{adjusted_rand_score(ref, sp):.3f}")
    a = ax[1]
    a.scatter(p2[ref == 0, 0], p2[ref == 0, 1], c="navy", s=4, alpha=0.5,
              label="ellipse / ring")
    a.scatter(p2[ref == 1, 0], p2[ref == 1, 1], c="orange", s=6, alpha=0.9,
              label=f"central blob ({frac_blob:.0%})")
    tt = np.linspace(-np.pi, np.pi, 400)
    Rt = np.interp(tt, np.concatenate([th - 2*np.pi, th, th + 2*np.pi]),
                   np.concatenate([R, R, R]))
    a.plot(c[0] + Rt * np.cos(tt), c[1] + Rt * np.sin(tt), "r--", lw=1.3,
           label="fitted boundary")
    a.legend(loc="best", fontsize=8)
    a.set_title(f"reference split: ellipse + blob\nsilhouette = {s_ref:.3f}"
                f" (vs 0.29 for arc split)")
    a = ax[2]
    d = np.linalg.norm(p2 - c, axis=1)
    m = d <= np.quantile(d, 0.55)
    a.scatter(p2[m, 0], p2[m, 1], c=ref[m], cmap="autumn", s=6, alpha=0.85)
    a.set_title("centre zoom (inner 55%): dense blob vs sparse ring")
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.suptitle(ds, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{ds}_ring_blob.png", bbox_inches="tight")
    plt.close(fig)

print("saved *_ring_blob.png to sol_16_23/")
