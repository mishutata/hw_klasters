"""ds16/ds23 round 4: per-angle boundary at the density valley.

A global scalar frac cannot be right everywhere: the central blob is wide and
elliptical, so its edge sits at different q = d/R(theta) in different directions.
The Bayes-optimal inner/outer cut (under the blob+ring generative model) is where
the two densities cross, i.e. the per-angle density valley. We estimate it with a
1D KDE of q inside each 10-degree angle bin, smooth it circularly, fill bins
without a clean two-peak structure by interpolation, and label inner iff
q < v_local(theta).

Validation (no silhouette in the decision):
  - bootstrap ARI stability of the whole procedure (70% resamples),
  - anchor checks (deep-blob points must be inner, deep-ring outer),
  - comparison vs the old global frac=0.80 and the round-3 scalar picks.
"""
import sys
sys.path.insert(0, "my_solution6")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import solution as v6

rng = np.random.default_rng(42)
NB = 36


def radius_profile(p2, nb=NB):
    """Per-angle ring centerline R(theta): median radius of the outer shell,
    circularly smoothed. Returns c, theta-centers, R."""
    c0 = p2.mean(0)
    d0 = np.linalg.norm(p2 - c0, axis=1)
    outer = d0 >= np.quantile(d0, 0.45)
    ang = np.arctan2(p2[:, 1] - c0[1], p2[:, 0] - c0[0])
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    Rb = np.full(nb, np.nan)
    for i, (e, f) in enumerate(zip(edges[:-1], edges[1:])):
        m = (ang >= e) & (ang < f) & outer
        if m.any():
            Rb[i] = np.median(d0[m])
    for i in range(nb):
        if np.isnan(Rb[i]):
            j = (i + 1) % nb
            while np.isnan(Rb[j]):
                j = (j + 1) % nb
            Rb[i] = Rb[j]
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    sm = np.array([sum(k[j] * Rb[(i - j) % nb] for j in range(5)) for i in range(nb)])
    th = (edges[:-1] + edges[1:]) / 2
    return c0, th, sm


def q_of(p2, c, th, R):
    d0 = np.linalg.norm(p2 - c, axis=1)
    a = (np.arctan2(p2[:, 1] - c[1], p2[:, 0] - c[0]) + np.pi) % (2 * np.pi) - np.pi
    t_ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
    R_ext = np.concatenate([R, R, R])
    return d0 / np.interp(a, t_ext, R_ext)


def per_angle_valley(p2, c, th, R, grid=None):
    """Per-bin density valley between the blob peak (q<0.8) and ring peak
    (q>0.85). Returns (v_per_bin [may contain NaN], grid)."""
    if grid is None:
        grid = np.linspace(0, 1.5, 300)
    q = q_of(p2, c, th, R)
    ang = (np.arctan2(p2[:, 1] - c[1], p2[:, 0] - c[0]) + np.pi) % (2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, NB + 1)
    v = np.full(NB, np.nan)
    for i, (e, f) in enumerate(zip(edges[:-1], edges[1:])):
        qs = q[(ang >= e) & (ang < f)]
        if len(qs) < 15:
            continue
        dens = gaussian_kde(qs)(grid)
        ib = grid.searchsorted(0.8)
        ir = grid.searchsorted(0.85)
        pb, pr = dens[:ib].max(), dens[ir:].max()
        if pb <= 0 or pr <= 0:
            continue
        bi, ri = int(np.argmax(dens[:ib])), ir + int(np.argmax(dens[ir:]))
        if ri <= bi:
            continue
        seg = dens[bi:ri + 1]
        jv = bi + int(np.argmin(seg))
        if dens[jv] < 0.8 * min(dens[bi], dens[ri]):   # a real dip, not a plateau
            v[i] = grid[jv]
    return v, grid


def smooth_fill(v, nb=NB):
    """Circular smoothing of valid bins + interpolation fill of the rest."""
    out = np.array(v, float)
    good = ~np.isnan(out)
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    if good.sum() >= 4:
        gg = out[good]
        sm = np.full(nb, np.nan)
        for i in range(nb):
            if good[i]:
                sm[i] = sum(k[j] * out[(i - j) % nb] for j in range(5)) \
                    if all(good[(i - j) % nb] for j in range(5)) else out[i]
        # interpolate NaN bins from the smoothed good ones (circular)
        idx = np.arange(nb)
        gi = idx[good]
        # unwrap circularly: work on an extended axis
        ext_i = np.concatenate([gi - nb, gi, gi + nb])
        ext_v = np.concatenate([sm[gi % nb], sm[gi], sm[gi % nb]])
        for j in range(nb):
            if not good[j]:
                out[j] = float(np.interp(j, ext_i, ext_v))
    elif good.any():
        out.fill(float(out[good].mean()))
    else:
        out.fill(0.85)
    return out


def labels_per_angle(p2, c, th, R, v):
    q = q_of(p2, c, th, R)
    ang = (np.arctan2(p2[:, 1] - c[1], p2[:, 0] - c[0]) + np.pi) % (2 * np.pi) - np.pi
    t_ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
    v_ext = np.concatenate([v, v, v])
    vloc = np.interp(ang, t_ext, v_ext)
    return (q < vloc).astype(int), q


def fit_labels(p2):
    c, th, R = radius_profile(p2)
    v_raw, grid = per_angle_valley(p2, c, th, R)
    v = smooth_fill(v_raw)
    lab, q = labels_per_angle(p2, c, th, R, v)
    return lab, q, v_raw, v


def bootstrap_ari(fit, p2, n_boot=30, frac=0.7):
    n = len(p2)
    full = fit(p2)[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        sub = p2[idx]
        # extend subsample boundary to the full set
        c_s, th_s, R_s = radius_profile(sub)
        _, _, v_raw_s, v_s = fit(sub)
        lab_full_ext, _ = labels_per_angle(p2, c_s, th_s, R_s, v_s)
        vals.append(adjusted_rand_score(full, lab_full_ext))
    return float(np.mean(vals)), float(np.std(vals))


datas = {}
for ds in ("ds16", "ds23"):
    X = v6._clean(v6.load_ds(ds)) if hasattr(v6, "load_ds") else None
# load via the same path as other dev scripts
import pandas as pd
tab = pd.read_csv("public_data/train.csv")
_cols = ["dataset_id"] + (["id"] if "id" in tab.columns else [])
datas = {k: tab[tab.dataset_id == k].drop(columns=_cols).values for k in ("ds16", "ds23")}

res = {}
for ds in ("ds16", "ds23"):
    Xc = v6._clean(datas[ds])
    sc = StandardScaler().fit_transform(Xc)
    p2 = PCA(n_components=2, random_state=42).fit_transform(sc)
    lab, q, v_raw, v = fit_labels(p2)
    ari, sd = bootstrap_ari(fit_labels, p2, n_boot=30)
    frac = float(lab.mean())
    # anchors (frac-independent definitions)
    deep_blob = q < 0.6
    deep_ring = q >= 0.95
    e_blob = 1 - lab[deep_blob].mean() if deep_blob.any() else float("nan")
    e_ring = lab[deep_ring].mean() if deep_ring.any() else float("nan")
    # comparison vs global fracs
    ari_vs_old = adjusted_rand_score(lab, (q < 0.80).astype(int))
    print(f"== {ds} n={len(p2)} inner_frac={frac:.3f}  bootstrap ARI={ari:.3f}(+/-{sd:.3f})")
    print(f"   deep-blob(<0.6) mislabeled outer: {e_blob:.1%} ({int(deep_blob.sum())} pts)")
    print(f"   deep-ring(>=0.95) mislabeled inner: {e_ring:.1%} ({int(deep_ring.sum())} pts)")
    print(f"   ARI vs old global frac=0.80: {ari_vs_old:.3f}")
    print(f"   valley bins ok: {int((~np.isnan(v_raw)).sum())}/{NB}, raw median={np.nanmedian(v_raw):.3f}")
    res[ds] = dict(p2=p2, lab=lab, q=q, v_raw=v_raw, v=v,
                   c=None)

# figure: labels + valley profile vs old boundary
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for r, ds in enumerate(("ds16", "ds23")):
    d = res[ds]
    p2, lab, q, v_raw, v = d["p2"], d["lab"], d["q"], d["v_raw"], d["v"]
    c0 = p2.mean(0)
    a = axes[r][0]
    a.scatter(p2[lab == 1, 0], p2[lab == 1, 1], s=4, c="tab:orange", label="inner (blob)")
    a.scatter(p2[lab == 0, 0], p2[lab == 0, 1], s=4, c="navy", alpha=0.6, label="outer (ring)")
    th_f = np.linspace(-np.pi, np.pi, 400)
    c, th_c, R_c = radius_profile(p2)

    def circ(t, vals):
        t_ext = np.concatenate([th_c - 2 * np.pi, th_c, th_c + 2 * np.pi])
        return np.interp(t, t_ext, np.concatenate([vals, vals, vals]))

    # boundary in data coords: d = v(theta) * R(theta)
    bb = circ(th_f, v) * circ(th_f, R_c)
    a.plot(c0[0] + bb * np.cos(th_f), c0[1] + bb * np.sin(th_f), "g-", lw=2,
           label="per-angle valley boundary")
    ob = 0.80 * circ(th_f, R_c)
    a.plot(c0[0] + ob * np.cos(th_f), c0[1] + ob * np.sin(th_f), "k--", lw=1.5,
           label="old global frac=0.80")
    a.set_title(f"{ds}: per-angle boundary (inner={lab.mean():.2f})")
    a.legend(fontsize=8); a.set_aspect("equal")

    a = axes[r][1]
    edges = np.linspace(-np.pi, np.pi, NB + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    a.plot(np.degrees(mid), v_raw, "o", ms=4, label="raw valley per bin")
    a.plot(np.degrees(mid), [v[i] for i in range(NB)], "g-", lw=2, label="smoothed + filled")
    a.axhline(0.80, color="k", ls="--", lw=1, label="old global frac")
    a.set_xlabel("angle (deg)"); a.set_ylabel("v(theta) = boundary in q units")
    a.set_title(f"{ds}: per-angle boundary profile")
    a.legend(fontsize=8)

    a = axes[r][2]
    a.hist(q[lab == 1], bins=60, alpha=0.6, color="tab:orange", label="inner q", range=(0, 1.5))
    a.hist(q[lab == 0], bins=60, alpha=0.6, color="navy", label="outer q", range=(0, 1.5))
    a.set_xlabel("q = d / R(theta)")
    a.set_title(f"{ds}: q distribution by label")
    a.legend(fontsize=8)
fig.suptitle("ds16/ds23 round 4: per-angle density-valley boundary (no silhouette in decision)",
             fontsize=13)
plt.tight_layout()
plt.savefig("dev_v6/ring_radius_tune4.png", dpi=110)
print("saved dev_v6/ring_radius_tune4.png")
