# -*- coding: utf-8 -*-
"""ds16/ds23 radius tuning, round 2.

Round 1 showed silhouette/stability both improve as the inner/outer boundary
moves outward (frac 0.80 -> 0.90), and that global KDE valley detection is
unreliable here (blob->ring transition is a slope, not a dip). Round 2:
  D2: fine frac sweep 0.75..1.00 on the current q45-shell ellipse;
  F:  per-angle ring inner edge: R_edge(theta) = low quantile of the outer
      shell radii in each angle bin, boundary at s * R_edge(theta);
  anchor check: % of 'ring core' (q>=0.9) mislabeled inner + % of 'blob core'
      (densest 20% by kNN distance) mislabeled outer.
Saves dev_v6/ring_radius_tune2.png
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(ROOT))  # project root
plt.rcParams.update({'figure.dpi': 130})

spec6 = importlib.util.spec_from_file_location('sol_v6', 'my_solution6/solution.py')
v6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(v6)
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}


def ellipse_boundary(p, shell_q=0.45):
    c0 = p.mean(0)
    d0 = np.linalg.norm(p - c0, axis=1)
    outer = d0 >= np.quantile(d0, shell_q)
    ang = (np.arctan2(p[:, 1] - c0[1], p[:, 0] - c0[0]) + np.pi) % (2 * np.pi) - np.pi
    nb = 36
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


def loc_radius(p, c, th, R):
    d = np.linalg.norm(p - c, axis=1)
    a = (np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]) + np.pi) % (2 * np.pi) - np.pi
    t_ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
    R_ext = np.concatenate([R, R, R])
    return d / np.interp(a, t_ext, R_ext)


def labels_from(p, c, th, R, frac):
    return (loc_radius(p, c, th, R) < frac).astype(int)


def ring_inner_edge(p, shell_q=0.45, pct=0.20, nb=36):
    """Per-angle inner edge of the ring: low quantile of outer-shell radii."""
    c0 = p.mean(0)
    d0 = np.linalg.norm(p - c0, axis=1)
    outer = d0 >= np.quantile(d0, shell_q)
    ang = (np.arctan2(p[:, 1] - c0[1], p[:, 0] - c0[0]) + np.pi) % (2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    Rb = np.full(nb, np.nan)
    for i, (e, f) in enumerate(zip(edges[:-1], edges[1:])):
        m = (ang >= e) & (ang < f) & outer
        if m.sum() >= 3:
            Rb[i] = np.quantile(d0[m], pct)
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


def bootstrap_ari(fit_and_label, p, n_boot=30, seed=42):
    rng = np.random.default_rng(seed)
    aris = []
    lab_full = fit_and_label(p)
    for _ in range(n_boot):
        idx = rng.choice(len(p), size=int(0.7 * len(p)), replace=False)
        lab_sub = fit_and_label(p[idx])
        aris.append(adjusted_rand_score(lab_full[idx], lab_sub))
    return float(np.mean(aris)), float(np.std(aris))


def anchor_errors(p, c, th, R, frac):
    # % ring core (q>=0.9) labeled inner, % blob core (densest 20%) outer.
    q = loc_radius(p, c, th, R)
    lab = (q < frac).astype(int)
    nn = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(p)
    dd, _ = nn.kneighbors(p)
    blob_core = np.zeros(len(p), bool)
    blob_core[np.argsort(dd[:, -1])[: int(0.2 * len(p))]] = True
    ring_core = q >= 0.9
    e_ring = lab[ring_core].mean() if ring_core.any() else float('nan')
    e_blob = 1 - lab[blob_core].mean() if blob_core.any() else float('nan')
    return e_ring, e_blob


for ds in ('ds16', 'ds23'):
    X = groups[ds]
    n = len(X)
    Z = v6._reduce(v6._clean(X), v6.NORM_PRIOR.get(ds, 'minmax'))
    sc = StandardScaler().fit_transform(v6._clean(X))
    p2 = PCA(n_components=2, random_state=42).fit_transform(sc)
    c, th, R = ellipse_boundary(p2)
    q = loc_radius(p2, c, th, R)
    print('=' * 78)
    print('%s  n=%d' % (ds, n))

    # ---- D2: fine frac sweep ----
    fracs = np.round(np.arange(0.75, 1.001, 0.025), 3)
    rows = []
    for f in fracs:
        lab = labels_from(p2, c, th, R, f)
        sil = silhouette_score(Z, lab)
        ari, sd = bootstrap_ari(lambda p_, f=f: labels_from(p_, *ellipse_boundary(p_), f), p2)
        er, eb = anchor_errors(p2, c, th, R, f)
        rows.append((f, sil, ari, sd, er, eb))
        print('  D2 frac=%.3f  sil=%.3f  ARI=%.3f(%.3f)  ringMisLabeledInner=%.1f%% blobOuter=%.1f%%'
              % (f, sil, ari, sd, er * 100, eb * 100))

    # ---- F: per-angle ring inner edge ----
    for pct in (0.15, 0.20, 0.30):
        for s in (0.95, 1.0):
            ce, thee, Re = ring_inner_edge(p2, shell_q=0.45, pct=pct)
            lab = (loc_radius(p2, ce, thee, Re) < s).astype(int)
            sil = silhouette_score(Z, lab)
            ari, sd = bootstrap_ari(lambda p_, pct=pct, s=s:
                                    (loc_radius(p_, *ring_inner_edge(p_, pct=pct)) < s).astype(int),
                                    p2)
            qe = loc_radius(p2, ce, thee, Re)
            er, eb = anchor_errors(p2, ce, thee, Re, 1.0)
            print('  F pct=%.2f s=%.2f   sil=%.3f  ARI=%.3f(%.3f)  ringIn=%.1f%% blobOut=%.1f%%'
                  % (pct, s, sil, ari, sd, er * 100, eb * 100))

# ---------- figures ----------
fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5))
for r, ds in enumerate(('ds16', 'ds23')):
    X = groups[ds]
    Z = v6._reduce(v6._clean(X), v6.NORM_PRIOR.get(ds, 'minmax'))
    sc = StandardScaler().fit_transform(v6._clean(X))
    p2 = PCA(n_components=2, random_state=42).fit_transform(sc)
    c, th, R = ellipse_boundary(p2)
    q = loc_radius(p2, c, th, R)
    tt = np.linspace(-np.pi, np.pi, 400)
    Rt = np.interp(tt, np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi]),
                   np.concatenate([R, R, R]))
    a = axes[r][0]
    a.scatter(p2[:, 0], p2[:, 1], c=q, cmap='viridis', s=4, alpha=0.7)
    for f, col in ((0.80, 'red'), (0.90, 'orange')):
        a.plot(c[0] + f * Rt * np.cos(tt), c[1] + f * Rt * np.sin(tt), ls='--', lw=1.2,
               color=col, label='frac=%.2f' % f)
    ce, thee, Re = ring_inner_edge(p2, pct=0.20)
    Rte = np.interp(tt, np.concatenate([thee - 2 * np.pi, thee, thee + 2 * np.pi]),
                    np.concatenate([Re, Re, Re]))
    a.plot(ce[0] + Rte * np.cos(tt), ce[1] + Rte * np.sin(tt), ':', lw=1.4,
           color='magenta', label='ring inner edge (p20)')
    a.set_title('%s: points colored by q=d/R(theta)' % ds)
    a.legend(fontsize=8, loc='best')
    a = axes[r][1]
    fracs = np.linspace(0.75, 1.0, 60)
    sils, aris = [], []
    for f in fracs:
        lab = labels_from(p2, c, th, R, f)
        sils.append(silhouette_score(Z, lab))
        ari, _ = bootstrap_ari(lambda p_, f=f: labels_from(p_, *ellipse_boundary(p_), f), p2, n_boot=10)
        aris.append(ari)
    a.plot(fracs, sils, 'b-', label='silhouette(Z)')
    a2 = a.twinx()
    a2.plot(fracs, aris, 'g--', label='bootstrap ARI (10x)')
    a.set_xlabel('frac'); a.set_ylabel('silhouette', color='b')
    a2.set_ylabel('ARI', color='g')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8)
    a.set_title('%s: sweep' % ds)
    a = axes[r][2]
    nn = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(p2)
    dd, _ = nn.kneighbors(p2)
    blob_core = np.zeros(len(p2), bool)
    blob_core[np.argsort(dd[:, -1])[: int(0.2 * len(p2))]] = True
    ring_core = q >= 0.9
    a.scatter(p2[~blob_core & ~ring_core, 0], p2[~blob_core & ~ring_core, 1],
              c='lightgray', s=3, alpha=0.4)
    a.scatter(p2[blob_core, 0], p2[blob_core, 1], c='orange', s=6, alpha=0.9,
              label='blob core (densest 20%)')
    a.scatter(p2[ring_core, 0], p2[ring_core, 1], c='navy', s=6, alpha=0.9,
              label='ring core (q>=0.9)')
    for f, col in ((0.80, 'red'), (0.90, 'orange')):
        a.plot(c[0] + f * Rt * np.cos(tt), c[1] + f * Rt * np.sin(tt), ls='--', lw=1.2,
               color=col, label='frac=%.2f' % f)
    a.legend(fontsize=8, loc='best')
    a.set_title('%s: anchor populations' % ds)
for rr in range(2):
    for cc in range(3):
        axes[rr][cc].set_xlabel('PC1'); axes[rr][cc].set_ylabel('PC2')
fig.suptitle('ds16/ds23 radius tuning round 2', fontsize=14)
fig.tight_layout()
fig.savefig('dev_v6/ring_radius_tune2.png', bbox_inches='tight')
print('saved dev_v6/ring_radius_tune2.png')