# -*- coding: utf-8 -*-
"""ds16/ds23 radius tuning, round 3 — NO silhouette in the decision.

Overall silhouette is a bad metric for non-bulb shapes (rings/arcs): it rewards
compact convex clusters, so moving the inner/outer boundary can improve it even
while stealing ring points. Round 3 selects frac with structural criteria only:

  1. anchor errors (frac-independent reference sets):
       e_ring = % of 'ring core'  (q >= 0.9)  mislabeled inner
       e_blob = % of 'blob core'  (densest 20% by 10-NN dist in PCA-2) outer
  2. bootstrap ARI: refit the whole boundary on a 70% resample, label the full
     set with it, ARI vs the full-set labels at that frac (30 boots).
  3. ambiguous mass: % of all points inside |q - frac| < 0.05 (sitting on the
     boundary). The true boundary should sit in an empty valley.
  4. per-angle density valley cross-check: 1-D KDE of radius in each angle bin,
     valley between blob peak and ring peak -> local valley fraction v(theta);
     frac_valley = median(v) over bins with two clear peaks.

Decision: smallest e_ring + e_blob among fracs on the bootstrap-ARI plateau;
tie-break by smaller ambiguous mass. Silhouette reported for reference only.
Saves dev_v6/ring_radius_tune3.png
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(ROOT))  # project root
plt.rcParams.update({'figure.dpi': 130})

spec6 = importlib.util.spec_from_file_location('sol_v6', 'my_solution6/solution.py')
v6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(v6)
df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}

NB = 36


def ellipse_boundary(p, shell_q=0.45):
    c0 = p.mean(0)
    d0 = np.linalg.norm(p - c0, axis=1)
    outer = d0 >= np.quantile(d0, shell_q)
    ang = (np.arctan2(p[:, 1] - c0[1], p[:, 0] - c0[0]) + np.pi) % (2 * np.pi) - np.pi
    nb = NB
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


def bootstrap_ari(fit_and_label, p, n_boot=30, seed=42):
    rng = np.random.default_rng(seed)
    aris = []
    lab_full = fit_and_label(p)
    for _ in range(n_boot):
        idx = rng.choice(len(p), size=int(0.7 * len(p)), replace=False)
        lab_sub = fit_and_label(p[idx])
        aris.append(adjusted_rand_score(lab_full[idx], lab_sub))
    return float(np.mean(aris)), float(np.std(aris))


def per_angle_valley(p, c, th, R, n_grid=200):
    """Per-angle-bin density valley between blob peak and ring peak -> v(theta)."""
    d = np.linalg.norm(p - c, axis=1)
    a = (np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]) + np.pi) % (2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, NB + 1)
    Rloc = loc_radius(p, c, th, R)
    vals = []
    for i, (e, f) in enumerate(zip(edges[:-1], edges[1:])):
        m = (a >= e) & (a < f)
        if m.sum() < 15:
            continue
        qbin = Rloc[m]
        grid = np.linspace(0.25, 1.25, n_grid)
        dens = gaussian_kde(qbin)(grid)
        # blob peak: max density left of 0.75; ring peak: max density right of 0.85
        ml, mr = grid < 0.75, grid > 0.85
        if not ml.any() or not mr.any():
            continue
        pk_l = grid[ml][np.argmax(dens[ml])]
        pk_r = grid[mr][np.argmax(dens[mr])]
        mid = (grid >= pk_l) & (grid <= pk_r)
        if not mid.any():
            continue
        v = grid[mid][np.argmin(dens[mid])]
        # valley must be a real dip: dens(valley) < 0.75 * min(peak dens)
        if dens[np.argmin(dens[mid])] < 0.75 * min(dens[ml].max(), dens[mr].max()):
            vals.append(v)
    return np.array(vals)


CHOSEN = {}
for ds in ('ds16', 'ds23'):
    X = groups[ds]
    n = len(X)
    Z = v6._reduce(v6._clean(X), v6.NORM_PRIOR.get(ds, 'minmax'))
    sc = StandardScaler().fit_transform(v6._clean(X))
    p2 = PCA(n_components=2, random_state=42).fit_transform(sc)
    c, th, R = ellipse_boundary(p2)
    q = loc_radius(p2, c, th, R)

    # frac-independent anchor sets
    nn = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(p2)
    dd, _ = nn.kneighbors(p2)
    blob_core = np.zeros(n, bool)
    blob_core[np.argsort(dd[:, -1])[: int(0.2 * n)]] = True
    ring_core = q >= 0.9
    print('=' * 78)
    print('%s  n=%d   ring_core=%d blob_core=%d' % (ds, n, ring_core.sum(), blob_core.sum()))

    fracs = np.round(np.arange(0.70, 1.001, 0.01), 2)
    rows = []
    for f in fracs:
        lab = labels_from(p2, c, th, R, f)
        er = float(lab[ring_core].mean())
        eb = float(1 - lab[blob_core].mean())
        amb = float((np.abs(q - f) < 0.05).mean())
        ari, sd = bootstrap_ari(lambda p_, f=f: labels_from(p_, *ellipse_boundary(p_), f), p2)
        sil = silhouette_score(Z, lab)   # reference only
        rows.append((f, er, eb, amb, ari, sd, sil))
    rows = np.array(rows)

    # per-angle valley cross-check
    vv = per_angle_valley(p2, c, th, R)
    frac_valley = float(np.median(vv)) if len(vv) else float('nan')

    print('  frac   e_ring  e_blob  ambig  ARI(30boot)   sil(ref)')
    for f, er, eb, amb, ari, sd, sil in rows:
        mark = ''
        print('  %.2f   %5.1f%%  %5.1f%%  %5.1f%%  %.3f(%.3f)   %.3f%s'
              % (f, er * 100, eb * 100, amb * 100, ari, sd, sil, mark))
    print('  per-angle valley: n_bins=%d  median v = %.3f' % (len(vv), frac_valley))

    # decision: anchors first (both <= 2%), then ARI plateau, then ambiguity
    ok = (rows[:, 1] <= 0.02) & (rows[:, 2] <= 0.02)
    if not ok.any():
        ok = (rows[:, 1] + rows[:, 2]) <= np.quantile(rows[:, 1] + rows[:, 2], 0.25)
    cand = rows[ok]
    ari_plateau = cand[:, 4] >= cand[:, 4].max() - 0.02
    cand = cand[ari_plateau]
    best = cand[np.lexsort((cand[:, 3], -cand[:, 4]))][0]
    CHOSEN[ds] = float(best[0])
    print('  -> CHOSEN frac = %.2f  (e_ring=%.1f%% e_blob=%.1f%% ambig=%.1f%% ARI=%.3f)'
          % (best[0], best[1] * 100, best[2] * 100, best[3] * 100, best[4]))

# ---------- figures ----------
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
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
    lab_old = (q < 0.80).astype(int)
    lab_new = (q < CHOSEN[ds]).astype(int)
    cm = plt.cm.tab10
    for lab, tag in ((lab_old, 'old frac=0.80'), (lab_new, 'chosen frac=%.2f' % CHOSEN[ds])):
        a.scatter(p2[lab == 0, 0], p2[lab == 0, 1], s=3, alpha=0.6,
                  c=cm(0), label='%s (inner)' % tag)
        a.scatter(p2[lab == 1, 0], p2[lab == 1, 1], s=3, alpha=0.6,
                  c=cm(1), label='%s (ring)' % tag)
    a.plot(c[0] + CHOSEN[ds] * Rt * np.cos(tt), c[1] + CHOSEN[ds] * Rt * np.sin(tt),
           'k--', lw=1.4, label='chosen boundary')
    a.set_title('%s: old vs chosen split' % ds)
    a.legend(fontsize=7)

    a = axes[r][1]
    fracs = np.linspace(0.70, 1.00, 61)
    nn_f = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(p2)
    dd_f, _ = nn_f.kneighbors(p2)
    bc_f = np.zeros(len(p2), bool); bc_f[np.argsort(dd_f[:, -1])[: int(0.2 * len(p2))]] = True
    rc_f = q >= 0.9
    ers, ebs, ambs, aris = [], [], [], []
    for f in fracs:
        lab = labels_from(p2, c, th, R, f)
        ers.append(lab[rc_f].mean())
        ebs.append(1 - lab[bc_f].mean())
        ambs.append((np.abs(q - f) < 0.05).mean())
        ari, _ = bootstrap_ari(lambda p_, f=f: labels_from(p_, *ellipse_boundary(p_), f), p2, n_boot=10)
        aris.append(ari)
    ers, ebs, ambs, aris = map(np.array, (ers, ebs, ambs, aris))
    a.plot(fracs, ers * 100, 'r-', label='ring core mislabeled inner')
    a.plot(fracs, ebs * 100, 'b-', label='blob core mislabeled outer')
    a.plot(fracs, ambs * 100, 'm:', label='ambiguous mass (|q-frac|<.05)')
    a2 = a.twinx()
    a2.plot(fracs, aris, 'g--', label='bootstrap ARI (10x)')
    a.axvline(CHOSEN[ds], color='k', lw=1)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7)
    a.set_xlabel('frac'); a.set_ylabel('%')
    a2.set_ylabel('ARI', color='g')
    a.set_title('%s: structural metrics vs frac' % ds)

    a = axes[r][2]
    edges = np.linspace(-np.pi, np.pi, NB + 1)
    ang = (np.arctan2(p2[:, 1] - c[1], p2[:, 0] - c[0]) + np.pi) % (2 * np.pi) - np.pi
    Rloc = q
    binv = []
    for i, (e, f) in enumerate(zip(edges[:-1], edges[1:])):
        m = (ang >= e) & (ang < f)
        v = None
        if m.sum() >= 15:
            grid = np.linspace(0.25, 1.25, 200)
            dens = gaussian_kde(Rloc[m])(grid)
            ml, mr = grid < 0.75, grid > 0.85
            pk_l = grid[ml][np.argmax(dens[ml])]; pk_r = grid[mr][np.argmax(dens[mr])]
            midm = (grid >= pk_l) & (grid <= pk_r)
            if dens[midm].min() < 0.75 * min(dens[ml].max(), dens[mr].max()):
                v = grid[midm][np.argmin(dens[midm])]
        binv.append(v)
    xs = [t for t, v in zip((edges[:-1] + edges[1:]) / 2, binv) if v is not None]
    ys = [v for v in binv if v is not None]
    a.scatter(xs, ys, c='navy', s=18, zorder=3)
    if ys:
        a.axhline(np.median(ys), color='k', ls='--', lw=1.2,
                  label='median = %.3f' % np.median(ys))
    a.axhline(CHOSEN[ds], color='r', ls='-', lw=1.4, label='chosen frac')
    a.set_xlabel('angle (rad)'); a.set_ylabel('local valley v(theta)')
    a.legend(fontsize=7)
    a.set_title('%s: per-angle density valleys' % ds)

    a = axes[r][3]
    bc = bc_f
    rc = rc_f
    a.scatter(p2[~bc & ~rc, 0], p2[~bc & ~rc, 1], c='lightgray', s=3, alpha=0.4)
    a.scatter(p2[bc, 0], p2[bc, 1], c='orange', s=6, alpha=0.9, label='blob core')
    a.scatter(p2[rc, 0], p2[rc, 1], c='navy', s=6, alpha=0.9, label='ring core (q>=0.9)')
    for f, col in ((0.80, 'gray'), (CHOSEN[ds], 'red')):
        a.plot(c[0] + f * Rt * np.cos(tt), c[1] + f * Rt * np.sin(tt), ls='--', lw=1.3,
               color=col, label='frac=%.2f' % f)
    a.legend(fontsize=7)
    a.set_title('%s: anchors vs boundaries' % ds)

for rr in range(2):
    for cc in range(4):
        axes[rr][cc].set_xlabel('PC1'); axes[rr][cc].set_ylabel('PC2')
fig.suptitle('ds16/ds23 radius tuning round 3 (silhouette-free selection)', fontsize=14)
fig.tight_layout()
fig.savefig('dev_v6/ring_radius_tune3.png', bbox_inches='tight')
print('saved dev_v6/ring_radius_tune3.png')
print('CHOSEN:', CHOSEN)
