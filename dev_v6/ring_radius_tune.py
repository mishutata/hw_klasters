# -*- coding: utf-8 -*-
"""ds16/ds23: tune the ring/blob radius boundary (v6 keeps k=2 ring split here).

The v5/v6 rule is fixed: R(theta) = per-angle median of the outer shell
(points with d >= q45), label inside if d < 0.80 * R(theta). The fraction 0.80
is a guess, and the per-angle median can sag where the ring has gaps.

Candidates tested here (all in PCA-2 space, labels compared in solver Z space):
  A  current: shell q=0.45, frac=0.80
  B  global valley: q = d/R(theta); 1-D KDE on q; boundary at the density
     minimum between the blob peak (q<1) and ring peak (q~1) -> optimal frac
  C  per-angle valley: iterate {coarse q split -> per-angle gap midpoint}
     so each angle gets its own boundary radius following the local gap
  D  fraction sweep on the current ellipse: frac in 0.60..0.95 step 0.01
  E  shell quantile sweep (0.30/0.40/0.50) x best-of-B/C frac

Metrics per candidate: silhouette in solver Z, inner fraction, bootstrap ARI
(30x70% resamples), and figures for visual judgement.
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
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(ROOT))
plt.rcParams.update({'figure.dpi': 150})

spec6 = importlib.util.spec_from_file_location('sol_v6', 'my_solution6/solution.py')
v6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(v6)

df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
groups = {ds: g[xcols].values.astype(float) for ds, g in df.groupby('dataset_id')}


def ellipse_boundary(p, shell_q=0.45, nb=36):
    """Per-angle median radius of the outer shell (v6 rule)."""
    c0 = p.mean(0)
    d0 = np.linalg.norm(p - c0, axis=1)
    outer = d0 >= np.quantile(d0, shell_q)
    ang = np.arctan2(p[:, 1] - c0[1], p[:, 0] - c0[0])
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
    q = loc_radius(p, c, th, R)
    return (q < frac).astype(int), q


def global_valley_frac(q, lo=0.45, hi=0.98):
    """frac* = density minimum of the 1-D radial profile between blob and ring."""
    m = (q > lo) & (q < hi)
    if m.sum() < 30:
        return 0.80
    kde = gaussian_kde(q[m], bw_method=0.15)
    grid = np.linspace(lo, hi, 400)
    dens = kde(grid)
    # smooth a bit more, then argmin
    kern = np.array([1, 2, 3, 2, 1], float); kern /= kern.sum()
    ds = np.convolve(dens, kern, mode='same')
    return float(grid[np.argmin(ds)])


GRID36 = (np.arange(36) + 0.5) * 2 * np.pi / 36 - np.pi


def per_angle_valley(p, iters=3, nb=72):
    """Boundary radius per angle at the local density gap between blob and ring.

    Start from a coarse inside/outside split (q < 0.8); in each fine angle bin
    take midpoint of (max inner radius, min outer radius) when there is a clean
    gap, else the KDE valley of the radii; refit the split; repeat.
    Returns R_valley on the 36-bin grid GRID36 (radii, not fractions).
    """
    c0 = p.mean(0)
    d0 = np.linalg.norm(p - c0, axis=1)
    ang = (np.arctan2(p[:, 1] - c0[1], p[:, 0] - c0[0]) + np.pi) % (2 * np.pi) - np.pi
    _, th36, R36c = ellipse_boundary(p, shell_q=0.45)
    inside = loc_radius(p, c0, th36, R36c) < 0.80
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    th_f = (edges[:-1] + edges[1:]) / 2
    bnd = np.full(nb, np.nan)
    for it in range(iters):
        keep = bnd.copy() if it > 0 else None
        for i, (e, f) in enumerate(zip(edges[:-1], edges[1:])):
            m = (ang >= e) & (ang < f)
            if m.sum() < 8:
                bnd[i] = keep[i] if keep is not None and not np.isnan(keep[i]) else np.nan
                continue
            di, do = d0[m][inside[m]], d0[m][~inside[m]]
            if len(di) < 3 or len(do) < 3:
                bnd[i] = keep[i] if keep is not None and not np.isnan(keep[i]) else np.nan
                continue
            lo_i, hi_o = di.max(), do.min()
            if hi_o > lo_i:                      # clean gap -> midpoint
                bnd[i] = 0.5 * (lo_i + hi_o)
            else:                                # overlap -> density valley in bin
                allr = d0[m]
                kde = gaussian_kde(allr, bw_method=0.2)
                grid = np.linspace(allr.min(), allr.max(), 120)
                ds = kde(grid)
                kern = np.array([1, 2, 3, 2, 1], float); kern /= kern.sum()
                ds = np.convolve(ds, kern, mode='same')
                bnd[i] = grid[np.argmin(ds)]
        for i in range(nb):                      # fill empty bins (circular)
            if np.isnan(bnd[i]):
                j = (i + 1) % nb
                while np.isnan(bnd[j]):
                    j = (j + 1) % nb
                bnd[i] = bnd[j]
        k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
        sm = np.array([sum(k[j] * bnd[(i - j) % nb] for j in range(5)) for i in range(nb)])
        R36 = np.interp(GRID36, th_f, sm)
        inside = d0 < np.interp(ang,
                                np.concatenate([GRID36 - 2 * np.pi, GRID36, GRID36 + 2 * np.pi]),
                                np.concatenate([R36, R36, R36]))
    return c0, R36


def make_fit_label(shell_q=0.45, frac=None, per_angle=False):
    """Return f(points) -> labels, refitting the boundary on the given points."""
    def f(p):
        if per_angle:
            c_, Rv = per_angle_valley(p)
            lab, _ = labels_from(p, c_, GRID36, Rv, 1.0)
            return lab
        c_, th_, R_ = ellipse_boundary(p, shell_q=shell_q)
        fr = frac if frac is not None else global_valley_frac(loc_radius(p, c_, th_, R_))
        lab, _ = labels_from(p, c_, th_, R_, fr)
        return lab
    return f


def bootstrap_ari(p, lab_full, fit_label, n_boot=30, frac_resample=0.7, seed=42):
    """Refit the boundary on each 70% resample; ARI vs full-data labels."""
    rng = np.random.default_rng(seed)
    n = len(p)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(frac_resample * n), replace=False)
        lab_sub = fit_label(p[idx])
        if len(np.unique(lab_sub)) < 2:
            continue
        aris.append(adjusted_rand_score(lab_full[idx], lab_sub))
    return float(np.mean(aris)), float(np.std(aris))


def boundary_curve(c, th, R):
    tt = np.linspace(-np.pi, np.pi, 400)
    Rt = np.interp(tt, np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi]),
                   np.concatenate([R, R, R]))
    return c[0] + Rt * np.cos(tt), c[1] + Rt * np.sin(tt)


results = {}
for ds in ("ds16", "ds23"):
    X = groups[ds]
    Z = v6._reduce(v6._clean(X), v6.NORM_PRIOR.get(ds, 'minmax'))
    sc = StandardScaler().fit_transform(v6._clean(X))
    p2 = PCA(n_components=2, random_state=42).fit_transform(sc)

    c, th, R = ellipse_boundary(p2, shell_q=0.45)
    q = loc_radius(p2, c, th, R)
    fracB = global_valley_frac(q)

    # per-angle valley boundary (returns centre + 36-bin radii)
    try:
        cC, Rv = per_angle_valley(p2)
    except Exception as e:
        cC, Rv = None, None
        print(ds, 'per_angle_valley failed:', e)

    # (display name, labels on full data, refit function for bootstrap)
    cand = {}
    cand['A: current (q45 shell, frac=0.80)'] = (
        labels_from(p2, c, th, R, 0.80)[0], make_fit_label(shell_q=0.45, frac=0.80))
    cand['B: global valley frac=%.3f' % fracB] = (
        labels_from(p2, c, th, R, fracB)[0], make_fit_label(shell_q=0.45))
    if Rv is not None:
        cand['C: per-angle valley'] = (
            labels_from(p2, cC, GRID36, Rv, 1.0)[0], make_fit_label(per_angle=True))
    for f in (0.70, 0.75, 0.85, 0.90):
        cand['D: frac=%.2f' % f] = (
            labels_from(p2, c, th, R, f)[0], make_fit_label(shell_q=0.45, frac=f))
    for sq in (0.30, 0.50):
        cand['E: shell q=%.2f + valley frac' % sq] = (
            labels_from(p2, *ellipse_boundary(p2, shell_q=sq),
                        global_valley_frac(loc_radius(p2, *ellipse_boundary(p2, shell_q=sq))))[0],
            make_fit_label(shell_q=sq))

    print('=' * 78)
    print(ds, ' n=%d' % len(X), '  q quantiles .1/.25/.5/.75/.9:',
          np.round(np.quantile(q, [0.1, 0.25, 0.5, 0.75, 0.9]), 3))
    print('%-34s %8s %9s %9s %9s' % ('candidate', 'sil(Z)', 'frac_in', 'bootARI', 'bootSD'))
    rows = []
    for name, (lab, fit) in cand.items():
        sil = silhouette_score(Z, lab) if len(np.unique(lab)) == 2 else float('nan')
        fi = float(lab.mean())
        ari, sd = bootstrap_ari(p2, lab, fit, n_boot=30)
        rows.append((name, sil, fi, ari, sd, lab))
        print('%-34s %8.3f %9.3f %9.3f %9.3f' % (name, sil, fi, ari, sd))
    results[ds] = {'Z': Z, 'p2': p2, 'c': c, 'th': th, 'R': R,
                   'cC': cC, 'Rv': Rv, 'rows': rows}

# ---------------- figures ----------------
fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
for r, ds in enumerate(("ds16", "ds23")):
    res = results[ds]
    p2, c, th, R, Rv = res['p2'], res['c'], res['th'], res['R'], res['Rv']
    # panel 1: current A
    a = axes[r][0]
    labA = res['rows'][0][5]
    a.scatter(p2[labA == 0, 0], p2[labA == 0, 1], c='navy', s=4, alpha=0.5)
    a.scatter(p2[labA == 1, 0], p2[labA == 1, 1], c='orange', s=6, alpha=0.9)
    bx, by = boundary_curve(c, th, R)
    a.plot(bx, by, 'r--', lw=1.4)
    a.set_title('%s A: current frac=0.80\nsil %.3f' % (ds, res['rows'][0][1]))
    # panel 2: global valley B
    a = axes[r][1]
    rowB = [x for x in res['rows'] if x[0].startswith('B:')][0]
    labB = rowB[5]
    a.scatter(p2[labB == 0, 0], p2[labB == 0, 1], c='navy', s=4, alpha=0.5)
    a.scatter(p2[labB == 1, 0], p2[labB == 1, 1], c='orange', s=6, alpha=0.9)
    fracB = rowB[0].split('=')[1]
    bx, by = boundary_curve(c, th, R * float(fracB))
    a.plot(bx, by, 'g-', lw=1.4, label='frac=%.3f' % float(fracB))
    bx2, by2 = boundary_curve(c, th, R)
    a.plot(bx2, by2, 'r--', lw=0.8, alpha=0.6, label='R(theta)')
    a.legend(fontsize=7)
    a.set_title('%s B: global valley frac=%.3f\nsil %.3f' % (ds, float(fracB), rowB[1]))
    # panel 3: per-angle valley C
    a = axes[r][2]
    if Rv is not None:
        rowC = [x for x in res['rows'] if x[0].startswith('C:')][0]
        labC = rowC[5]
        a.scatter(p2[labC == 0, 0], p2[labC == 0, 1], c='navy', s=4, alpha=0.5)
        a.scatter(p2[labC == 1, 0], p2[labC == 1, 1], c='orange', s=6, alpha=0.9)
        bx, by = boundary_curve(c, th, Rv)
        a.plot(bx, by, 'm-', lw=1.4)
        a.set_title('%s C: per-angle valley\nsil %.3f' % (ds, rowC[1]))
    else:
        a.axis('off')
    # panel 4: radial profile q with markers for A/B/C boundaries
    a = axes[r][3]
    q = loc_radius(p2, c, th, R)
    a.hist(q, bins=60, range=(0, 1.5), density=True, color='steelblue', alpha=0.7)
    kde = gaussian_kde(q, bw_method=0.15)
    grid = np.linspace(0, 1.5, 400)
    a.plot(grid, kde(grid), 'k-', lw=1, label='KDE')
    a.axvline(0.80, color='r', ls='--', lw=1, label='A: 0.80')
    rowB = [x for x in res['rows'] if x[0].startswith('B:')][0]
    a.axvline(float(rowB[0].split('=')[1]), color='g', ls='-', lw=1,
              label='B: %.3f' % float(rowB[0].split('=')[1]))
    if Rv is not None:
        qC = loc_radius(p2, c, th, Rv)
        a.axvline(1.0, color='m', ls=':', lw=1.5, label='C: d=Rvalley')
    a.set_xlim(0, 1.5)
    a.legend(fontsize=7)
    a.set_title('%s radial profile q=d/R(theta)' % ds)
for r in range(2):
    for cc in range(3):
        axes[r][cc].set_xlabel('PC1')
        axes[r][cc].set_ylabel('PC2')
    axes[r][3].set_xlabel('q = d / R(theta)')
fig.suptitle('ds16/ds23 ring/blob radius tuning (PCA-2 view)', fontsize=14)
fig.tight_layout()
fig.savefig('dev_v6/ring_radius_tune.png', bbox_inches='tight')
print('saved dev_v6/ring_radius_tune.png')
