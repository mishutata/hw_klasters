"""Clustering solution FINAL — v8. Identical to v7 except two pin fixes found
by the test-size simulation (dev_v8/sim_test_eval.py, 80/20 pseudo-test):
  - ds04: winner PINNED to GaussianMixture (k stays pinned at 3). At test size
    (~190 rows) Spectral degrades (sim ARI 0.46) while GMM k=3 is stable at
    ~1.0 across seeds; KMeans 0.95.
  - ds10: k PINNED to 2 + winner PINNED to SpectralClustering. The pipeline's
    +/-1 window picked k=3 on small samples, but the stable structure is k=2:
    Spectral k=2 sim ARI ~0.85 (vs 0.41 for v7's k=3 pick).
Everything else = v7 (scored 0.48):
  - ds16/ds23 back to the ring/blob split (user verdict from the pictures):
    k PINNED to 2 and the geometric ellipse boundary is applied unconditionally,
    with per-dataset radius tuned on public train (dev_v6/ring_radius_tune3):
    frac ds16=0.93 (bootstrap ARI 0.943), ds23=0.90 (zero ring error, e_blob
    8.5% -> 2%). v6's k=4/Spectral arc-splitting for these two is dropped.
  - ds18 stays k PINNED to 4 + SpectralClustering (user-verified on public train
    pictures, dev_v6/ds18_k4_spectral.png).
  - v5 pins kept: ds09 (k=3) and ds19/ds20 (k=4), winner SpectralClustering;
v5 was identical to v3 (0.45 submission) except:
  - k is PINNED for ds04/ds09/ds19/ds20 (no +/-1 silhouette window): true cluster
    count visually verified on public train (dev_v5/k3_ds04.png / k3_ds09.png,
    dev_v5/k4_ds19.png / k4_ds20.png).
  - winner is PINNED to SpectralClustering for ds09/ds19/ds20: at the pinned k
    it recovers the real blobs while KMeans/GMM smear them (same figures).

v3 pipeline: per-dataset normalization), + ring/blob fix for ds16/ds23.

Pipeline (per dataset):
 1. NaN-safe clean -> per-dataset norm (minmax / log-sym / robust, chosen on
    public train: analysis/norm_research) -> PCA (95% variance, <=8 dims).
 2. k = prior (validated on public train), refined on a +/-1 silhouette window.
 3. Density probe: d5 = distance to 5th NN; core set = d5 <= max(p95, 2*median).
 4. Stability probe: KMeans at best_k with 3 seeds. If stable (mean pairwise ARI
    >= 0.8) fit on all points; otherwise fit on the core set so outlier tails stop
    destabilising the optimisation, and extend labels to tail points by nearest
    centroid in Z-space.
 5. Candidate ensemble: KMeans + GaussianMixture (fit on chosen set, predict all),
    SpectralClustering (fit on chosen set, tail extended), plus DBSCAN and HDBSCAN
    with auto-tuned eps / min_cluster_size as *confirming* candidates only: they
    join the vote if noise fraction is low AND they agree (NMI >= 0.5) with the best
    parametric candidate. Noise points are reassigned to the nearest cluster.
 6. Winner: silhouette + 0.3 * mean consensus NMI (deterministic).
 7. ds16/ds23 only: these are an ellipse ring + dense central blob, and k=2
    clusterers split them into arcs; the labels are replaced by the
    ellipse-aware ring/blob boundary fitted in PCA-2 space with per-dataset
    radius (frac ds16=0.93 / ds23=0.90; validated in sol_16_23/ and
    dev_v6/ring_radius_tune3, bootstrap ARI 0.943/0.94).

Why DBSCAN only as confirmer: on this data clusters overlap in density, so an
auto-tuned DBSCAN mis-splits or drowns most datasets; its reliable contribution is
core-point/outlier detection (steps 3-4), which measurably stabilises KMeans
(ds01: cross-seed ARI 0.41 -> 0.86).

Usage: labels = cluster(X, dataset_id)   # deterministic, random_state=42
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
try:
    from sklearn.cluster import HDBSCAN
except ImportError:  # scikit-learn < 1.3 has no HDBSCAN
    HDBSCAN = None
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# k validated on public train data (see notebook.ipynb).
K_PRIOR = {
    "ds00": 2, "ds01": 3, "ds02": 2, "ds03": 2, "ds04": 3, "ds05": 3,
    "ds06": 3, "ds07": 3, "ds08": 2, "ds09": 3, "ds10": 3, "ds11": 4,
    "ds12": 2, "ds13": 2, "ds14": 2, "ds15": 4, "ds16": 2, "ds17": 2,
    "ds18": 2, "ds19": 4, "ds20": 4, "ds21": 3, "ds22": 2, "ds23": 2,
}

# k pinned (no +/-1 silhouette window) where the true cluster count was visually
# verified on public train: ds04 = 3 elongated blobs; ds09 = 3 clouds;
# ds18 = 4 (three ring arcs + center); ds19/ds20 = 4 clusters;
# ds16/ds23 = 2 (ring + central blob; the geometric branch below returns its labels);
# ds10 = 2: at test size (~190 rows) the stable structure is k=2 — Spectral k=2
# sim ARI ~0.85 vs 0.41 when the +/-1 window picks k=3 (dev_v8/sim_test_eval.py).
K_PIN = {"ds04": 3, "ds09": 3, "ds10": 2, "ds16": 2, "ds18": 4,
         "ds19": 4, "ds20": 4, "ds23": 2}

# winner pinned where the true partition was visually verified on public train:
# at the pinned k SpectralClustering recovers the real blobs of ds09/ds18/
# ds19/ds20/ds10, KMeans/GMM smear them. ds04 is the exception: at test size
# GMM k=3 is stable (~1.0 across seeds) while Spectral degrades (dev_v8 sim).
# (ds16/ds23 are handled by the geometric ring/blob branch below, which
# overrides the winner entirely.)
WINNER_PIN = {"ds04": "GaussianMixture", "ds09": "SpectralClustering",
              "ds10": "SpectralClustering", "ds18": "SpectralClustering",
              "ds19": "SpectralClustering", "ds20": "SpectralClustering"}

# Best input normalization per dataset on public train (analysis/norm_research):
# minmax wins 15/24, log-sym 8/24, robust 1/24; 'standard' never wins.
NORM_PRIOR = {
    "ds00": "log-sym", "ds05": "log-sym", "ds06": "log-sym",
    "ds07": "log-sym", "ds10": "log-sym", "ds18": "log-sym",
    "ds21": "log-sym", "ds22": "log-sym", "ds02": "robust",
}


def _clean(X):
    X = np.asarray(X, dtype=float)
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep] if keep.any() else np.zeros((len(X), 1))
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = np.isnan(col)
        if bad.any():
            X[bad, j] = np.nanmean(col)
    return X


def _apply_norm(X, method):
    if method == "standard":
        return StandardScaler().fit_transform(X)
    if method == "robust":
        return RobustScaler().fit_transform(X)
    if method == "minmax":
        return MinMaxScaler().fit_transform(X)
    # log-sym: signed log compresses heavy tails, then standardize
    Xa = np.log1p(np.abs(X)) * np.sign(X)
    return StandardScaler().fit_transform(Xa)


def _reduce(X, norm="minmax"):
    Z = _apply_norm(X, norm)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    if X.shape[1] > 3:
        p = PCA(random_state=42).fit(Z)
        cum = np.cumsum(p.explained_variance_ratio_)
        n = max(2, min(8, int(np.searchsorted(cum, 0.95) + 1), X.shape[1]))
        Z = PCA(n_components=n, random_state=42).fit_transform(Z)
    return Z


def _sil(Z, lab):
    try:
        u = np.unique(lab)
        if len(u) < 2 or len(u) >= len(Z):
            return -1.0
        return float(silhouette_score(Z, lab))
    except Exception:
        return -1.0


def _consensus(cands):
    out = []
    for i, a in enumerate(cands):
        vals = [normalized_mutual_info_score(a, b)
                for j, b in enumerate(cands) if j != i]
        out.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(out)


def _nn_dist(Z, k=5):
    return cKDTree(Z).query(Z, k=k + 1)[0][:, k]


def _core_mask(Z):
    """DBSCAN core-point idea: keep points whose kth-NN distance is not an outlier."""
    n = len(Z)
    k = 5 if n >= 200 else 3
    d = _nn_dist(Z, k)
    thr = max(np.quantile(d, 0.95), np.median(d) * 2.0)
    m = d <= thr
    frac = float(1.0 - m.mean())
    if not (0.003 <= frac <= 0.08):      # no clear tail -> do not trim
        return np.ones(n, dtype=bool), 0.0
    return m, frac


def _reassign_noise(Z, lab):
    out = lab.copy()
    ni = np.where(lab < 0)[0]
    if len(ni) and (lab >= 0).any():
        uniq = np.unique(lab[lab >= 0])
        cents = np.array([Z[lab == c].mean(0) for c in uniq])
        D = ((Z[ni][:, None, :] - cents[None, :, :]) ** 2).sum(-1)
        out[ni] = uniq[np.argmin(D, axis=1)]
    return out


def _extend(Zfit, lab_fit, Zall):
    """Label points outside the fit set by nearest cluster centroid in Z-space."""
    if len(Zall) == len(Zfit):
        return np.asarray(lab_fit, dtype=int)
    uniq = np.unique(lab_fit)
    cents = np.array([Zfit[lab_fit == c].mean(0) for c in uniq])
    D = ((Zall[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    return uniq[np.argmin(D, axis=1)]


def cluster(X, dataset_id=None):
    """Return labels only (grading API). Use explain() for the full report."""
    return _run(X, dataset_id)[0]


def explain(X, dataset_id=None):
    """Same pipeline as cluster(); returns dict with 'labels', winner, per-method
    scores, k info, KM stability and core-trim stats. Keys: labels, k_prior,
    k_used, sil_by_k, km_stability, core_frac_removed, fit_set, methods, winner."""
    labels, info = _run(X, dataset_id)
    out = {'labels': labels}
    out.update(info)
    return out


def _run(X, dataset_id=None):
    Xc = _clean(X)
    n = len(Xc)
    if n < 4:
        lab = np.zeros(n, dtype=int)
        info = {'k_prior': None, 'k_used': 1, 'sil_by_k': {},
                'km_stability': 1.0, 'core_frac_removed': 0.0, 'fit_set': 'all',
                'methods': {'constant': 0.0}, 'winner': 'constant'}
        return lab, info
    ds = str(dataset_id or "").lower().strip("'\" ")
    Z = _reduce(Xc, NORM_PRIOR.get(ds, "minmax"))
    k0 = K_PIN.get(ds, K_PRIOR.get(ds, 3))
    if ds in K_PIN:                      # visually verified: no +/-1 window
        ks = [k0]
    else:
        ks = sorted({max(2, k0 - 1), k0, min(k0 + 1, max(2, n // 5))})

    # --- choose k: prior first; override only on a strong silhouette signal ---
    sils = {}
    for k in ks:
        sils[k] = _sil(Z, KMeans(n_clusters=k, n_init=15, random_state=42).fit(Z).labels_)
    best_k = max(ks, key=lambda k: -sils[k])
    if best_k != k0 and sils[best_k] - sils[k0] < 0.04:
        best_k = k0  # e.g. keep ring+center as 2, don't split the ring
    best_k = min(best_k, max(2, n // 3))

    # --- v2: stability probe + density-based core set ---
    fits = [KMeans(n_clusters=best_k, n_init=15, random_state=s).fit(Z)
            for s in (42, 7, 123)]
    stab = float(np.mean([adjusted_rand_score(fits[i].labels_, fits[j].labels_)
                          for i, j in ((0, 1), (0, 2), (1, 2))]))
    core_m, frac_rm = _core_mask(Z)
    use_core = stab < 0.8 and frac_rm > 0
    Zfit = Z[core_m] if use_core else Z

    # --- ensemble at best_k ---
    cands = []
    km = KMeans(n_clusters=best_k, n_init=15, random_state=42).fit(Zfit)
    cands.append(('KMeans', _extend(Zfit, km.labels_, Z)))
    try:
        gm = GaussianMixture(n_components=best_k, covariance_type="full",
                             n_init=3, random_state=42).fit(Zfit)
        lab_gm = gm.predict(Z)
        if len(np.unique(lab_gm)) == best_k:
            cands.append(('GaussianMixture', lab_gm))
    except Exception:
        pass
    try:
        sp = SpectralClustering(n_clusters=best_k, affinity="nearest_neighbors",
                                n_neighbors=min(len(Zfit) - 1,
                                                max(10, int(np.sqrt(len(Zfit))))),
                                random_state=42).fit_predict(Zfit)
        if len(np.unique(sp)) == best_k:
            cands.append(('SpectralClustering', _extend(Zfit, sp, Z)))
    except Exception:
        pass

    # --- v2: DBSCAN with auto eps (confirming candidate only) ---
    d5 = _nn_dist(Z, 5 if n >= 200 else 3)
    db_lab = None
    for eps in np.unique(np.quantile(d5, [0.60, 0.75, 0.90, 0.95])):
        for ms in (max(3, n // 100), 5):
            lab = DBSCAN(eps=float(eps), min_samples=ms).fit_predict(Z)
            ncl = len(np.unique(lab)) - (1 if (lab < 0).any() else 0)
            noise = float((lab < 0).mean())
            if 2 <= ncl <= max(3, k0 + 3) and noise <= 0.15:
                lab_r = _reassign_noise(Z, lab)
                if db_lab is None or _sil(Z, lab_r) > _sil(Z, db_lab):
                    db_lab = lab_r
    if db_lab is not None and max(
            normalized_mutual_info_score(db_lab, lab) for _, lab in cands) >= 0.5:
        cands.append(('DBSCAN', db_lab))

    # --- v2: HDBSCAN (confirming candidate only) ---
    try:
        hd = HDBSCAN(min_cluster_size=max(10, n // (4 * best_k))).fit_predict(Z)
        ncl = len(np.unique(hd)) - (1 if (hd < 0).any() else 0)
        noise = float((hd < 0).mean())
        if 2 <= ncl and noise <= 0.35:
            hd_r = _reassign_noise(Z, hd)
            if max(normalized_mutual_info_score(hd_r, lab) for _, lab in cands) >= 0.5:
                cands.append(('HDBSCAN', hd_r))
    except Exception:
        pass

    score = np.array([_sil(Z, c) for _, c in cands]) + 0.3 * _consensus([c for _, c in cands])
    pin = WINNER_PIN.get(ds)
    if pin is not None and any(nm == pin for nm, _ in cands):
        i = [nm for nm, _ in cands].index(pin)   # visually verified winner (ds04/ds09/ds19/ds20)
    else:
        i = int(np.argmax(score))
    labels = np.asarray(cands[i][1], dtype=int)
    info = {
        'k_prior': k0,
        'k_used': best_k,
        'sil_by_k': {k: round(float(s), 3) for k, s in sils.items()},
        'km_stability': round(stab, 3),
        'core_frac_removed': round(frac_rm, 3),
        'fit_set': 'core' if use_core else 'all',
        'methods': {name: round(float(s), 3) for (name, _), s in zip(cands, score)},
        'winner': cands[i][0],
    }
    # ds16/ds23: ring + central blob geometry beats arc-splitting (sol_16_23/);
    # applied unconditionally — silhouette is a poor selector for non-bulb shapes.
    if ds in RING_DS:
        rb = _ring_blob_labels(Xc, ds)
        if rb is not None:
            info['winner'] = 'ring/blob'
            info['methods'] = {**info['methods'],
                               'ring/blob': round(_sil(Z, rb), 3)}
            return rb, info
    return labels, info


# --- ds16/ds23: ellipse ring + central blob fix (validated in sol_16_23/) ---
# Standard k=2 clusterers (KMeans/Spectral/GMM) split these datasets into arcs:
# ARI ~ 0 vs the true ring/blob split. Here we fit an ellipse-aware radial
# boundary in PCA-2 space and label inside/outside instead; bootstrap ARI
# 0.93/0.94 across 70% resamples (sol_16_23/eval_ring_blob.py).
RING_DS = ("ds16", "ds23")
# per-dataset boundary radius frac, tuned on public train in dev_v6/ring_radius_tune3
# (structural criteria + bootstrap ARI): ds16 0.93 -> bootstrap ARI 0.943;
# ds23 0.90 -> zero ring error, e_blob 8.5% -> 2%.
RING_FRAC = {"ds16": 0.93, "ds23": 0.90}


def _ellipse_boundary(p):
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


def _inside_boundary(p, c, th, R, frac=0.80):
    """True if point is clearly inside the fitted boundary."""
    d = np.linalg.norm(p - c, axis=1)
    a = (np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]) + np.pi) % (2 * np.pi) - np.pi
    t_ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
    R_ext = np.concatenate([R, R, R])
    Rloc = np.interp(a, t_ext, R_ext)
    return d < frac * Rloc


def _ring_blob_labels(Xc, ds):
    """Reference ring/blob split for ds16/ds23; None if the fit degenerates."""
    sc = StandardScaler().fit_transform(Xc)
    p2 = PCA(n_components=2, random_state=42).fit_transform(sc)
    c, th, R = _ellipse_boundary(p2)
    lab = _inside_boundary(p2, c, th, R, frac=RING_FRAC.get(ds, 0.80)).astype(int)
    frac = float(lab.mean())
    if not 0.15 < frac < 0.85:                  # degenerate fit -> fall back
        return None
    return lab
