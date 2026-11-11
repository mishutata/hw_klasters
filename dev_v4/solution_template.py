"""Clustering solution FINAL — v4 (transductive ensemble anchored on public train).

v3 clustered only the ~20% of points that cluster() receives at scoring time,
so k-selection and winner selection were noisy. v4 anchors everything on the
public 80% train set, embedded below as per-dataset artifacts fitted ONCE on
public_data/train.csv (see notebook, section 'Artifacts'): norm parameters,
PCA transform, the Z-space point cloud Z_tr of the whole public train.
No labels are used anywhere — only unlabeled public data.

At call time:
 1. Test points are transformed into the train's Z-space with saved parameters
    (fallback: refit locally on the given points if the dataset is unknown).
 2. The ensemble runs on the POOLED set Z_pool = [Z_tr; Z_te] (~5x more points):
    k window (+/-1 silhouette around the prior), KMeans stability probe,
    density core mask, candidates KMeans + GaussianMixture + SpectralClustering
    (DBSCAN/HDBSCAN as confirming candidates only), winner = silhouette + 0.3*consensus NMI.
 3. Labels are returned for the test rows only.
 4. ds16/ds23: ellipse ring + dense central blob; an ellipse-aware radial boundary
    is fitted on pooled Z[:, :2] and inside/outside becomes the labels.

Deterministic: random_state=42 everywhere. Usage: labels = cluster(X, dataset_id)
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

# Best input normalization per dataset on public train (analysis/norm_research):
# minmax wins 15/24, log-sym 8/24, robust 1/24; 'standard' never wins.
NORM_PRIOR = {
    "ds00": "log-sym", "ds05": "log-sym", "ds06": "log-sym",
    "ds07": "log-sym", "ds10": "log-sym", "ds18": "log-sym",
    "ds21": "log-sym", "ds22": "log-sym", "ds02": "robust",
}

# ds16/ds23 have ring + central blob geometry (see notebook section 3).
RING_DS = ("ds16", "ds23")

# Per-dataset artifacts fitted on public_data/train.csv (unlabeled):
#   d_in, d_pca, k0, norm, nprm (norm params), pmean, pcomp (PCA params),
#   ztr = hex string of raw float32 bytes of Z_tr (whole public train)
ART = None  # __ART__


def _uhex(s, shape):
    # hex of raw float32 bytes (float noise is incompressible, so no zlib/gzip
    # -- only numpy + builtins are needed to decode)
    return np.frombuffer(bytes.fromhex(s), dtype=np.float32).copy().reshape(shape)


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


def _fit_norm_params(X, method):
    """Parameters of the per-dataset normalization fitted on public train."""
    if method == "standard":
        s = StandardScaler().fit(X)
        return [s.mean_.tolist(), s.scale_.tolist()]
    if method == "robust":
        s = RobustScaler().fit(X)
        return [s.center_.tolist(), s.scale_.tolist()]
    if method == "minmax":
        s = MinMaxScaler().fit(X)
        return [s.data_min_.tolist(), s.scale_.tolist()]
    # log-sym: signed log compresses heavy tails, then standardize
    Xa = np.log1p(np.abs(X)) * np.sign(X)
    s = StandardScaler().fit(Xa)
    return [s.mean_.tolist(), s.scale_.tolist()]


def _apply_norm(X, method, prm):
    if method == "log-sym":
        X = np.log1p(np.abs(X)) * np.sign(X)
    mean, scale = np.asarray(prm[0]), np.asarray(prm[1])
    return (X - mean) / scale


def _fit_pca(Z):
    if Z.shape[1] <= 3:
        return Z.shape[1], np.zeros(Z.shape[1]), np.eye(Z.shape[1])
    p = PCA(random_state=42).fit(Z)
    cum = np.cumsum(p.explained_variance_ratio_)
    n = max(2, min(8, int(np.searchsorted(cum, 0.95) + 1), Z.shape[1]))
    return n, p.mean_[:].copy(), p.components_[:n].copy()


def _pca_transform(Z, d_pca, pmean, pcomp):
    if d_pca == Z.shape[1] and np.allclose(pcomp, np.eye(d_pca)):
        return Z - pmean
    return (Z - pmean) @ pcomp.T


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


def _pipeline(Z_tr, Z_te, ds):
    """Full ensemble on the pooled set. Returns (labels_pool, info)."""
    n_tr = len(Z_tr)
    Z = np.vstack([Z_tr, Z_te]) if n_tr else Z_te
    n = len(Z)
    k0 = K_PRIOR.get(ds, 3)
    ks = sorted({max(2, k0 - 1), k0, min(k0 + 1, max(2, n // 5))})

    # --- choose k: prior first; override only on a strong silhouette signal ---
    sils = {}
    for k in ks:
        sils[k] = _sil(Z, KMeans(n_clusters=k, n_init=15, random_state=42).fit(Z).labels_)
    best_k = max(ks, key=lambda k: -sils[k])
    if best_k != k0 and sils[best_k] - sils[k0] < 0.04:
        best_k = k0  # e.g. keep ring+center as 2, don't split the ring
    best_k = min(best_k, max(2, n // 3))

    # --- stability probe + density-based core set (on pooled data) ---
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

    # --- DBSCAN with auto eps (confirming candidate only) ---
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

    # --- HDBSCAN (confirming candidate only) ---
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
    i = int(np.argmax(score))
    labels = np.asarray(cands[i][1], dtype=int)
    info = {
        'k_prior': k0,
        'k_used': best_k,
        'sil_by_k': {k: round(float(s), 3) for k, s in sils.items()},
        'km_stability': round(stab, 3),
        'core_frac_removed': round(frac_rm, 3),
        'fit_set': 'core' if use_core else 'all',
        'pooled': int(n_tr),
        'methods': {name: round(float(s), 3) for (name, _), s in zip(cands, score)},
        'winner': cands[i][0],
    }

    # ds16/ds23: ring + central blob geometry beats arc-splitting
    if best_k == 2 and ds in RING_DS:
        rb = _ring_blob_labels(Z)
        if rb is not None:
            info['winner'] = 'ring/blob'
            info['methods'] = {**info['methods'],
                               'ring/blob': round(_sil(Z, rb), 3)}
            return rb, info
    return labels, info


def cluster(X, dataset_id=None):
    """Return labels only (grading API). Use explain() for the full report."""
    lab, _ = _run(X, dataset_id)
    return lab


def explain(X, dataset_id=None):
    """Same pipeline as cluster(); returns dict with 'labels' and diagnostics."""
    lab, info = _run(X, dataset_id)
    out = {'labels': lab}
    out.update(info)
    return out


def _run(X, dataset_id=None):
    Xc = _clean(X)
    n = len(Xc)
    ds = str(dataset_id or "").lower().strip("'\" ")
    art = ART.get(ds) if ART is not None else None
    if art is not None and Xc.shape[1] == art['d_in']:
        # transform test points into the train's Z-space with saved parameters
        Z0 = _apply_norm(Xc, art['norm'], art['nprm'])
        Z0 = np.where(np.isfinite(Z0), Z0, 0.0)
        Z_te = _pca_transform(Z0, art['d_pca'], np.asarray(art['pmean']), np.asarray(art['pcomp']))
        Z_tr = _uhex(art['ztr'], (art['n_tr'], art['d_pca']))
    else:
        # fallback: unknown dataset -> refit locally, cluster the given points only
        method = NORM_PRIOR.get(ds, "minmax")
        Z0 = _apply_norm(Xc, method, _fit_norm_params(Xc, method))
        Z0 = np.where(np.isfinite(Z0), Z0, 0.0)
        d_pca, pmean, pcomp = _fit_pca(Z0)
        Z_te = _pca_transform(Z0, d_pca, pmean, pcomp)
        Z_tr = np.zeros((0, d_pca))
    if n < 4 and len(Z_tr) < 4:
        lab = np.zeros(n, dtype=int)
        info = {'k_prior': None, 'k_used': 1, 'sil_by_k': {},
                'km_stability': 1.0, 'core_frac_removed': 0.0, 'fit_set': 'all',
                'pooled': int(len(Z_tr)), 'methods': {'constant': 0.0}, 'winner': 'constant'}
        return lab, info
    labels_pool, info = _pipeline(Z_tr, Z_te, ds)
    return np.asarray(labels_pool[len(Z_tr):], dtype=int), info


# --- ds16/ds23: ellipse ring + central blob fix (validated in sol_16_23/) ---
# Standard k=2 clusterers split these datasets into arcs; here we fit an
# ellipse-aware radial boundary in the top-2 Z dimensions and label
# inside/outside instead. Fitted on pooled data at call time, so it is more
# stable than fitting on ~250 test points alone.


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


def _ring_blob_labels(Z):
    """Reference ring/blob split in Z[:, :2]; None if the fit degenerates."""
    p = Z[:, :2]
    c, th, R = _ellipse_boundary(p)
    lab = _inside_boundary(p, c, th, R).astype(int)
    frac = float(lab.mean())
    if not 0.15 < frac < 0.85:                  # degenerate fit -> fall back
        return None
    return lab
