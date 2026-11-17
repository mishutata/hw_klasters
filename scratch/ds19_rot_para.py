"""ds19: rotate (x6s,x7s) by theta, then slice with TWO parabolas in that 2-D frame.
Each point -> nearest of the two fitted parabolas. Compare vs champion spectral labels."""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score

df = pd.read_csv('public_data/train.csv')
d19 = df[df.dataset_id == 'ds19']
Xraw = d19[[c for c in d19.columns if c.startswith('x')]].values.astype(float)
mask = ~np.all(np.isnan(Xraw), axis=0)
cols = [c for c, m in zip([c for c in d19.columns if c.startswith('x')], mask) if m]
X = Xraw[:, mask].copy()
ni = np.where(np.isnan(X))
if len(ni[0]): X[ni] = np.take(np.nanmean(X, axis=0), ni[1])
Xs = StandardScaler().fit_transform(X)
ix6, ix7 = cols.index('x6'), cols.index('x7')
Q = Xs[:, [ix6, ix7]]

Z4 = PCA(4, random_state=42).fit_transform(Xs)
lab_champ = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=20,
                               assign_labels='kmeans', random_state=42).fit_predict(Z4)

def rot(P, deg):
    th = np.radians(deg); c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return P @ R.T

def fit_curve(P):
    """best of y=f(x), x=g(y), PCA-frame quadratic; returns zero-arg sampler + rms"""
    forms = []
    a, b, c = np.polyfit(P[:, 0], P[:, 1], 2)
    lo, hi = P[:, 0].min(), P[:, 0].max()
    def s1(n=400):
        xx = np.linspace(lo, hi, n); return np.stack([xx, np.polyval([a, b, c], xx)], axis=1)
    forms.append((s1, float(np.sqrt(np.mean((P[:, 1] - (a*P[:, 0]**2 + b*P[:, 0] + c))**2))), 'y=f(x)'))
    a, b, c = np.polyfit(P[:, 1], P[:, 0], 2)
    lo, hi = P[:, 1].min(), P[:, 1].max()
    def s2(n=400):
        yy = np.linspace(lo, hi, n); return np.stack([np.polyval([a, b, c], yy), yy], axis=1)
    forms.append((s2, float(np.sqrt(np.mean((P[:, 0] - (a*P[:, 1]**2 + b*P[:, 1] + c))**2))), 'x=g(y)'))
    m = P.mean(0); Pc = P - m
    w, V = np.linalg.eigh(np.cov(Pc.T)); u_ax = V[:, np.argmax(w)]; v_ax = np.array([-u_ax[1], u_ax[0]])
    u, v = Pc @ u_ax, Pc @ v_ax
    a, b, c = np.polyfit(u, v, 2)
    lo, hi = u.min(), u.max()
    def s3(n=400):
        uu = np.linspace(lo, hi, n); return m + np.outer(uu, u_ax) + np.outer(np.polyval([a, b, c], uu), v_ax)
    forms.append((s3, float(np.sqrt(np.mean((v - (a*u**2 + b*u + c))**2))), 'PCA'))
    f, rms, name = min(forms, key=lambda t: t[1])
    return f, rms, name

def dist_to(Pm, cv):
    return np.linalg.norm(Pm[:, None, :] - cv[None, :, :], axis=2).min(axis=1)

def _em_once(P, lab, iters=30):
    for it in range(iters):
        fits = []
        for g in (0, 1):
            Pm = P[lab == g]
            if len(Pm) < 5:
                fits.append(None); continue
            f, rms, name = fit_curve(Pm)
            # robust refit on lower-60% distance inliers
            d = dist_to(Pm, f())
            keep = d <= max(np.quantile(d, 0.6), 1e-6)
            if keep.sum() >= 5:
                f2, rms2, name2 = fit_curve(Pm[keep])
                if rms2 <= rms: f = f2
            fits.append(f)
        Dm = np.stack([dist_to(P, f()) if f is not None else np.full(len(P), np.inf) for f in fits], axis=1)
        lab_new = Dm.argmin(axis=1)
        # guard: empty cluster -> keep old
        if len(np.unique(lab_new)) < 2: break
        if np.array_equal(lab_new, lab): break
        lab = lab_new
    return lab, fits

def em_two_paras(P, seed=42, n_init=10, iters=30):
    """multi-restart EM; keep partition with lowest total distance-to-own-curve"""
    best = None
    for r in range(n_init):
        lab0 = KMeans(2, n_init=1, random_state=seed + 7*r).fit_predict(P)
        lab, fits = _em_once(P, lab0, iters)
        if len(np.unique(lab)) < 2: continue
        Dm = np.stack([dist_to(P[lab == g], fits[g]()).mean() for g in (0, 1)])
        tot = sum(dist_to(P[lab == g], fits[g]())[np.ones(len(P[lab == g]), bool)].sum() for g in (0, 1))
        if best is None or tot < best[2]:
            best = (lab, fits, tot)
    return best[0], best[1]

print(f"{'deg':>5} {'ARI_vs_champ':>13} {'bal(y/r)':>10} {'mean d own':>11}  stable(sub ARIs)")
for deg in (0, -20, -15, -10, -5, 5, 10, 15, 20):
    Pr = rot(Q, deg)
    lab, fits = em_two_paras(Pr)
    ari = adjusted_rand_score(lab_champ, lab)
    d_own = np.array([dist_to(Pr[lab == g], fits[g]()).mean() for g in (0, 1)]) if all(f is not None for f in fits) else [np.nan]*2
    st = []
    for s in (1, 2, 3):
        idx = np.sort(np.random.RandomState(s).choice(len(Q), int(len(Q)*0.4), replace=False))
        l1, _ = em_two_paras(Pr[idx], seed=s)
        st.append(round(adjusted_rand_score(lab[idx], l1), 3))
    print(f"{deg:>5} {ari:>13.3f} {int((lab==0).sum())}/{int((lab==1).sum()):>9} "
          f"{np.mean(d_own):>11.3f}  {st}")
