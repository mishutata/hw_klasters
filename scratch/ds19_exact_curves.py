"""ds19: exact formulas for the two visible lines (yellow = upper-left arm, red = lower-right arm)
in the (x6, x7) plane. Arms are hooks -> parametric fits with arc-length parameter + implicit conic.
Outputs: coefficients printed (std + raw coords), overlay figure dev/v10/ds19_red_yellow_formulas.png"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering

df = pd.read_csv('public_data/train.csv')
d19 = df[df.dataset_id == 'ds19']
Xraw = d19[[c for c in d19.columns if c.startswith('x')]].values.astype(float)
mask = ~np.all(np.isnan(Xraw), axis=0)
cols = [c for c, m in zip([c for c in d19.columns if c.startswith('x')], mask) if m]
X = Xraw[:, mask].copy()
ni = np.where(np.isnan(X))
if len(ni[0]): X[ni] = np.take(np.nanmean(X, axis=0), ni[1])
mu = X.mean(axis=0); sd = X.std(axis=0)
Xs = (X - mu) / sd
ix6, ix7 = cols.index('x6'), cols.index('x7')

Z4 = PCA(4, random_state=42).fit_transform(Xs)
lab = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=20,
                         assign_labels='kmeans', random_state=42).fit_predict(Z4)
m0, m1 = lab == 0, lab == 1
if Xs[m0][:, ix6].mean() < Xs[m1][:, ix6].mean():
    yellow, red = m0, m1
else:
    yellow, red = m1, m0
print(f"yellow n={yellow.sum()} (x6bar {Xs[yellow][:,ix6].mean():+.2f}, x7bar {Xs[yellow][:,ix7].mean():+.2f})")
print(f"red    n={red.sum()}  (x6bar {Xs[red][:,ix6].mean():+.2f}, x7bar {Xs[red][:,ix7].mean():+.2f})")

def arc_order(P):
    cen = P.mean(axis=0)
    d = ((P - cen)**2).sum(axis=1)
    start = int(d.argmax())
    unvisited = set(range(len(P))); order = [start]; unvisited.remove(start); cur = start
    while unvisited:
        dd = np.array([((P[i]-P[cur])**2).sum() for i in unvisited])
        nxt = list(unvisited)[int(dd.argmin())]
        order.append(nxt); unvisited.remove(nxt); cur = nxt
    s = np.zeros(len(order))
    for k in range(1, len(order)):
        s[k] = s[k-1] + np.linalg.norm(P[order[k]] - P[order[k-1]])
    return np.array(order), s / s[-1]

def fit_param(P, deg):
    """parametric polynomial p(t) = sum_d C[d, j] t^d  (j=0: x6s, j=1: x7s), t in [0,1] arc-length"""
    order, t = arc_order(P)
    T = np.vander(t, deg+1, increasing=True)          # (n, deg+1): [1, t, ..., t^deg]
    C = np.linalg.lstsq(T, P[order], rcond=None)[0]   # (deg+1, 2)
    tt = np.linspace(0, 1, 400)
    curve = np.vander(tt, deg+1, increasing=True) @ C # (400, 2)
    pts = T @ C                                       # (n, 2) fitted at own t
    rms = float(np.sqrt(((pts - P[order])**2).sum(axis=1)).mean())
    return C, rms, curve

def fit_conic(P):
    x, y = P[:, 0], P[:, 1]
    D = np.stack([x**2, x*y, y**2, x, y, np.ones(len(x))], axis=1)
    c = np.linalg.svd(D, full_matrices=False)[2][-1].copy()
    c = c / max(abs(c).max(), 1e-300)                # normalize: largest |coef| = 1
    F = D @ c
    gx = 2*c[0]*x + c[1]*y + c[3]; gy = c[1]*x + 2*c[2]*y + c[4]
    rms = float(np.abs(F).mean() / np.sqrt(gx**2 + gy**2).mean())
    return c, rms

def fmt_poly(coefs):
    """coefs: list for x(t) = c0 + c1 t + ... ; returns pretty string"""
    terms = []
    for d, a in enumerate(coefs):
        if abs(a) < 5e-4 and d > 0: continue
        tv = {0: '', 1: 't', 2: 't^2', 3: 't^3'}[d]
        terms.append(f"{a:+.5f}{tv}" if d else f"{a:+.5f}")
    return " ".join(terms)

fig, ax = plt.subplots(figsize=(7.5, 7))
ax.scatter(Xs[:, ix6], Xs[:, ix7], s=9, c='#9ecae1', alpha=.8, edgecolors='none')
summary = {}
for name, m, col in (('YELLOW', yellow, '#f4c430'), ('RED', red, '#e63946')):
    P = Xs[m][:, [ix6, ix7]]
    print(f"\n===== {name} arm (std coords x6s, x7s) =====")
    for deg in (2, 3):
        C, rms, curve = fit_param(P, deg)
        print(f"parametric deg-{deg}: rms={rms:.4f}")
        if name == 'YELLOW':
            ax.plot(curve[:, 0], curve[:, 1], color=col, lw=2.2, alpha=.95,
                    label=f'{name} deg-{deg} (rms {rms:.3f})')
        if deg == 2:
            Cq = C
            print(f"  x6s(t) = {fmt_poly(C[:,0])}")
            print(f"  x7s(t) = {fmt_poly(C[:,1])}")
    c, crms = fit_conic(P)
    disc = c[1]**2 - 4*c[0]*c[2]
    kind = 'ellipse' if disc < 0 else ('parabola' if abs(disc) < 1e-6 else 'hyperbola')
    print(f"conic [{kind}]: {c[0]:+.4f}*x6s^2 {c[1]:+.4f}*x6s*x7s {c[2]:+.4f}*x7s^2 "
          f"{c[3]:+.4f}*x6s {c[4]:+.4f}*x7s {c[5]:+.4f} = 0   (geo-rms={crms:.4f})")
    summary[name] = (Cq, crms)

# raw-coordinate version of the deg-2 parametric fits: x6(t)=mu6+sd6*x6s(t) etc.
print("\n===== RAW coordinates (x6, x7 as in train.csv; t in [0,1]) =====")
for name in ('YELLOW', 'RED'):
    Cq = summary[name][0]
    for j, feat in ((0, 'x6'), (1, 'x7')):
        a = mu[ix6 if j == 0 else ix7] + sd[ix6 if j == 0 else ix7] * Cq[:, j]
        print(f"  {name} {feat}(t) = {fmt_poly(a)}")

ax.set_xlabel('x6 (std)'); ax.set_ylabel('x7 (std)')
ax.legend(fontsize=8, loc='upper right')
ax.set_title('ds19: fitted formulas for the yellow and red lines')
plt.tight_layout()
plt.savefig('dev/v10/ds19_red_yellow_formulas.png', dpi=130)
print('\nsaved dev/v10/ds19_red_yellow_formulas.png')
