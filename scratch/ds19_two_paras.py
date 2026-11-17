"""ds19: draw 2 parabolas (yellow = upper-left arm, red = lower-right) as in user's screenshot.
Every point is assigned to the NEAREST parabola (no spectral clustering).
Outputs: formulas (std + raw), figure dev/v10/ds19_red_yellow_formulas.png"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
mu = X.mean(axis=0); sd = X.std(axis=0)
Xs = (X - mu) / sd
ix6, ix7 = cols.index('x6'), cols.index('x7')
Q = Xs[:, [ix6, ix7]]

# champion labels ONLY for ARI reference (not used for the division)
Z4 = PCA(4, random_state=42).fit_transform(Xs)
lab_champ = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=20,
                               assign_labels='kmeans', random_state=42).fit_predict(Z4)

def fit_parabola(P):
    m = P.mean(axis=0)
    Pc = P - m
    w, V = np.linalg.eigh(np.cov(Pc.T))
    u_ax = V[:, np.argmax(w)]
    v_ax = np.array([-u_ax[1], u_ax[0]])
    u = Pc @ u_ax; v = Pc @ v_ax
    a, b, c = np.polyfit(u, v, 2)
    return dict(m=m, u_ax=u_ax, v_ax=v_ax, abc=(a, b, c), u0=u.min(), u1=u.max())

def para_curve(f, n=400):
    a, b, c = f['abc']
    u = np.linspace(f['u0'], f['u1'], n)
    q = a*u**2 + b*u + c
    return f['m'] + np.outer(u, f['u_ax']) + np.outer(q, f['v_ax'])

def assign_nearest(Qp, fits):
    D = np.stack([np.linalg.norm(Qp[:, None, :] - para_curve(f)[None], axis=2).min(axis=1) for f in fits], axis=1)
    return D.argmin(axis=1), D

# init: KMeans on the (x6s,x7s) plane only (no spectral anywhere in the division)
lab = KMeans(2, n_init=10, random_state=42).fit_predict(Q)
for it in range(15):
    fits = [fit_parabola(Q[lab == g]) for g in (0, 1)]
    lab_new, D = assign_nearest(Q, fits)
    print(f"iter {it}: group sizes {(lab==0).sum():4d} / {(lab==1).sum():4d}", end='')
    if np.array_equal(lab_new, lab):
        print("  (stable)")
        break
    lab = lab_new
else:
    print("  (max iters)")

cens = [Q[lab == g].mean(axis=0) for g in (0, 1)]
gY = int(np.argmin([c[0] - c[1] for c in cens]))   # yellow = upper-left arm (smaller x6bar)
gR = 1 - gY

def fmt_poly(coefs):
    terms = []
    for d, a in enumerate(coefs):
        if abs(a) < 5e-4 and d > 0: continue
        tv = {0: '', 1: 't', 2: 't^2'}[d]
        terms.append(f"{a:+.5f}{tv}" if d else f"{a:+.5f}")
    return " ".join(terms)

print(f"\nYELLOW arm: group {gY}, n={(lab==gY).sum()}")
print(f"RED    arm: group {gR}, n={(lab==gR).sum()}")
print(f"ARI(nearest-parabola division, champion spectral labels):", round(adjusted_rand_score(lab_champ, lab), 4))

fig, ax = plt.subplots(figsize=(7.5, 7))
summary = {}
for name, g, col in (('YELLOW', gY, '#f4c430'), ('RED', gR, '#e63946')):
    f = fits[g]
    Pm = Q[lab == g]
    d_own = np.linalg.norm(Pm[:, None, :] - para_curve(f)[None], axis=2).min(axis=1)
    print(f"\n===== {name} parabola =====")
    a, b, c = f['abc']
    print(f"  in arm frame (u along principal axis): v = {a:+.5f} u^2 {b:+.5f} u {c:+.5f},  u in [{f['u0']:.3f}, {f['u1']:.3f}]")
    print(f"  mean dist of its points to the parabola: {d_own.mean():.3f} std-units")
    # parametric quadratic in t in [0,1] (std coords)
    L = f['u1'] - f['u0']; u0 = f['u0']
    Aq = a*L**2; Bq = 2*a*u0*L + b*L; Cq = a*u0**2 + b*u0 + c
    for j, feat in ((0, 'x6s'), (1, 'x7s')):
        coefs = [f['m'][j] + f['u_ax'][j]*u0 + f['v_ax'][j]*Cq,
                 f['u_ax'][j]*L + f['v_ax'][j]*Bq,
                 f['v_ax'][j]*Aq]
        print(f"  {feat}(t) = {fmt_poly(coefs)}")
    for j, feat in ((0, 'x6'), (1, 'x7')):
        k = ix6 if j == 0 else ix7
        coefs = [mu[k] + sd[k]*(f['m'][j] + f['u_ax'][j]*u0 + f['v_ax'][j]*Cq),
                 sd[k]*(f['u_ax'][j]*L + f['v_ax'][j]*Bq),
                 sd[k]*f['v_ax'][j]*Aq]
        print(f"  {feat}(t) = {fmt_poly(coefs)}   [raw]")
    summary[name] = (g, col)

for name in ('YELLOW', 'RED'):
    g, col = summary[name]
    f = fits[g]
    cv = para_curve(f)
    ax.plot(cv[:, 0], cv[:, 1], color=col, lw=2.6, alpha=.95, zorder=5,
            label=f'{name} parabola')
for name in ('YELLOW', 'RED'):
    g, col = summary[name]
    Pm = Q[lab == g]
    ax.scatter(Pm[:, 0], Pm[:, 1], s=9, c=col, alpha=.85, edgecolors='none', zorder=4,
               label=f'{name} points (n={len(Pm)})')

ax.set_xlabel('x6 (std)'); ax.set_ylabel('x7 (std)')
ax.legend(fontsize=8, loc='upper right')
ax.set_title('ds19: two parabolas as drawn; every point assigned to its nearest parabola (no spectral)')
plt.tight_layout()
plt.savefig('dev/v10/ds19_red_yellow_formulas.png', dpi=130)
print('\nsaved dev/v10/ds19_red_yellow_formulas.png')
