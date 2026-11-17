"""ds19 parabola theory — where do the two parabolas live in other coordinates?

After _clean, ds19 has 8 real features (x1..x8; x9..x14 are all-NaN columns dropped).
For every pair of standardized features we run the SAME two-quadratic EM as the
probe (both axis orientations, keep best rms) and score:
  - ARI vs the canonical para2 partition (from (x1s, x5s))
  - rms_sum of the two quadratic fits in that view
Outputs:
  dev/v10/ds19_parab_coord_heatmap.png — upper-triangular ARI map over all 28 pairs
  dev/v10/ds19_parab_coord_panels.png  — story panels for top alternative planes
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

df = pd.read_csv('public_data/train.csv')
d19 = df[df.dataset_id == 'ds19']
Xraw = d19[[c for c in d19.columns if c.startswith('x')]].values.astype(float)
cols_all = [c for c in d19.columns if c.startswith('x')]

# same _clean as the probe: drop all-NaN columns, mean-impute the rest
mask = ~np.all(np.isnan(Xraw), axis=0)
cols = [c for c, m in zip(cols_all, mask) if m]
X = Xraw[:, mask].copy()
nan_idx = np.where(np.isnan(X))
if len(nan_idx[0]):
    X[nan_idx] = np.take(np.nanmean(X, axis=0), nan_idx[1])
Xs = StandardScaler().fit_transform(X)
n, p = Xs.shape
print(f"{n} points, {p} features: {cols}")

def em_two_quadratics(Q):
    """Same EM as probes/ds19parab/_parabola2_labels. Q is (n,2). Returns
    (labels, fits[(a,b,c) x2], rms_sum) for the given axis order."""
    best = None
    for s in (0, 1, 2, 3, 4):
        lab = KMeans(2, n_init=5, random_state=s).fit_predict(Q)
        small = False
        for _ in range(60):
            cs = []
            for i in (0, 1):
                m = lab == i
                if m.sum() < 8:
                    small = True
                    break
                A = np.vstack([Q[m, 0]**2, Q[m, 0], np.ones(m.sum())]).T
                c, *_ = np.linalg.lstsq(A, Q[m, 1], rcond=None)
                cs.append(c)
            if small:
                break
            r = np.stack([np.abs(Q[:, 1] - (cs[i][0]*Q[:, 0]**2 + cs[i][1]*Q[:, 0] + cs[i][2])) for i in (0, 1)])
            nl = r.argmin(0)
            if (nl == lab).all():
                break
            lab = nl
        if small:
            continue
        rms = 0.0
        for i in (0, 1):
            m = lab == i
            A = np.vstack([Q[m, 0]**2, Q[m, 0], np.ones(m.sum())]).T
            c, *_ = np.linalg.lstsq(A, Q[m, 1], rcond=None)
            rms += float(np.sqrt(np.mean((Q[m, 1] - A@c)**2)))
        bal = min(float((lab == 0).mean()), float((lab == 1).mean()))
        if best is None or (rms, -bal) < best[0]:
            best = ((rms, -bal), lab.copy(), cs)
    return best[1], best[2], best[0][0]

# canonical partition: (x1, x5) -> indices 0 and 4
can_lab, can_fits, can_rms = em_two_quadratics(Xs[:, [0, 4]])
print(f"canonical (x1,x5): rms_sum={can_rms:.3f}")

# --- all-pairs: EM in both orientations, keep best ----------------------------
res = {}
for i in range(p):
    for j in range(i + 1, p):
        Q = Xs[:, [i, j]]
        lab_a, fits_a, rms_a = em_two_quadratics(Q)
        lab_b, fits_b, rms_b = em_two_quadratics(Q[:, ::-1])
        if rms_a <= rms_b:
            res[(cols[i], cols[j])] = (adjusted_rand_score(lab_a, can_lab), rms_a, lab_a, fits_a, 'y~f(x)')
        else:
            res[(cols[i], cols[j])] = (adjusted_rand_score(lab_b, can_lab), rms_b, lab_b, fits_b, 'x~f(y)')

names = cols
M = np.zeros((p, p))
for i in range(p):
    for j in range(i + 1, p):
        M[i, j] = res[(names[i], names[j])][0]
        M[j, i] = M[i, j]

fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(np.triu(M, k=1), cmap='viridis', vmin=0, vmax=1, aspect='equal')
for i in range(p):
    for j in range(i + 1, p):
        v = M[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                color='white' if v > 0.45 else 'black')
    ax.text(-0.55, i, names[i], ha='right', va='center', fontsize=11)
    ax.text(i, -0.55, names[i], ha='center', va='bottom', fontsize=11, rotation=45)
ax.set_xticks(range(p)); ax.set_yticks(range(p))
ax.set_title('ds19: ARI of per-pair two-parabola EM vs canonical (x1,x5) partition\n(high = the same two arcs are visible in this coordinate plane)', fontsize=12)
fig.colorbar(im, ax=ax, label='ARI vs canonical para2 split')
plt.tight_layout()
plt.savefig('dev/v10/ds19_parab_coord_heatmap.png', dpi=150)
print("saved dev/v10/ds19_parab_coord_heatmap.png")

# --- story panels: canonical + top-3 alternative planes -----------------------
top = sorted(((k, v) for k, v in res.items() if k != ('x1', 'x5')), key=lambda kv: -kv[1][0])[:3]
panels = [('x1', 'x5')] + [k for k, _ in top]
C0, C1 = '#f4c430', '#6a3d9a'
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('ds19 — the two parabolas seen in other coordinate planes (same EM procedure)', fontsize=15)
for ax, (ci, cj) in zip(axes.ravel(), panels):
    i, j = names.index(ci), names.index(cj)
    Q = Xs[:, [i, j]]
    if (ci, cj) == ('x1', 'x5'):
        ari, rms_sum, lab, fits, orient = 1.0, can_rms, can_lab, can_fits, 'y~f(x)'
    else:
        ari, rms_sum, lab, fits, orient = res[(ci, cj)]
    for k in (0, 1):
        m = lab == k
        ax.scatter(Q[m, 0], Q[m, 1], s=9, c=(C0, C1)[k], alpha=0.75, edgecolors='none')
        if orient == 'y~f(x)':
            xs = np.linspace(Q[:, 0].min(), Q[:, 0].max(), 400)
            ax.plot(xs, fits[k][0]*xs**2 + fits[k][1]*xs + fits[k][2], '-', lw=2.2, c=(C0, C1)[k])
        else:  # x~f(y): draw parametrically via the fit (swap axes)
            ys = np.linspace(Q[:, 1].min(), Q[:, 1].max(), 400)
            ax.plot(fits[k][0]*ys**2 + fits[k][1]*ys + fits[k][2], ys, '-', lw=2.2, c=(C0, C1)[k])
    tag = 'canonical' if (ci, cj) == ('x1', 'x5') else f"ARI vs canonical = {ari:.3f}"
    ax.set_title(f'({ci}s, {cj}s): rms_sum={rms_sum:.2f}, {tag}')
    ax.set_xlabel(f'{ci} (std)'); ax.set_ylabel(f'{cj} (std)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('dev/v10/ds19_parab_coord_panels.png', dpi=150)
print("saved dev/v10/ds19_parab_coord_panels.png")

print("\ntop planes by ARI vs canonical:")
for k, v in sorted(res.items(), key=lambda kv: -kv[1][0])[:8]:
    print(f"  {k[0]}-{k[1]}: ARI={v[0]:.3f} rms_sum={v[1]:.2f} orient={v[4]}")
