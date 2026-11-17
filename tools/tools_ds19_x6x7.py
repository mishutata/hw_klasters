"""ds19 row4/col5 of the all-pairs grid = (x6, x7) plane.
User sees 2 distinct clusters there -> find the best k=2 split in that plane
and compare with champion q6 partition + canonical para2 membership.
Outputs: dev/v10/ds19_x6x7.png, results printed compactly.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

df = pd.read_csv('public_data/train.csv')
d19 = df[df.dataset_id == 'ds19']
Xraw = d19[[c for c in d19.columns if c.startswith('x')]].values.astype(float)
cols_all = [c for c in d19.columns if c.startswith('x')]
mask = ~np.all(np.isnan(Xraw), axis=0)
cols = [c for c, m in zip(cols_all, mask) if m]
X = Xraw[:, mask].copy()
ni = np.where(np.isnan(X))
if len(ni[0]):
    X[ni] = np.take(np.nanmean(X, axis=0), ni[1])
Xs = StandardScaler().fit_transform(X)
n, p = Xs.shape
i6, i7 = cols.index('x6'), cols.index('x7')
Q = Xs[:, [i6, i7]]

# --- champion q6 ds19 labels (StandardScaler -> PCA(4) -> spectral knn nn=20 k=2)
Zc = PCA(n_components=4, random_state=42).fit_transform(Xs)
lab_champ = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=20,
                               random_state=42, assign_labels='kmeans').fit_predict(Zc)

# --- canonical para2 membership (EM two quadratics in (x1,x5))
def em_two_quadratics(Q):
    best = None
    for s in (0, 1, 2, 3, 4):
        lab = KMeans(2, n_init=5, random_state=s).fit_predict(Q)
        small = False
        for _ in range(60):
            cs = []
            for i in (0, 1):
                m = lab == i
                if m.sum() < 8:
                    small = True; break
                A = np.vstack([Q[m, 0]**2, Q[m, 0], np.ones(m.sum())]).T
                c, *_ = np.linalg.lstsq(A, Q[m, 1], rcond=None)
                cs.append(c)
            if small: break
            r = np.stack([np.abs(Q[:, 1] - (cs[i][0]*Q[:, 0]**2 + cs[i][1]*Q[:, 0] + cs[i][2])) for i in (0, 1)])
            nl = r.argmin(0)
            if (nl == lab).all(): break
            lab = nl
        if small: continue
        rms = 0.0
        for i in (0, 1):
            m = lab == i
            A = np.vstack([Q[m, 0]**2, Q[m, 0], np.ones(m.sum())]).T
            c, *_ = np.linalg.lstsq(A, Q[m, 1], rcond=None)
            rms += float(np.sqrt(np.mean((Q[m, 1] - A@c)**2)))
        bal = min(float((lab == 0).mean()), float((lab == 1).mean()))
        if best is None or (rms, -bal) < best[0]:
            best = ((rms, -bal), lab.copy())
    return best[1]

lab_canon = em_two_quadratics(Xs[:, [0, 4]])

# --- k=2 methods IN the (x6,x7) plane ------------------------------
def stab(lab):
    ari = []
    for s in range(5):
        l2 = KMeans(2, n_init=3, random_state=100+s).fit_predict(Q)
        ari.append(adjusted_rand_score(lab, l2))
    return float(np.mean(ari))

res = []
def add(name, lab):
    res.append((name, adjusted_rand_score(lab, lab_champ), adjusted_rand_score(lab, lab_canon),
                silhouette_score(Q, lab), stab(lab)))

for s in (0, 1, 2, 3, 4):
    add(f'kmeans_s{s}', KMeans(2, n_init=5, random_state=s).fit_predict(Q))
for cov in ('full', 'diag'):
    g = GaussianMixture(2, covariance_type=cov, n_init=5, random_state=0).fit(Q)
    add(f'gmm_{cov}', g.predict(Q))
for nn in (10, 15, 20, 30, 40):
    add(f'spectral_nn{nn}', SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=nn,
                                               random_state=42, assign_labels='kmeans').fit_predict(Q))
# 1D cuts
for ax in (0, 1):
    cut = np.median(Q[:, ax])
    add(f'median_x{6+ax}', (Q[:, ax] > cut).astype(int))

res.sort(key=lambda r: -r[1])
print(f"{'method':14s} {'ARI_champ':>9s} {'ARI_canon':>9s} {'sil':>5s} {'stab':>5s}")
for name, ac, an, sil, st in res:
    print(f"{name:14s} {ac:9.3f} {an:9.3f} {sil:5.2f} {st:5.2f}")

best = max(res, key=lambda r: r[1])
lab_best = KMeans(2, n_init=5, random_state=0).fit_predict(Q) if best[0].startswith('kmeans') else None
# recompute the winner's labels for plotting (deterministic pick: kmeans_s0 family or gmm_full)
name_b = best[0]
if name_b.startswith('kmeans'):
    lab_b = KMeans(2, n_init=5, random_state=int(name_b[-1])).fit_predict(Q)
elif name_b.startswith('gmm'):
    lab_b = GaussianMixture(2, covariance_type=name_b[4:], n_init=5, random_state=0).fit(Q).predict(Q)
else:
    nn = int(name_b.split('nn')[1])
    lab_b = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=nn,
                               random_state=42, assign_labels='kmeans').fit_predict(Q)
print(f"\nbest: {name_b}  sizes={np.bincount(lab_b)}")
print(f"champ vs canon ARI = {adjusted_rand_score(lab_champ, lab_canon):.3f}")

# --- figure ----------------------------------------------------------
C0, C1 = '#f4c430', '#6a3d9a'
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
for ax, lab, ttl in ((axes[0], lab_canon, 'canonical para2 membership (x1,x5) EM'),
                     (axes[1], lab_b, f'best k=2 found IN (x6,x7): {name_b}\nARI vs champion = {adjusted_rand_score(lab_b, lab_champ):.3f}'),
                     (axes[2], lab_champ, 'champion q6 partition (PCA4 + spectral nn20)')):
    for k in (0, 1):
        m = lab == k
        ax.scatter(Q[m, 0], Q[m, 1], s=9, c=(C0, C1)[k], alpha=0.75, edgecolors='none')
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel('x6 (std)'); ax.set_ylabel('x7 (std)')
fig.suptitle('ds19 in the (x6, x7) plane — row4/col5 of the all-pairs grid', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('dev/v10/ds19_x6x7.png', dpi=130)
print('saved dev/v10/ds19_x6x7.png')
