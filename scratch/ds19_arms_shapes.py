"""diagnostic: show each arm's point cloud alone + the current greedy-walk path order"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def arc_order(P):
    cen = P.mean(axis=0)
    d = ((P - cen)**2).sum(axis=1)
    start = int(d.argmax())
    unvisited = set(range(len(P))); order = [start]; unvisited.remove(start); cur = start
    while unvisited:
        dd = np.array([((P[i]-P[cur])**2).sum() for i in unvisited])
        nxt = list(unvisited)[int(dd.argmin())]
        order.append(nxt); unvisited.remove(nxt); cur = nxt
    return np.array(order)

fig, axes = plt.subplots(1, 2, figsize=(15.5, 7))
for ax, (name, m, col) in zip(axes, (('YELLOW', yellow, '#f4c430'), ('RED', red, '#e63946'))):
    P = Xs[m][:, [ix6, ix7]]
    ax.scatter(P[:, 0], P[:, 1], s=8, c=col, alpha=.8)
    order = arc_order(P)
    ax.plot(P[order, 0], P[order, 1], 'b-', lw=.6, alpha=.6, label='greedy walk order')
    # mark the two ends of the walk
    ax.scatter([P[order[0],0], P[order[-1],0]], [P[order[0],1], P[order[-1],1]],
               c='black', s=60, zorder=5, marker='s', label='walk ends')
    ax.set_title(f'{name} arm (n={len(P)}) + current walk path')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('scratch/ds19_arms_shapes.png', dpi=110)
print('saved scratch/ds19_arms_shapes.png')
