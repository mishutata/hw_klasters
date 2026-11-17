import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
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
Z4 = PCA(4, random_state=42).fit_transform(Xs)
lab = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=20,
                         assign_labels='kmeans', random_state=42).fit_predict(Z4)
ix = {c: i for i, c in enumerate(cols)}
Q = Xs[:, [ix['x6'], ix['x7']]]

for g in (0, 1):
    xy = Q[lab == g]
    cy = np.polyfit(xy[:, 0], xy[:, 1], 2)
    cx = np.polyfit(xy[:, 1], xy[:, 0], 2)
    ry = 1 - np.sum((xy[:,1]-np.polyval(cy, xy[:,0]))**2)/np.sum((xy[:,1]-xy[:,1].mean())**2)
    rx = 1 - np.sum((xy[:,0]-np.polyval(cx, xy[:,1]))**2)/np.sum((xy[:,0]-xy[:,0].mean())**2)
    print(f"lobe {g}: n={len(xy)}, x6 range [{xy[:,0].min():.2f},{xy[:,0].max():.2f}], "
          f"x7 range [{xy[:,1].min():.2f},{xy[:,1].max():.2f}], R2(y~x)={ry:.3f}, R2(x~y)={rx:.3f}")

# cut at the neck: x6 < -0.09
cut = (Q[:, 0] < -0.09).astype(int)
print("ARI(champion, cut x6<-0.09):", round(adjusted_rand_score(lab, cut), 4))

# nearest lobe by euclidean distance to lobe point sets (subsampled k=1)
rng = np.random.RandomState(0)
lab2 = np.zeros(len(Q), int)
for i in range(len(Q)):
    d = []
    for g in (0, 1):
        idx = rng.choice(np.where(lab == g)[0], size=min(300, (lab==g).sum()), replace=False)
        d.append(((Q[idx] - Q[i])**2).sum(axis=1).min())
    lab2[i] = int(d[0] < d[1])
print("ARI(champion, nearest-lobe euclidean):", round(adjusted_rand_score(lab, lab2), 4))

# where do the two lobes overlap in x6?
o = np.sum((Q[lab==0,0].min() <= Q[lab==1,0]) & (Q[lab==1,0].min() <= Q[lab==0,0].max()))
print("x6 overlap:", max(Q[lab==0,0].min(), Q[lab==1,0].min()), "<=", min(Q[lab==0,0].max(), Q[lab==1,0].max()))
