import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'my_solver'); sys.path.insert(0,'analysis')
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.spatial import cKDTree

df = pd.read_csv('public_data/train.csv')
def prep(ds):
    d = df[df['dataset_id'].str.replace('_aug','')==ds]
    cols=[c for c in d.columns if c.startswith('x')]
    X=d[cols].to_numpy(float)
    keep=~np.all(np.isnan(X),axis=0); X=X[:,keep]
    for j in range(X.shape[1]):
        b=np.isnan(X[:,j])
        if b.any(): X[b,j]=np.nanmean(X[:,j][~b])
    Z=StandardScaler().fit_transform(X); Z=np.where(np.isfinite(Z),Z,0.0)
    if X.shape[1]>3:
        p=PCA(random_state=42).fit(Z); cum=np.cumsum(p.explained_variance_ratio_)
        n=max(2,min(8,int(np.searchsorted(cum,0.95)+1),X.shape[1]))
        Z=PCA(n_components=n,random_state=42).fit_transform(Z)
    return Z

Z = prep('ds01')
tree=cKDTree(Z); d,_=tree.query(Z,k=6); d5=d[:,5]
core = d5 <= max(np.quantile(d5,0.95), np.median(d5)*2)
km_all  = KMeans(3,n_init=15,random_state=42).fit(Z)
Zc=Z[core]; km_core=KMeans(3,n_init=15,random_state=42).fit(Zc)
lab_trim=km_core.predict(Z)
print("ARI all-vs-trim %.3f | sil all %.3f core %.3f" % (adjusted_rand_score(km_all.labels_,lab_trim), silhouette_score(Z,km_all.labels_), silhouette_score(Zc,km_core.labels_)))
cents=km_core.cluster_centers_; D=((Z[:,None,:]-cents[None,:,:])**2).sum(-1); near=D.argmin(1)
print("removed total:", (~core).sum())
idx=np.arange(len(Z))
for c in range(3):
    mk=core&(near==c); mr=(~core)&(near==c)
    dk=D[idx[mk], near[mk]].mean() if mk.sum() else float('nan')
    dr=D[idx[mr], near[mr]].mean() if mr.sum() else float('nan')
    print(f"c{c}: kept n={mk.sum()} d2={dk:.1f} | removed n={mr.sum()} d2={dr:.1f}")
for k in (2,3,4,5):
    print(f"k={k}: sil_all={silhouette_score(Z,KMeans(k,n_init=15,random_state=42).fit_predict(Z)):.3f}  sil_core={silhouette_score(Zc,KMeans(k,n_init=15,random_state=42).fit_predict(Zc)):.3f}")
