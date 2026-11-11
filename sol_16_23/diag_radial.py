import sys; sys.path.insert(0, r"..\my_solver")
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from solution import _clean
df = pd.read_csv(r"..\public_data\train.csv")

def boundary(p):
    c0 = p.mean(0)
    d0 = np.linalg.norm(p - c0, axis=1)
    outer = d0 > np.quantile(d0, 0.55)
    a = np.arctan2(p[outer,1]-c0[1], p[outer,0]-c0[0])
    r = d0[outer]
    nb = 36; th = (np.arange(nb)+0.5)*(2*np.pi/nb) - np.pi
    R = np.array([np.median(r[(a>=t-dt/2)&(a<t+dt/2)]) if ((a>=t-dt/2)&(a<t+dt/2)).sum()>=3 else np.nan for t,dt in zip(th, np.full(nb, 2*np.pi/nb))])
    idx = np.arange(nb); Rf = np.interp(idx, np.concatenate([idx-nb, idx, idx+nb]), np.concatenate([R,R,R]))
    return c0, th, Rf

for ds in ("ds16", "ds23"):
    X = df[df.dataset_id == ds].drop(columns=["id","dataset_id"]).values.astype(float)
    p2 = PCA(2, random_state=42).fit_transform(StandardScaler().fit_transform(_clean(X)))
    c, th, R = boundary(p2)
    d = np.linalg.norm(p2 - c, axis=1)
    a = (np.arctan2(p2[:,1]-c[1], p2[:,0]-c[0]) + np.pi) % (2*np.pi) - np.pi
    t_ext = np.concatenate([th-2*np.pi, th, th+2*np.pi]); R_ext = np.concatenate([R,R,R])
    q = d / np.interp(a, t_ext, R_ext)
    print(f"== {ds}  q=d/Rloc quantiles .05-.95:", np.round(np.quantile(q, [i/20 for i in range(1,20)]), 2))
    h, e = np.histogram(q, bins=40, range=(0, 1.6))
    print("   hist q:0..1.6:", h)
