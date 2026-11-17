import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

df = pd.read_csv('public_data/train.csv')
K_PRIOR = {'ds00':2,'ds01':3,'ds02':2,'ds03':2,'ds04':3,'ds05':3,'ds06':3,'ds07':3,
           'ds08':2,'ds09':3,'ds10':3,'ds11':4,'ds12':2,'ds13':2,'ds14':2,'ds15':4,
           'ds16':2,'ds17':2,'ds18':2,'ds19':4,'ds20':4,'ds21':3,'ds22':2,'ds23':2}
print('ds      d  k_prior  BIC full-cov (k=2..6)          argmin')
for ds, g in sorted(df.groupby('dataset_id')):
    X = g.drop(columns=['id', 'dataset_id']).values.astype(float)
    keep = np.nanmean(np.isnan(X), axis=0) < 0.9
    X = X[:, keep]
    sc = StandardScaler().fit_transform(X)
    p = min(8, max(2, sc.shape[1]))
    evr = PCA(n_components=p, svd_solver='full').fit(sc).explained_variance_ratio_
    d = int(np.searchsorted(np.cumsum(evr), 0.95)) + 1
    Z = PCA(n_components=d, random_state=42).fit_transform(sc)
    bics = []
    for k in range(2, 7):
        gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=3,
                              random_state=42, reg_covar=1e-4).fit(Z)
        bics.append(gmm.bic(Z))
    best = int(np.argmin(bics)) + 2
    ks = K_PRIOR[ds]
    mark = 'OK ' if best == ks else 'DIFF'
    print(f'{ds} {d:3d} {ks:6d}   ' + ' '.join(f'{b:8.0f}' for b in bics) + f'  k={best} {mark}')
