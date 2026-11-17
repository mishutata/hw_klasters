import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score
import sys; sys.path.insert(0, 'my_solution_final')
import solution as S

df = pd.read_csv('public_data/train.csv')
xcols = [c for c in df.columns if c.startswith('x')]
rng = np.random.RandomState(0)
print(f"{'ds':5} {'n':>5} {'kP':>2} {'kU':>2} {'stab':>5} {'core%':>6} {'winner':18} {'dropA':>6} {'dropB':>6} {'avg':>6}")
for ds, g in df.groupby('dataset_id'):
    X = g[xcols].values.astype(float)
    lab_full = S.cluster(X, ds)
    i = S.explain(X, ds)
    # dropA: fit on 50% (points idx1), compare its labels to full-run labels on those same points
    ari = []
    for t in range(2):
        n = len(X); m = n//2
        idx = rng.permutation(n)[:m]
        sub = X[idx]
        lab_sub = S.cluster(sub, ds)
        ari.append(adjusted_rand_score(lab_full[idx], lab_sub))
    print(f"{ds:5} {len(X):>5} {i['k_prior']:>2} {i['k_used']:>2} {i['km_stability']:>5.2f} "
          f"{i['core_frac_removed']*100:>5.1f}% {i['winner']:18} {ari[0]:>6.3f} {ari[1]:>6.3f} {np.mean(ari):>6.3f}")
