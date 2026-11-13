# -*- coding: utf-8 -*-
"""Per-slice diagnostics for the unstable datasets: which component wobbles?
k window (k_used changing), winner flipping, or the partition itself."""
import importlib.util, os, sys
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("sol5", "my_solution5/solution.py")
sol5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol5)
sys.path.insert(0, os.path.join(os.getcwd(), "dev_v14"))
from stability_analysis import DATA, slices_of

UNSTABLE = ["ds08", "ds10", "ds13", "ds14", "ds15", "ds16", "ds23"]
for ds in UNSTABLE:
    X = DATA[ds]
    lab_full, info_full = sol5._run(X, ds)
    idxs = slices_of(len(X))
    print(f"== {ds}  full: k={info_full['k_used']} winner={info_full['winner']} "
          f"fit={info_full['fit_set']} stab={info_full['km_stability']}")
    for s, ix in enumerate(idxs):
        Xi = X[ix]
        lab_i, info_i = sol5._run(Xi, ds)
        from sklearn.metrics import adjusted_rand_score
        a = adjusted_rand_score(lab_i, lab_full[ix])
        print(f"   slice{s}: k={info_i['k_used']} winner={info_i['winner']:<18} "
              f"fit={info_i['fit_set']:<4} ARI_vs_full={a:.3f}  sil_by_k={info_i['sil_by_k']}")
    print()
