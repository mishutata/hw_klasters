"""Verify my_solution_final: identical to v3 on 22 datasets, ring/blob winner
on ds16/ds23 with ARI ~0.93-0.94 vs the sol_16_23 reference split."""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


V3 = load(r"..\my_solution3\solution.py", "v3mod")
FIN = load(r"..\my_solution_final\solution.py", "finmod")

# --- sol_16_23 ellipse/ring reference (same code as final_compare.py) ---

def ellipse_boundary(p):
    c0 = p.mean(0)
    d0 = np.linalg.norm(p - c0, axis=1)
    outer = d0 >= np.quantile(d0, 0.45)
    ang = np.arctan2(p[:, 1] - c0[1], p[:, 0] - c0[0])
    nb = 36
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    Rb = np.full(nb, np.nan)
    for i, (e, f) in enumerate(zip(edges[:-1], edges[1:])):
        msk = (ang >= e) & (ang < f) & outer
        if msk.any():
            Rb[i] = np.median(d0[msk])
    for i in range(nb):
        if np.isnan(Rb[i]):
            j = (i + 1) % nb
            while np.isnan(Rb[j]):
                j = (j + 1) % nb
            Rb[i] = Rb[j]
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    sm = np.array([sum(k[j] * Rb[(i - j) % nb] for j in range(5)) for i in range(nb)])
    th = (edges[:-1] + edges[1:]) / 2
    return c0, th, sm


def inside_boundary(p, c, th, R, frac=0.80):
    d = np.linalg.norm(p - c, axis=1)
    a = (np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]) + np.pi) % (2 * np.pi) - np.pi
    t_ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
    R_ext = np.concatenate([R, R, R])
    Rloc = np.interp(a, t_ext, R_ext)
    return d < frac * Rloc


def ring_ref(X):
    X = FIN._clean(X)
    sc = StandardScaler().fit_transform(X)
    p2 = PCA(n_components=2, random_state=42).fit_transform(sc)
    c, th, R = ellipse_boundary(p2)
    return inside_boundary(p2, c, th, R).astype(int)


df = pd.read_csv(r"..\public_data\train.csv")
ok = True
print(f"{'ds':5s} {'v3 winner':>16s} {'final winner':>14s}  ARI(v3,fin)  ring-ref(final)")
tot_t = 0.0
for ds, g in df.groupby("dataset_id"):
    X = g.drop(columns=["id", "dataset_id"]).values.astype(float)
    i3 = V3.explain(X, ds)
    t0 = time.time()
    if_ = FIN.explain(X, ds)
    tot_t += time.time() - t0
    a = adjusted_rand_score(i3["labels"], if_["labels"])
    line = f"{ds:5s} {i3['winner']:>16s} {if_['winner']:>14s}  {a:9.3f}"
    if ds in ("ds16", "ds23"):
        ref = ring_ref(X)
        ar = adjusted_rand_score(ref, if_["labels"])
        line += f"  {ar:9.3f}"
        ok &= (if_["winner"] == "ring/blob") and (ar > 0.85) and (a < 0.999)
    else:
        ok &= (a >= 1.0 - 1e-9) and (if_["winner"] == i3["winner"]) and \
              (if_["k_used"] == i3["k_used"])
    print(line)
print(f"\nmean final() time over 24 datasets: {tot_t/24:.2f}s")
print("RESULT:", "PASS - all checks ok" if ok else "FAIL")
