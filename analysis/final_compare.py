"""Compare v1 (my_solver) vs v2 (my_solution2) vs v3 (my_solution3) on all 24
public datasets, plus the sol_16_23 ellipse/ring reference split for ds16/ds23.
Writes final_compare.csv; prints only where versions disagree."""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.preprocessing import StandardScaler


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


V1 = load(r"..\my_solver\solution.py", "v1mod")
V2 = load(r"..\my_solution2\solution.py", "v2mod")
V3 = load(r"..\my_solution3\solution.py", "v3mod")

# --- sol_16_23 ellipse/ring reference (same code as sol_16_23/eval_ring_blob.py)

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
    t_ext = np.concatenate([th - 2*np.pi, th, th + 2*np.pi])
    R_ext = np.concatenate([R, R, R])
    Rloc = np.interp(a, t_ext, R_ext)
    return d < frac * Rloc


def ring_ref(X):
    X = V3._clean(X)
    sc = StandardScaler().fit_transform(X)
    p2 = PCA(n_components=2, random_state=42).fit_transform(sc)
    c, th, R = ellipse_boundary(p2)
    return inside_boundary(p2, c, th, R).astype(int)


df = pd.read_csv(r"..\public_data\train.csv")
rows = []
for ds, g in df.groupby("dataset_id"):
    X = g.drop(columns=["id", "dataset_id"]).values.astype(float)
    rec = {"ds": ds, "n": len(X)}
    for tag, mod in (("v1", V1), ("v2", V2), ("v3", V3)):
        t0 = time.time()
        info = mod.explain(X, ds)
        rec[f"{tag}_k"] = info["k_used"]
        rec[f"{tag}_win"] = info["winner"]
        rec[f"{tag}_score"] = round(info["methods"][info["winner"]], 3) \
            if isinstance(info.get("methods"), dict) else ""
        rec[f"{tag}_t"] = round(time.time() - t0, 2)
        rec[f"{tag}_lab"] = info["labels"]
    for a, b in (("v1", "v2"), ("v1", "v3"), ("v2", "v3")):
        rec[f"ARI_{a}{b}"] = round(
            adjusted_rand_score(rec[f"{a}_lab"], rec[f"{b}_lab"]), 3)
    if ds in ("ds16", "ds23"):
        ref = ring_ref(X)
        for tag in ("v1", "v2", "v3"):
            rec[f"ARI_{tag}_ringref"] = round(
                adjusted_rand_score(ref, rec[f"{tag}_lab"]), 3)
    rows.append(rec)

out = pd.DataFrame(rows)
out.drop(columns=[c for c in out.columns if c.endswith("_lab")], inplace=True)
out.to_csv("final_compare.csv", index=False)

print(f"{'ds':5s} {'k':>4s} {'winner(v3)':>22s}  ARI v1v2 v1v3 v2v3   ring-ref(v1/v2/v3)")
for r in rows:
    rr = ""
    if "ARI_v1_ringref" in r:
        rr = f"  ringref: {r['ARI_v1_ringref']:.2f}/{r['ARI_v2_ringref']:.2f}/{r['ARI_v3_ringref']:.2f}"
    flag = "" if min(r["ARI_v1v2"], r["ARI_v1v3"], r["ARI_v2v3"]) >= 0.98 else "  <-- differs"
    print(f"{r['ds']:5s} {r['v3_k']:>4d} {r['v3_win']:>22s}  "
          f"{r['ARI_v1v2']:.2f} {r['ARI_v1v3']:.2f} {r['ARI_v2v3']:.2f}{rr}{flag}")
print("saved analysis/final_compare.csv")