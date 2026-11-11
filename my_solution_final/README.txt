my_solution_final — combined best-of-everything version
=======================================================

solution.py here = my_solution3/solution.py (v3) + the ring/blob fix for
ds16/ds23. Nothing else changed: same pipeline, same priors, same seeds.

Provenance of each piece
------------------------
1. Base ensemble (KMeans + GMM + Spectral)   <- my_solver (v1)
   the original pipeline.
2. DBSCAN/HDBSCAN as confirming candidates   <- my_solution2 (v2, DBSCAN research)
   auto-tuned density clustering only joins the vote when it agrees with the
   best parametric candidate; its reliable contribution is core-point /
   outlier detection (steps 3-4), which stabilises KMeans (ds01 cross-seed
   ARI 0.41 -> 0.86).
3. Per-dataset input normalization           <- analysis/norm_research -> v3
   minmax wins 15/24, log-sym 8/24, robust 1/24; 'standard' never wins.
4. Ring/blob split for ds16/ds23            <- sol_16_23/eval_ring_blob.py
   these are an ellipse ring + dense central blob; plain k=2 clusterers split
   them into arcs (ARI ~0 vs the true split). The fitted boundary is stable:
   bootstrap ARI 0.93 / 0.94 over 50x 70% resamples.

How it works (see solution.py docstring for details)
----------------------------------------------------
1. NaN-safe clean -> per-dataset norm -> PCA (95% variance, <=8 dims)
2. k = prior (validated on public train), refined on a +/-1 silhouette window
3. Density probe: core set from d5 (distance to 5th NN)
4. Stability probe: KMeans at best_k with 3 seeds; if unstable, fit on the
   core set and extend labels to tail points by nearest centroid in Z-space
5. Candidates: KMeans + GMM + Spectral (+ DBSCAN/HDBSCAN as confirming only)
6. Winner: silhouette + 0.3 * mean consensus NMI (deterministic)
7. NEW for ds16/ds23: labels are replaced by the ellipse-aware ring/blob
   boundary fitted in PCA-2 space; winner is reported as 'ring/blob'

Usage
-----
labels = cluster(X, dataset_id)     # grading API, deterministic (random_state=42)
info   = explain(X, dataset_id)     # full report: winner, k_used, per-method scores, stability

Verification (analysis/final_verify.py, public train, all 24 datasets)
----------------------------------------------------------------------
- ds00-ds15, ds17-ds22 (22 sets): labels identical to v3 (ARI = 1.0), same winner and k
- ds16: winner 'ring/blob', ARI vs sol_16_23 reference split = 1.000
- ds23: winner 'ring/blob', ARI vs sol_16_23 reference split = 1.000
- mean runtime over the 24 datasets: ~0.5 s

Cross-version comparison of v1/v2/v3 on all 24 sets:
analysis/final_compare.csv (produced by analysis/final_compare.py)
