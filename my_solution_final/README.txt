my_solution_final — v6, the best submitted solution (Mean ARI 0.4846)
=====================================================================

solution.py here = my_solution6/solution.py, byte-identical. notebook.ipynb and
submission.zip are built from it (build_final.py). Do NOT submit anything else:
the late PCA experiments below all scored worse.

Score history (public submissions)
----------------------------------
  v3 0.45xxx   v7 ~0.46   v8 ~0.47   v10 ~0.45-0.48   (see git log / RESULT_RECORDs)
  v6 0.4846    <- BEST, this package
  v12 0.4655   median-reconstruction PCA rule, GLOBAL (no cap), tau=0.2
  v13 0.4662   same rule but ONLY for the 4 cap-bound datasets ds08/ds09/ds13/ds14
               (other 20 datasets bit-identical to v6 by construction)

Why we stopped at v6
--------------------
The only open hypothesis was that v6's hard PCA cap of 8 dims hurt the four
datasets with d95 > 8 (ds08/09/13/14 kept only 75-91% variance at cap 8).
v13 isolates exactly that change: since its other 20 datasets are bit-identical
to v6, its -0.018 loss vs v6 comes entirely from those four -> extra (noisy)
PCA dimensions make clustering WORSE there, not better. v12's near-identical
loss (-0.019) confirms the same conclusion and shows the ~7 other datasets that
drifted under v12 were net-neutral. The PCA-dimension angle is thus closed;
v6's 95%-variance/cap-8 rule stays as-is.

Pipeline (per dataset, see solution.py docstring)
-------------------------------------------------
1. NaN-safe clean -> per-dataset norm (minmax/log-sym/robust, chosen on public train)
2. PCA: 95% variance, cap 8 dims
3. k = prior (+/-1 silhouette window; pinned for ds04/ds09/ds16/ds18/ds19/ds20/ds23)
4. Density probe (d5 core set) + KMeans seed-stability probe (fit on core if unstable)
5. Ensemble KMeans/GMM/Spectral (+DBSCAN/HDBSCAN confirming only),
   winner = silhouette + 0.3*consensus NMI

Deterministic: random_state=42 everywhere.
