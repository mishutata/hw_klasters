my_solution_final — v5 (BEST SUBMISSION, Mean ARI 0.4846)
=========================================================

IMPORTANT CORRECTION (2026-11-13): the best score 0.4846 was produced by THIS
package — v5 (my_solution5/solution.py). It was previously mis-attributed to v6.
Verified: the best submission downloaded from the contest site is byte-identical
to my_solution5/solution.py (modulo CRLF) and to this folder's files.

submission.zip here is a BYTE-IDENTICAL copy of the proven 0.4846 package
(solution.py + original 15-cell analysis notebook), so resubmitting it reproduces
the recorded best score exactly.

v5 = v3 (0.45) + k/winner pins for ds04/ds09 (k=3, Spectral) and ds19/ds20 (k=4,
Spectral), visually verified on public train (dev_v5/k3_ds04.png etc.). Same
pipeline: clean -> per-dataset norm -> PCA 95%/cap-8 -> k prior +/-1 silhouette
-> core-set stability probe -> KMeans/GMM/Spectral ensemble + DBSCAN/HDBSCAN
confirming candidates.

Score history (hidden test, Mean ARI)
-------------------------------------
v5   0.4846  BEST — this package (my_solution5)
v7   0.4779
v13  0.4662  v6 base + median PCA rule ONLY on cap-bound ds08/09/13/14
v12  0.4655  v6 base + median PCA rule (global, no cap)
v8   0.4603
v10  0.4563
v3   0.45xx
(v6 and v11 have no score record: never submitted, or result not kept.)

Notes
-----
- v6 = v5 + pins ds18 (k=4/Spectral) and ds16/ds23 (k=4/Spectral, ring 3 arcs +
  center). Visually verified on public train but NEVER scored on the hidden test.
- v12/v13 were built on the v6 base, so their losses vs 0.4846 conflate the
  untested v6 pins with the median PCA rule; in any case both lost, and v13's
  isolation (only ds08/09/13/14 differ from its base) showed extra PCA dims hurt
  those four. PCA-dimension angle closed.
- If one more submission is wanted, the only remaining untested variant is v6
  alone (my_solution6): it tests whether the ds16/ds18/ds23 k=4/Spectral pins
  beat v5's handling of those three datasets. Otherwise: final = this folder.
