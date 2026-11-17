probe ds13k2 — q6 + ds13:(3->2,'gmm_tied') ONLY
=====================================================================
Built 2026-11-17 from my_solution_final (q6, BEST 0.538102).
Single-change probe (one-probe strategy): the combined k2_ds03_ds13
probe came back at 0.536072 (-0.002030 vs q6); this splits that delta
between ds03 and ds13, paired with probes/k2_ds03.

Change vs q6 (exactly one line in CFG):
  ds13: (3,'gmm_tied') -> (2,'gmm_tied')
All other datasets untouched.

Verified on public train:
  - 23/24 datasets byte-identical to q6
  - ds13 sizes [208,520,256] -> [259,725]
  - runtime all-24: 2.4 s

Submit: submission.zip (solution.py + notebook.ipynb).
solution_old_DO_NOT_SUBMIT.zip = q6 package for reference.

Readout vs q6 = 0.538102 (delta*24 = ARI-sum change on ds13 alone):
  delta > 0 -> ds13 k=2 helps; the combo's drop came from ds03.
  delta ~ 0 -> ds13 neutral; the verdict comes from k2_ds03.
  delta < 0 -> ds13 k=2 hurts by exactly that amount.

With k2_ds03 (same base, same day): the two single deltas should add up
to the combo's -0.0487 ARI-sum (any gap = interaction between the two
changes, expected ~0).
ds22 (user hunch: 3 clusters) stays OUT of both probes.
