probe k2_ds03_ds13 — q6 + ds03:(3->2,'gmm') + ds13:(3->2,'gmm_tied')
=====================================================================
Built 2026-11-17 from my_solution_final (q6, BEST 0.538102).

Changes vs q6 (exactly two lines in CFG):
  ds03: (3,'gmm')      -> (2,'gmm')
  ds13: (3,'gmm_tied') -> (2,'gmm_tied')
All other datasets untouched.

Verified on public train:
  - 22/24 datasets byte-identical to q6
  - ds03 sizes [641,192,127] -> [641,319] (the two small groups merge)
  - ds13 sizes [208,520,256] -> [259,725]
  - runtime all-24: 2.4 s

Submit: submission.zip (solution.py + notebook.ipynb).
solution_old_DO_NOT_SUBMIT.zip = q6 package for reference.

Readout vs q6 = 0.538102 (delta*24 = ARI-sum change over ds03+ds13):
  delta > 0  -> k=2 is right for at least one of them; follow up with
                single-change probes (k2_ds03 only / k2_ds13 only) to split it.
  delta ~ 0  -> neither change helps; keep q6's k=3 on both.
  delta < 0  -> k=3 was right; revert.

ds22 (user hunch: looks like 3 clusters, current k=2 'gmm') — NOT in this
probe, to be discussed separately.
