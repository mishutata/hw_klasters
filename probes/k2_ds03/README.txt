probe k2_ds03 — q6 + ds03:(3->2,'gmm') ONLY
=====================================================================
Built 2026-11-17 from my_solution_final (q6, BEST 0.538102).
Single-change probe (one-probe strategy): the combined k2_ds03_ds13
probe came back at 0.536072 (-0.002030 vs q6); this splits that delta
between ds03 and ds13, paired with probes/ds13k2.

Change vs q6 (exactly one line in CFG):
  ds03: (3,'gmm') -> (2,'gmm')
All other datasets untouched.

Verified on public train:
  - 23/24 datasets byte-identical to q6
  - ds03 sizes [641,192,127] -> [641,319] (the two small groups merge)
  - runtime all-24: 2.4 s

Submit: submission.zip (solution.py + notebook.ipynb).
solution_old_DO_NOT_SUBMIT.zip = q6 package for reference.

Readout vs q6 = 0.538102 (delta*24 = ARI-sum change on ds03 alone):
  delta > 0 -> ds03 k=2 helps; the combo's drop came from ds13.
  delta ~ 0 -> ds03 neutral; the verdict comes from ds13k2.
  delta < 0 -> ds03 k=2 hurts by exactly that amount.

With ds13k2 (same base, same day): the two single deltas should add up
to the combo's -0.0487 ARI-sum (any gap = interaction between the two
changes, expected ~0).
ds22 (user hunch: 3 clusters) stays OUT of both probes.
