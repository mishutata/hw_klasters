Probe k2_ds03 — q7_ds19friend + ds03:(3->2,'gmm') ONLY
=====================================================================
Built 2026-11-18 on champion q7_ds19friend (my_solution_final/, 0.551536).
Note: an earlier k2_ds03 folder existed, built on q6 (old spectral ds19);
it was stale and is replaced by this rebuild.

Changes vs champion (exactly one token in CFG):
  'ds03': (3, 'gmm') -> (2, 'gmm')
Everything else byte-identical (verified: single-token diff).

Verified on public train:
  - 23/24 datasets byte-identical to champion labels
  - ds03 sizes [641,192,127] -> [641,319] (two small groups merge)
  - runtime all-24: 2.4 s

Submit: submission.zip (solution.py + notebook.ipynb).
solution_old_DO_NOT_SUBMIT.zip = champion q7 package for reference only.

Readout vs champion = 0.551536 (delta*24 = ARI-sum change on ds03):
  delta > 0  -> k=2 helps ds03; fold into champion
  delta ~ 0  -> keep k=3
  delta < 0  -> revert
Prior estimate: combo k2_ds03_ds13 (q6 base) scored 0.536072 = q6 - 0.0488
ARI-sum combined, and k2_ds13 alone lost 0.118 on ds13 -> ds03's share ~=
+0.07 ARI-sum (predicted score ~ 0.5544). This probe checks it in isolation.
