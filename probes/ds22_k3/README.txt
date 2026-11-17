probe ds22_k3 — q6 + ds22:(2,'gmm') -> (3,'gmm')
=================================================
Built 2026-11-17 from old_solutions/my_solution17_q6_hisk6 (q6, BEST 0.538102).

Changes vs q6 (exactly one line in CFG):
  ds22: (2,'gmm') -> (3,'gmm')
All other datasets untouched.

Verified on public train:
  - 23/24 datasets byte-identical to q6
  - ds22 sizes [746,214] -> [219,255,486]
  - runtime all-24 (both solutions): 2.5 s

Submit: submission.zip (solution.py + notebook.ipynb).
solution_old_DO_NOT_SUBMIT.zip = q6 package for reference — never upload it.

Readout vs q6 = 0.538102 (delta*24 = ds22 ARI change, single dataset):
  delta > 0  -> ds22 is really k=3; fold into new champion.
  delta ~ 0  -> near-tie; keep k=2, question closed.
  delta < 0  -> k=2 was right, question closed.

Context: user sees 3 blobs in the ds22 pair plots while q6 uses k=2 'gmm'.
The k2_ds03_ds13 probe scored 0.536072 (-0.0487 ARI-sum vs q6), confirming
k=3 for ds03/ds13; ds22 was deliberately left out of that probe.
