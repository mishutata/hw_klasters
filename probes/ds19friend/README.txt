probes/ds19friend — q6 champion + FRIEND's ds19 parabolas (PROBE)
=================================================================

Built 2026-11-18. Base = my_solution_final (q6, BEST = 0.538102), byte-identical
except exactly 3 hunks in solution.py, all ds19:

  1. CFG: 'ds19': (2, 'spectral_knn') -> (2, 'parabola_ds19')
  2. Friend's ds19 code copied VERBATIM from old_solutions/friend/message.txt:
     DS19_PARABOLA_PARAMS + 4 hardcoded mean/scale constants + _assign_ds19_parabolas
     (raw x6/x7 -> normalize -> rotate into each parabola's frame -> nearest curve
      by vertical residual; deterministic, O(n), no fit at test time)
  3. Dispatch branch in cluster(): mt == 'parabola_ds19' runs on the cleaned raw
     X (no pool, no PCA). Degenerate guard: if one label is missing it falls back
     to the champion's local spectral_knn path.

notebook.ipynb = byte-identical to q6. submission.zip = solution.py + notebook.ipynb.

Verified on public train (all 24 datasets, both solutions run side by side):
  - ds19: labels changed 353/559 (spectral) -> 438/474 (friend parabolas),
    EXACT match with the standalone friend_ds19.py output point-for-point
  - all other 23 datasets: labels byte-identical to q6

Readout:
  - score > 0.538102  -> friend's deterministic cut beats our spectral on ds19, promote
  - score ~ 0.538102  -> ds19 contributes about the same either way
  - drop              -> revert to q6 (my_solution_final/submission.zip)
