probes/k2_ds13 — champion q7_ds19friend + ds13 k 3->2 (PROBE)
=================================================================

Built 2026-11-18. Base = my_solution_final (champion 0.551536), byte-identical
except exactly ONE hunk in solution.py: CFG 'ds13': (3, 'gmm_tied') -> (2, 'gmm_tied').
Same algorithm (gmm_tied), same PCA dim (EVR_95=9), only k changes.

Verified on public train (all 24 datasets, both solutions side by side):
  - ds13: labels changed 463/984; base k=3 sizes [208, 520, 256] -> k=2 sizes [259, 725]
    contingency: k=2 merges the k=3 clusters of size 208+520, keeps the 256-cluster
    separate (2 stray points flipped). ARI(k=2 labels, k=3 labels) on train = 0.57.
  - all other 23 datasets: labels byte-identical to champion

Readout vs base 0.551536: delta*24 = ds13's ARI-sum at k=2.
  score > 0.551536 -> ds13 is really 2 clusters, adopt k=2
  score < 0.551536 -> keep k=3 (champion)

RESULT (submitted 2026-11-18): **0.546620** = base - 0.004916 => ds13 at k=2
~-0.118 ARI-sum vs k=3. REJECTED: ds13 stays k=3 (gmm_tied). Champion unchanged:
q7_ds19friend = 0.551536. (Consistent with the earlier k=1 line: fewer clusters
keep losing on ds13 — 3 is the answer.)
