# Clustering Challenge — repo map

Task: cluster 24 datasets (ds00–ds23), metric = Mean ARI. Statement: `условия.txt`.
Champion: **q6 = 0.538102** → `my_solution_final/` (its README.txt is the full score log).

| Folder | What's inside |
|---|---|
| `my_solution_final/` | **Champion q6**: solution.py + notebook.ipynb + submission.zip + README.txt (score/probe log) |
| `probes/` | Active single-variable probes, each = champion + exactly one change: `kplus1`, `init_ninit30`, `k1_ds08`, `k1_ds13`, `k1_ds14`, `ds19parab`. Readout: `(score − 0.538102) × 24` = that dataset's ARI-sum |
| `old_solutions/` | All previous packages: `my_solution2…17*` (incl. A/B & p/q variants), `my_solver`, `sol_16_23`, `friend` (groupmate's) |
| `pictures/` | Figures only: `pair_plots/`, `pair_plots_v9/`, `pair_plots_v10/` (from champion), `clustering_pictures*/`, loose check PNGs |
| `dev/` | Versioned dev workspaces (scripts + figures + logs): `v4…v16`, `analysis/`, `analysis2/`. Scripts assume CWD = repo root |
| `notebook_creation/` | Notebook builders (`build_nb*.py`, `tools_build_nb16.py`, `build_final.py`) and clean-run verification dirs `nb7_run/ nb8_run/ nb10_run/` |
| `tools/` | Build & figure tools. **Active:** `tools_build_submission.py` (rebuilds champion notebook+zip), `tools_pair_plots_v10.py`. Rest are one-shot historical builders |
| `scratch/` | Working files: configs, logs, tmp scripts/PNGs |
| `public_data/`, `starting_kit/`, `solver/` | Task material (train.csv, starter kit, reference baseline) — do not move |
| `submited scores/` | Scored submission zips, named by score |

## Conventions
- Probe = champion + ONE change, everything else byte-identical; submit one at a time.
- Pair plots: `python tools/tools_pair_plots_v10.py` (labels from champion, full train).
- Rebuild champion package: `python tools/tools_build_submission.py` (run from repo root).
- Layout reorganized 2026-11-17; path references in scripts were updated to match.
