PUBLIC TRAIN DATA

train.csv contains the public 80% split of each dataset.

Columns:
- id
- dataset_id
- x1, x2, ..., xp

Important:
- this is an unsupervised clustering task, therefore true cluster labels are NOT published;
- different dataset_id values may have different numbers of real features;
- unused trailing x-columns are empty (NaN) and should be removed per dataset;
- the hidden 20% test split is used for leaderboard scoring.
