"""Friend's ds19 parabola code — copied VERBATIM from old_solutions/friend/message.txt.

Only the ds19 part was taken (4 constants + 1 function). Nothing else from
that file was used. Scaffolding below just runs it on public_data/train.csv
to produce labels + a picture for comparison.

Usage:  python friend_ds19.py
Outputs (next to this file):
    friend_ds19_train_labels.npz   — id + label per train point
    friend_ds19_parabolas.png      — x6/x7 scatter, colored by the friend's labels
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ==================== COPIED FROM FRIEND (ds19 only) ====================
DS19_PARABOLA_PARAMS = [
    (-1.1793103448, np.array([-0.51450670080271, 0.30157143901503, 1.10187675542021])),
    (2.0000000000, np.array([-0.29481027741668, 0.82156823037054, 0.43004646553911])),
]

DS19_MEAN_X6 = -0.36252027523335
DS19_SCALE_X6 = 0.34766015183175
DS19_MEAN_X7 = -0.43738367249569
DS19_SCALE_X7 = 0.47794327687906

def _assign_ds19_parabolas(X_8d):
    """Deterministic assign для ds19 с train-hardcoded normalization."""
    x = (X_8d[:, 5] - DS19_MEAN_X6) / DS19_SCALE_X6
    y = (X_8d[:, 6] - DS19_MEAN_X7) / DS19_SCALE_X7
    dists = np.zeros((len(x), 2))
    for c, (theta, coef) in enumerate(DS19_PARABOLA_PARAMS):
        xr = x * np.cos(theta) + y * np.sin(theta)
        yr = -x * np.sin(theta) + y * np.cos(theta)
        dists[:, c] = np.abs(yr - np.polyval(coef, xr))
    return dists.argmin(1)
# ==================== END COPIED PART ====================


def main():
    df = pd.read_csv(os.path.join(HERE, '..', '..', 'public_data', 'train.csv'))
    d = df[df['dataset_id'] == 'ds19'].reset_index(drop=True)
    X = d[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']].to_numpy(float)
    lab = _assign_ds19_parabolas(X)

    np.savez(os.path.join(HERE, 'friend_ds19_train_labels.npz'),
             id=d['id'].to_numpy(), labels=lab)
    print('ds19 train points:', len(d))
    print('label counts:', dict(zip(*np.unique(lab, return_counts=True))))

    fig, ax = plt.subplots(figsize=(7, 7))
    for c in np.unique(lab):
        m = lab == c
        ax.scatter(d.loc[m, 'x6'], d.loc[m, 'x7'], s=8, label=f'friend label {c} (n={m.sum()})')
    # draw the two hardcoded curves in raw x6/x7 coords
    for theta, coef in DS19_PARABOLA_PARAMS:
        xr = np.linspace(-3, 3, 400)
        yr = np.polyval(coef, xr)
        xs = xr * np.cos(theta) - yr * np.sin(theta)
        ys = xr * np.sin(theta) + yr * np.cos(theta)
        ax.plot(DS19_MEAN_X6 + DS19_SCALE_X6 * xs, DS19_MEAN_X7 + DS19_SCALE_X7 * ys,
                'k--', lw=1.5)
    ax.set_xlabel('x6'); ax.set_ylabel('x7')
    ax.set_title("Friend's ds19: hardcoded parabolas (raw coords)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, 'friend_ds19_parabolas.png')
    fig.savefig(out, dpi=120)
    print('saved:', out)


if __name__ == '__main__':
    main()
