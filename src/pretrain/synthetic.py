"""A fake match, only to check that the plumbing runs end to end. It teaches nothing real.

    python src/pretrain/synthetic.py            # writes pretrain_data/matches/synthetic_{a,b}.npz
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain import data  # noqa: E402


def make(seed, minutes=6, fps=25):
    rng = np.random.default_rng(seed)
    n = minutes * 60 * fps
    length, width = 105.0, 68.0
    shared = np.random.default_rng(123)
    base = np.stack([shared.uniform(10, 95, 22), shared.uniform(5, 63, 22)], axis=1)
    teams = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    xy = np.zeros((2 * n, 22, 2), dtype=np.float32)
    label = np.zeros(2 * n, dtype=np.int8)
    blocks = []
    for half in range(2):
        state, t, lab = 0, 0, label[half * n:(half + 1) * n]
        while t < n:
            span = int(rng.integers(6, 40) * fps)
            state = 2 if rng.random() < 0.2 else int(rng.integers(0, 2))
            lab[t:t + span] = state
            t += span
        for i in range(n):
            s = lab[i]
            home_push = {0: 8.0, 1: -4.0, 2: 0.0}[int(s)]
            away_push = {0: 4.0, 1: -8.0, 2: 0.0}[int(s)]
            push = np.where(teams == 0, home_push, away_push)
            xy[half * n + i, :, 0] = base[:, 0] + push
            xy[half * n + i, :, 1] = base[:, 1]
        sway = np.cumsum(rng.normal(0, 0.03, (n, 22, 2)), axis=0).astype(np.float32)
        xy[half * n:(half + 1) * n] += sway
        blocks.append((half + 1, half * n, (half + 1) * n - 1))
    return {"xy": xy, "teams": teams, "label": label, "blocks": np.array(blocks, dtype=np.int64),
            "pitch": np.array([length, width], dtype=np.float32)}


if __name__ == "__main__":
    for seed, name in ((1, "synthetic_a"), (2, "synthetic_b")):
        match = make(seed)
        print(data.save(match, name), data.summary(match))
