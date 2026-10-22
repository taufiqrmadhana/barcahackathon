"""Make clean external tracking look like the competition's ATD, and add free symmetries.

The shape features never touch the ball, so the gap between a clean provider feed and this ATD is
in the players: slots that vanish for seconds, jitter, a person tracked twice. Those are the
corruptions below. Rates are assumptions, not measurements; compare `tracked_per_frame` with the
ATD's 20.5 (DATA.md) before trusting a run, and change them in CorruptParams.
"""

from dataclasses import dataclass

import numpy as np

FPS = 25


@dataclass
class CorruptParams:
    jitter: float = 0.25              # m, gaussian noise on every position
    dropout_per_min: float = 1.5      # dropouts per tracked player per minute
    dropout_s: tuple = (1.0, 25.0)    # log-uniform length of a dropout, seconds
    frame_miss: float = 0.01          # independent per-position loss
    dup_per_min: float = 0.3          # a tracked player also shown in a free slot, per minute
    dup_offset: float = 0.5           # m between the two copies
    swap_teams: float = 0.5           # probability of exchanging home and away
    mirror: float = 0.5               # probability of mirroring the pitch in x and in y


def tracked_per_frame(xy):
    return float((~np.isnan(xy[..., 0])).sum(axis=1).mean())


def corrupt(match, rng, cp=None):
    """(xy, teams, label) of a corrupted copy; the input is not modified."""
    cp = cp or CorruptParams()
    xy = match["xy"].copy()
    teams = match["teams"].copy()
    label = match["label"].copy()
    n, s, _ = xy.shape
    length, width = match["pitch"]

    xy += rng.normal(0.0, cp.jitter, xy.shape).astype(np.float32)
    xy[rng.random((n, s)) < cp.frame_miss] = np.nan

    minutes = n / FPS / 60.0
    present = ~np.isnan(xy[..., 0])
    for j in range(s):
        share = present[:, j].mean()
        for _ in range(rng.poisson(cp.dropout_per_min * minutes * share)):
            seconds = np.exp(rng.uniform(np.log(cp.dropout_s[0]), np.log(cp.dropout_s[1])))
            a = rng.integers(0, n)
            xy[a:a + int(seconds * FPS), j] = np.nan

    free = np.flatnonzero(np.isnan(xy[..., 0]).mean(axis=0) > 0.6)
    for _ in range(rng.poisson(cp.dup_per_min * minutes * 22)):
        if len(free) == 0:
            break
        src, dst = rng.integers(0, s), rng.choice(free)
        a, seconds = rng.integers(0, n), rng.uniform(2.0, 20.0)
        b = min(n, a + int(seconds * FPS))
        gap = np.isnan(xy[a:b, dst, 0])
        xy[a:b, dst][gap] = (xy[a:b, src] + cp.dup_offset)[gap]
        teams[dst] = teams[src]

    if rng.random() < cp.mirror:
        xy[..., 0] = length - xy[..., 0]
    if rng.random() < cp.mirror:
        xy[..., 1] = width - xy[..., 1]
    if rng.random() < cp.swap_teams:
        teams = (1 - teams).astype(np.int8)
        swapped = label.copy()
        swapped[label == 0], swapped[label == 1] = 1, 0
        label = swapped
    return xy, teams, label
