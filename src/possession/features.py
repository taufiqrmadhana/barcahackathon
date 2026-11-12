"""Per-frame geometric features from the ATD, in metres."""

import numpy as np

FPS = 25


def _runs(mask):
    """[(start, stop)] of True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def interpolate_ball(ball_xy, max_gap):
    """Linear fill of missing ball positions across gaps of at most max_gap frames.

    Returns (filled positions, mask of frames that were filled rather than observed).
    """
    filled = ball_xy.copy()
    filled_mask = np.zeros(len(ball_xy), dtype=bool)
    missing = np.isnan(ball_xy[:, 0])
    for start, stop in _runs(missing):
        if start == 0 or stop == len(ball_xy) or stop - start > max_gap:
            continue
        a, b = ball_xy[start - 1], ball_xy[stop]
        t = (np.arange(1, stop - start + 1) / float(stop - start + 1))[:, None]
        filled[start:stop] = a + (b - a) * t
        filled_mask[start:stop] = True
    return filled, filled_mask


def run_length(mask):
    """For every frame, the length of the True run it belongs to (0 where False)."""
    out = np.zeros(len(mask), dtype=np.int32)
    for start, stop in _runs(mask):
        out[start:stop] = stop - start
    return out


def team_distances(players_xy, ball_xy, teams):
    """(N, 2) distance in metres from the ball to the nearest home / away player. inf if none."""
    diff = players_xy - ball_xy[:, None, :]
    dist = np.sqrt(np.einsum("nsk,nsk->ns", diff, diff))
    dist = np.where(np.isnan(dist), np.inf, dist)
    return np.stack([dist[:, teams == side].min(axis=1) for side in (0, 1)], axis=1)


def build(atd, max_gap):
    length, width = atd["pitch"]
    scale = np.array([length, width], dtype=np.float32)

    players_xy = atd["players"][:, :, :2] * scale
    player_speed = atd["players"][:, :, 2]
    ball_raw = atd["ball"][:, :2] * scale

    ball_seen = ~np.isnan(ball_raw[:, 0])
    ball_xy, ball_filled = interpolate_ball(ball_raw, max_gap)
    ball_known = ~np.isnan(ball_xy[:, 0])

    dist = team_distances(players_xy, ball_xy, atd["teams"])

    # Ball speed from consecutive positions, so it does not depend on the ATD speed column.
    step = np.full(len(ball_xy), np.nan, dtype=np.float32)
    step[1:] = np.hypot(*(ball_xy[1:] - ball_xy[:-1]).T) * FPS

    seen_players = (~np.isnan(player_speed)).sum(axis=1)
    mean_speed = np.nansum(player_speed, axis=1) / np.maximum(seen_players, 1)
    norm = atd["ball"][:, :2]
    outside = (norm[:, 0] < 0) | (norm[:, 0] > 1) | (norm[:, 1] < 0) | (norm[:, 1] > 1)

    return {
        "ball_xy": ball_xy,
        "ball_seen": ball_seen,
        "ball_filled": ball_filled,
        "ball_known": ball_known,
        "missing_run": run_length(~ball_known),
        "dist": dist,
        "ball_speed": step,
        "player_speed": mean_speed.astype(np.float32),
        "ball_outside": outside & ball_seen,
    }
