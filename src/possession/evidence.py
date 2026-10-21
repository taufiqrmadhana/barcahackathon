"""Step 2: per-frame evidence. Who could be controlling the ball, is it free, is play stopped.

Control needs two things at once: a player close to the ball, and the ball moving with that
player. A ball flying past a defender at 20 m/s is near him for a frame and controlled by nobody.
"""

import numpy as np

from .clean import run_length, since_last, until_next, FPS


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def smooth(x, width):
    if width <= 1:
        return x
    kernel = np.ones(width, dtype=np.float64) / width
    pad = np.pad(np.asarray(x, dtype=np.float64), (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def nearest_per_team(players_xy, players_vel, teams, ball_xy, ball_vel, chunk=20000):
    """Per team: distance to the nearest slot, its speed relative to the ball, and the runner-up."""
    n = len(ball_xy)
    dist = np.full((n, 2), np.inf, dtype=np.float32)
    second = np.full((n, 2), np.inf, dtype=np.float32)
    rel = np.full((n, 2), np.nan, dtype=np.float32)
    count2 = np.zeros((n, 2), dtype=np.int8)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        diff = players_xy[lo:hi] - ball_xy[lo:hi, None, :]
        d = np.hypot(diff[..., 0], diff[..., 1])
        d = np.where(np.isnan(d), np.inf, d)
        dv = players_vel[lo:hi] - ball_vel[lo:hi, None, :]
        r = np.hypot(dv[..., 0], dv[..., 1])
        for side in (0, 1):
            cols = np.flatnonzero(teams == side)
            ds = d[:, cols]
            order = np.argsort(ds, axis=1)
            best = order[:, 0]
            rows = np.arange(hi - lo)
            dist[lo:hi, side] = ds[rows, best]
            second[lo:hi, side] = ds[rows, order[:, 1]]
            rel[lo:hi, side] = r[:, cols][rows, best]
            count2[lo:hi, side] = (ds < 2.0).sum(axis=1)
    return dist, second, rel, count2


def control(dist, rel, ball_speed, p):
    """(N, 2) probability that the nearest player of each team controls the ball."""
    prox = sigmoid((p.ctrl_radius - dist) / p.ctrl_softness)
    rel = np.where(np.isnan(rel), np.abs(np.nan_to_num(ball_speed))[:, None] * 0.5 + 1.0, rel)
    sync = np.exp(-np.maximum(0.0, rel - p.sync_free) / p.sync_scale)
    return prox * sync


def ball_location(ball_xy, pitch, p):
    length, width = pitch
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    beyond = np.maximum.reduce([-x, x - length, -y, y - width])
    near_line = np.minimum.reduce([x, length - x, y, width - y])
    corner = (np.minimum(x, length - x) < p.corner_zone) & (np.minimum(y, width - y) < p.corner_zone)
    six = (np.minimum(x, length - x) < 6.0) & (np.abs(y - width / 2.0) < 10.0)
    centre = np.hypot(x - length / 2.0, y - width / 2.0) < 1.5
    return {"beyond": beyond, "near_line": near_line, "corner": corner, "six": six,
            "centre": centre}


def team_motion(players_xy, players_vel, teams):
    speed = np.hypot(players_vel[..., 0], players_vel[..., 1])
    out = {}
    with np.errstate(invalid="ignore"):
        out["speed"] = np.nanmean(speed, axis=1)
        out["count"] = np.stack([(~np.isnan(players_xy[:, teams == s, 0])).sum(1)
                                 for s in (0, 1)], axis=1)
    out["speed"] = np.nan_to_num(out["speed"], nan=0.0)
    return out


def build(trk, p):
    ball = trk["ball"]
    xy, vel, speed = ball["xy"], ball["vel"], ball["speed"]
    known = ball["known"]

    dist, second, rel, crowd = nearest_per_team(trk["players_xy"], trk["players_vel"],
                                                trk["teams"], xy, vel)
    ctrl = control(dist, rel, speed, p)
    ctrl[~known] = np.nan
    ctrl[ball["filled"]] *= p.filled_trust

    motion = team_motion(trk["players_xy"], trk["players_vel"], trk["teams"])
    loc = ball_location(xy, trk["pitch"], p)

    still = known & (np.nan_to_num(speed, nan=99.0) < p.still_speed)
    still_run = run_length(still)

    seen = ball["seen"]
    last_seen_beyond = np.full(len(xy), np.nan, dtype=np.float32)
    idx = np.where(seen, np.arange(len(xy)), -1)
    idx = np.maximum.accumulate(idx)
    has = idx >= 0
    last_seen_beyond[has] = loc["beyond"][idx[has]]
    missing_run = run_length(~known)

    return {
        "known": known,
        "seen": seen,
        "dist": dist,
        "second": second,
        "rel": rel,
        "crowd": crowd,
        "ctrl": ctrl,
        "ball_speed": np.nan_to_num(speed, nan=0.0),
        "ball_speed_smooth": smooth(np.nan_to_num(speed, nan=0.0), p.ball_smooth),
        "player_speed": motion["speed"],
        "player_speed_smooth": smooth(motion["speed"], p.slow_window),
        "count": motion["count"],
        "beyond": np.where(known, loc["beyond"], np.nan),
        "near_line": np.where(known, loc["near_line"], np.nan),
        "corner": known & loc["corner"],
        "six": known & loc["six"],
        "centre": known & loc["centre"],
        "still": still,
        "still_run": still_run,
        "missing_run": missing_run,
        "since_seen": since_last(seen),
        "until_seen": until_next(seen),
        "last_seen_beyond": last_seen_beyond,
        "fps": FPS,
    }
