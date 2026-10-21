"""Step 1: tracking in metres, cleaned, with velocities.

Slots are not people: a slot is only trusted for its team and for short-range motion, so player
velocities are computed per slot over a few frames and dropped wherever the slot jumps.
"""

import numpy as np

FPS = 25


def runs(mask):
    """(starts, stops) of the True runs of a boolean array."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def run_length(mask):
    """Length of the True run each frame belongs to, 0 where False."""
    out = np.zeros(len(mask), dtype=np.int32)
    for start, stop in zip(*runs(mask)):
        out[start:stop] = stop - start
    return out


def since_last(mask):
    """Frames since mask was last True (0 on True frames, large before the first)."""
    idx = np.where(mask, np.arange(len(mask)), -10 ** 7)
    return np.arange(len(mask)) - np.maximum.accumulate(idx)


def until_next(mask):
    return since_last(mask[::-1])[::-1]


def velocity(xy, k):
    """Central difference over +-k frames in m/s, NaN where either end is missing."""
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (FPS / (2.0 * k))
    return v


def clean_players(players, scale, pitch, p):
    xy = players[:, :, :2] * scale
    length, width = pitch
    margin = p.player_margin
    off = ((xy[..., 0] < -margin) | (xy[..., 0] > length + margin)
           | (xy[..., 1] < -margin) | (xy[..., 1] > width + margin))
    xy[off] = np.nan

    vel = velocity(xy, p.vel_k)
    speed = np.hypot(vel[..., 0], vel[..., 1])
    vel[~(speed <= p.max_player_speed)] = np.nan
    return xy, vel


def drop_ball_spikes(xy, max_speed):
    """Observed ball points that jump away and straight back at an impossible speed."""
    seen = np.flatnonzero(~np.isnan(xy[:, 0]))
    bad = np.zeros(len(xy), dtype=bool)
    if len(seen) < 3:
        return bad
    pts = xy[seen]
    dt = np.diff(seen).astype(np.float32) / FPS
    step = np.hypot(*(pts[1:] - pts[:-1]).T) / dt
    over = step > max_speed
    spike = np.zeros(len(seen), dtype=bool)
    spike[1:-1] = over[:-1] & over[1:]
    lone_head = np.zeros(len(seen), dtype=bool)
    lone_head[0] = over[0]
    bad[seen[spike | lone_head]] = True
    return bad


def drop_ball_islands(xy, min_len, min_gap):
    """Very short observed runs isolated between long gaps: detector flickers, not a ball."""
    seen = ~np.isnan(xy[:, 0])
    bad = np.zeros(len(xy), dtype=bool)
    gap_before, gap_after = since_last(seen), until_next(seen)
    n = len(xy)
    for start, stop in zip(*runs(seen)):
        if stop - start >= min_len:
            continue
        before = gap_before[start - 1] if start > 0 else n
        after = gap_after[stop] if stop < n else n
        if before >= min_gap and after >= min_gap:
            bad[start:stop] = True
    return bad


def interpolate(xy, max_gap, max_speed):
    """Linear fill across short gaps whose implied speed is plausible. Returns (xy, filled)."""
    out = xy.copy()
    filled = np.zeros(len(xy), dtype=bool)
    starts, stops = runs(np.isnan(xy[:, 0]))
    for start, stop in zip(starts, stops):
        if start == 0 or stop == len(xy) or stop - start > max_gap:
            continue
        a, b = xy[start - 1], xy[stop]
        if np.hypot(*(b - a)) * FPS / (stop - start + 1) > max_speed:
            continue
        t = (np.arange(1, stop - start + 1) / float(stop - start + 1))[:, None]
        out[start:stop] = a + (b - a) * t
        filled[start:stop] = True
    return out, filled


def clean_ball(ball, scale, p):
    xy = ball[:, :2] * scale
    raw_seen = ~np.isnan(xy[:, 0])
    spikes = drop_ball_spikes(xy, p.ball_spike_speed)
    xy[spikes] = np.nan
    islands = drop_ball_islands(xy, p.ball_island_len, p.ball_island_gap)
    xy[islands] = np.nan
    seen = ~np.isnan(xy[:, 0])
    xy, filled = interpolate(xy, p.max_gap, p.ball_spike_speed)
    vel = velocity(xy, p.vel_k)
    return {
        "xy": xy,
        "seen": seen,
        "filled": filled,
        "known": ~np.isnan(xy[:, 0]),
        "vel": vel,
        "speed": np.hypot(vel[:, 0], vel[:, 1]),
        "n_dropped": int((raw_seen & ~seen).sum()),
    }


def attack_sign(players_xy, teams, blocks, pitch):
    """+1 if home attacks towards x = length in that half, else -1, from where each team stands."""
    sign = np.zeros(len(players_xy), dtype=np.int8)
    for _, first, last in blocks:
        window = players_xy[first:last + 1:25, :, 0]
        home = np.nanmean(window[:, teams == 0])
        away = np.nanmean(window[:, teams == 1])
        sign[first:last + 1] = 1 if home < away else -1
    return sign


def build(atd, blocks, p):
    length, width = atd["pitch"]
    scale = np.array([length, width], dtype=np.float32)
    players_xy, players_vel = clean_players(atd["players"], scale, atd["pitch"], p)
    ball = clean_ball(atd["ball"], scale, p)
    return {
        "players_xy": players_xy,
        "players_vel": players_vel,
        "teams": atd["teams"],
        "ball": ball,
        "pitch": (length, width),
        "attack": attack_sign(players_xy, atd["teams"], blocks, atd["pitch"]),
    }
