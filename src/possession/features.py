"""Per-frame geometric and camera (homography) features from the ATD and XML metadata."""

import numpy as np

from .homography import apply

FPS = 25


def _runs(mask):
    """[(start, stop)] of True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def interpolate_ball(ball_xy, max_gap):
    """Linear fill of missing ball positions across gaps of at most max_gap frames."""
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
    """(N, 2) distance in metres from the ball to the nearest home / away player."""
    diff = players_xy - ball_xy[:, None, :]
    dist = np.sqrt(np.einsum("nsk,nsk->ns", diff, diff))
    dist = np.where(np.isnan(dist), np.inf, dist)
    return np.stack([dist[:, teams == side].min(axis=1) for side in (0, 1)], axis=1)


def forward_fill(values, mask, blocks):
    """Carry the last row where mask is True forward, restarting at each half."""
    out = np.full_like(values, np.nan)
    since = np.full(len(values), np.iinfo(np.int32).max, dtype=np.int64)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        m = mask[seg]
        idx = np.where(m, np.arange(m.size), -1)
        idx = np.maximum.accumulate(idx)
        have = idx >= 0
        block = values[seg]
        filled = np.full_like(block, np.nan)
        filled[have] = block[idx[have]]
        out[seg] = filled
        s = np.full(m.size, np.iinfo(np.int32).max, dtype=np.int64)
        s[have] = np.arange(m.size)[have] - idx[have]
        since[seg] = s
    return out, since


def robust_z(values, blocks, n):
    """(x - median) / (1.4826 MAD), statistics computed separately inside each half."""
    out = np.full(n, np.nan, dtype=np.float64)
    for _, first, last in blocks:
        seg = values[first:last + 1].astype(np.float64)
        med = np.nanmedian(seg)
        mad = np.nanmedian(np.abs(seg - med)) * 1.4826
        out[first:last + 1] = (seg - med) / (mad if mad > 1e-9 else 1.0)
    return out


def build_basic(atd, max_gap):
    length, width = atd["pitch"]
    scale = np.array([length, width], dtype=np.float32)

    players_xy = atd["players"][:, :, :2] * scale
    player_speed = atd["players"][:, :, 2]
    ball_raw = atd["ball"][:, :2] * scale

    ball_seen = ~np.isnan(ball_raw[:, 0])
    ball_xy, ball_filled = interpolate_ball(ball_raw, max_gap)
    ball_known = ~np.isnan(ball_xy[:, 0])

    dist = team_distances(players_xy, ball_xy, atd["teams"])

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


def build(atd, meta, camera, max_gap):
    feat = build_basic(atd, max_gap)
    n = meta.n_frames
    blocks = meta.blocks
    length, width = meta.pitch_m

    ball_norm = atd["ball"][:, :2].astype(np.float64)
    seen = feat["ball_seen"]
    last_ball, since = forward_fill(ball_norm, seen, blocks)
    lx, ly = last_ball[:, 0] * length, last_ball[:, 1] * width
    edge = np.minimum.reduce([lx, length - lx, ly, width - ly])
    feat["last_ball"] = last_ball
    feat["since_seen"] = since
    feat["last_edge_m"] = edge

    has_last = ~np.isnan(last_ball[:, 0])
    in_view = np.zeros(n, dtype=bool)
    in_view[has_last] = _in_view_rows(camera, last_ball, has_last)
    feat["last_in_view"] = in_view

    players = atd["players"][:, :, :2].astype(np.float64)
    vis = np.zeros(n, dtype=np.int16)
    half = meta.half_mask()
    rows = np.flatnonzero(half)
    for chunk in np.array_split(rows, max(1, len(rows) // 20000)):
        sub = _InViewSlice(camera, chunk)
        v = sub.in_view(np.nan_to_num(players[chunk], nan=-10.0))
        vis[chunk] = (v & ~np.isnan(players[chunk, :, 0])).sum(axis=1)
    feat["players_in_view"] = vis

    feat["quality"] = camera.quality
    feat["pan_speed"] = camera.pan_speed
    feat["view_width"] = camera.view_width
    feat["aim"] = camera.aim
    feat["attack_sign"] = meta.attack_sign_per_frame()

    feat["z_player_speed"] = robust_z(feat["player_speed"], blocks, n)
    feat["z_pan_speed"] = robust_z(camera.pan_speed, blocks, n)
    feat["z_ball_speed"] = robust_z(np.where(seen, feat["ball_speed"], np.nan), blocks, n)
    return feat


class _InViewSlice:
    """Camera.in_view restricted to a subset of frames, so big arrays are processed in chunks."""

    def __init__(self, camera, rows):
        self.camera, self.rows = camera, rows

    def in_view(self, xy_norm, margin=0.0):
        px, w = apply(self.camera.matrix[self.rows], xy_norm)
        px[w <= 0] = np.nan
        width, height = self.camera.image_size
        mu, mv = margin * width, margin * height
        with np.errstate(invalid="ignore"):
            return ((px[..., 0] >= mu) & (px[..., 0] <= width - mu)
                    & (px[..., 1] >= mv) & (px[..., 1] <= height - mv))


def _in_view_rows(camera, points, mask):
    rows = np.flatnonzero(mask)
    return _InViewSlice(camera, rows).in_view(points[rows, None, :], margin=0.03)[:, 0]
