"""Fuse the ATD ball with the video ball candidates into one ball track.

Per frame the ball is one of: a video candidate (detector box, projected to the pitch through the
homography at the bottom of the box, i.e. on the ground plane), the ATD ball, or "not seen". A
Viterbi pass per half picks the most plausible sequence:

  emission   video candidate: w_score * logit(score) + bonus if it agrees with the ATD ball
             ATD ball: a constant (it is right more often than not, but sometimes on an empty
             patch of grass or on the wrong player)
             not seen: 0
  transition displacement between consecutive frames, in metres: free up to the distance a fast
             ball covers in one frame, then a steep cost; entering / leaving "not seen" costs a
             constant, so one-frame flickers are not worth a jump.

Output arrays are in the ATD convention (normalised 0-1 pitch coordinates, NaN when not seen), so
the possession pipeline can consume the fused ball unchanged.
"""

import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CROPS = os.path.join(ROOT, "cache", "cv", "crops")

FPS = 25.0


def load_candidates(tag="atd", min_score=None):
    """{frame: (k, 5) x, y, w, h, score} in frame pixels, from every chunk of one or more detection
    passes ("atd+grid"); a frame seen by several passes gets the union, best scores first."""
    out = {}
    for t in tag.split("+"):
        floor = (min_score or {}).get(t, 0.0)
        for path in sorted(glob.glob(os.path.join(CROPS, t, "chunk_*.npz"))):
            d = np.load(path)
            off = np.r_[0, np.cumsum(d["n_ball"])]
            for i, f in enumerate(d["frames"]):
                c = d["ball"][off[i]:off[i + 1]]
                c = c[c[:, 4] >= floor]
                f = int(f)
                if f in out and len(out[f]):
                    c = np.concatenate([out[f], c])
                    c = c[np.argsort(-c[:, 4])]
                out[f] = c
    return out


def to_pitch(H, frames, u, v):
    """Image pixels -> normalised pitch coordinates through the inverse homography per frame."""
    Hi = np.linalg.inv(H[frames])
    p = np.einsum("nij,nj->ni", Hi, np.column_stack([u, v, np.ones(len(u))]))
    return p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]


class Params:
    k_max = 6            # video candidates per frame considered
    min_score = 0.08
    w_score = 1.0        # weight of logit(score)
    score_bias = 0.5     # added to every video candidate
    atd_agree_m = 1.5    # within this distance of the ATD ball a candidate gets the bonus
    w_agree = 2.0
    e_atd = 1.0          # emission of the ATD pseudo-candidate
    free_m = 1.4         # metres per frame a ball can move without cost (35 m/s)
    w_move = 1.0         # cost per metre beyond free_m...
    hard_m = 3.0         # ...and an impossible jump beyond this
    w_step = 0.15        # small cost per metre for any move, prefers smooth tracks
    c_lose = 1.5         # entering "not seen"
    c_find = 1.5         # leaving "not seen"
    margin_m = 0.5       # candidates further than this outside the pitch are dropped (spare balls,
                         # ball boys; a ball going out is left to the ATD)
    ghost_m = 6.0        # an ATD ball with no tracked player this close ...
    ghost_pen = 1.5      # ... loses this much emission (static ghosts on empty grass)
    offset_window = 12   # frames either side used to align ATD frames to nearby video frames
    smooth_sigma = 1.0   # Gaussian smoothing (frames) within each observed run; 0 = off


def _logit(s):
    s = np.clip(s, 1e-4, 1 - 1e-4)
    return np.log(s / (1 - s))


def fuse(atd_ball, cands, H, blocks, pitch, prm=Params, players=None):
    """Return dict with xy (n, 2) normalised, source (n,) int8 (0 none, 1 video, 2 atd),
    score (n,) of the chosen video candidate, and agree (n,) bool (video within atd_agree_m)."""
    n = len(atd_ball)
    L, W = pitch
    K = prm.k_max + 2  # video..., atd, none
    X = np.full((n, K), np.nan)
    Y = np.full((n, K), np.nan)
    E = np.full((n, K), -np.inf)
    S = np.zeros((n, K))
    # ATD pseudo-candidate
    a = ~np.isnan(atd_ball[:, 0])
    X[a, K - 2] = atd_ball[a, 0] * L
    Y[a, K - 2] = atd_ball[a, 1] * W
    E[a, K - 2] = prm.e_atd
    if players is not None and prm.ghost_pen:
        with np.errstate(all="ignore"):
            d = np.nanmin(np.hypot(players[:, :, 0] * L - X[:, K - 2:K - 1],
                                   players[:, :, 1] * W - Y[:, K - 2:K - 1]), axis=1)
        ghost = a & ~(d < prm.ghost_m)
        E[ghost, K - 2] -= prm.ghost_pen
    E[:, K - 1] = 0.0
    # video candidates
    frames = np.array(sorted(cands), dtype=np.int64)
    for f in frames:
        c = cands[f]
        c = c[c[:, 4] >= prm.min_score][:prm.k_max]
        if not len(c):
            continue
        fx, fy = to_pitch(H, np.full(len(c), f), c[:, 0], c[:, 1] + c[:, 3] / 2)
        x, y = fx * L, fy * W
        ok = (x > -prm.margin_m) & (x < L + prm.margin_m) & (y > -prm.margin_m) & (y < W + prm.margin_m)
        k = np.flatnonzero(ok)
        X[f, k], Y[f, k] = x[k], y[k]
        e = prm.w_score * _logit(c[k, 4]) + prm.score_bias
        if a[f]:
            d = np.hypot(x[k] - X[f, K - 2], y[k] - Y[f, K - 2])
            e = e + prm.w_agree * (d < prm.atd_agree_m)
        E[f, k] = e
        S[f, k] = c[k, 4]

    xy = np.full((n, 2), np.nan)
    source = np.zeros(n, np.int8)
    score = np.zeros(n)
    agree = np.zeros(n, bool)
    for _, first, last in blocks:
        path = _viterbi(X[first:last + 1], Y[first:last + 1], E[first:last + 1], prm)
        idx = np.arange(first, last + 1)
        st = path
        seen = st < K - 1
        xy[idx[seen], 0] = X[idx[seen], st[seen]] / L
        xy[idx[seen], 1] = Y[idx[seen], st[seen]] / W
        source[idx[seen]] = np.where(st[seen] == K - 2, 2, 1)
        score[idx[seen]] = S[idx[seen], st[seen]]
        vid = seen & (st < K - 2)
        if vid.any():
            fv = idx[vid]
            d = np.hypot(X[fv, st[vid]] - X[fv, K - 2], Y[fv, st[vid]] - Y[fv, K - 2])
            agree[fv] = d < prm.atd_agree_m
    xy = align_atd(xy, source, agree, atd_ball, prm.offset_window)
    if prm.smooth_sigma > 0:
        xy = smooth_runs(xy, prm.smooth_sigma)
    return {"xy": xy, "source": source, "score": score, "agree": agree}


def align_atd(xy, source, agree, atd_ball, window):
    """Shift ATD-sourced frames by the median (video - ATD) offset of agreeing video frames
    within +-window, so switching between the two sources does not make the ball jump."""
    out = xy.copy()
    vid = np.flatnonzero((source == 1) & agree & ~np.isnan(atd_ball[:, 0]))
    if not len(vid):
        return out
    off = xy[vid] - atd_ball[vid]
    atd_idx = np.flatnonzero(source == 2)
    lo = np.searchsorted(vid, atd_idx - window)
    hi = np.searchsorted(vid, atd_idx + window, side="right")
    for f, a, b in zip(atd_idx, lo, hi):
        if b > a:
            out[f] = xy[f] + np.median(off[a:b], axis=0)
    return out


def smooth_runs(xy, sigma):
    """Gaussian smoothing along time inside each run of observed frames (gaps are not bridged)."""
    out = xy.copy()
    seen = ~np.isnan(xy[:, 0])
    edges = np.diff(np.r_[0, seen.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    for a, b in zip(starts, stops):
        if b - a < 3:
            continue
        seg = xy[a:b]
        pad = np.pad(seg, ((r, r), (0, 0)), mode="edge")
        num = np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(2)], axis=1)
        out[a:b] = num / k.sum()
    return out


def _viterbi(X, Y, E, prm):
    n, K = E.shape
    none = K - 1
    back = np.zeros((n, K), np.int16)
    score = E[0].copy()
    for t in range(1, n):
        dx = X[t][None, :] - X[t - 1][:, None]
        dy = Y[t][None, :] - Y[t - 1][:, None]
        d = np.hypot(dx, dy)  # (K_prev, K_cur), NaN involving none / empty
        cost = prm.w_step * d + prm.w_move * np.maximum(0.0, d - prm.free_m)
        cost = np.where(d > prm.hard_m, np.inf, cost)
        cost = np.where(np.isnan(d), np.inf, cost)
        cost[none, :] = prm.c_find
        cost[:, none] = prm.c_lose
        cost[none, none] = 0.0
        tot = score[:, None] - cost
        b = np.argmax(tot, axis=0)
        score = tot[b, np.arange(K)] + E[t]
        back[t] = b
    path = np.zeros(n, np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
