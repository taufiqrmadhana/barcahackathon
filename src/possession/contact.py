"""Contact ownership adapted from Neo's src_s2/s2.py (origin/neo/loop-pipeline,
d61c78f). Use the original ATD as an independent vote alongside the video-fused model.

Contacts and flights are processed separately in each half. The vote abstains in long
ball gaps; the existing shape model handles them. No labels are read here.
"""
from dataclasses import replace

import numpy as np
from scipy.ndimage import maximum_filter1d

from .clean import runs, since_last, until_next, velocity


def ball_track(xy):
    """Neo's five-observation median filter, without wraparound at the half edges."""
    xy = xy.copy()
    idx = np.flatnonzero(np.isfinite(xy).all(axis=1))
    if len(idx) > 5:
        points = xy[idx]
        padded = np.pad(points, ((2, 2), (0, 0)), mode='edge')
        median = np.median(np.stack([padded[k:k + len(idx)] for k in range(5)]), axis=0)
        xy[idx[np.linalg.norm(points - median, axis=1) > 3.0]] = np.nan
    seen = np.isfinite(xy).all(axis=1)
    for a, b in zip(*runs(~seen)):
        if a and b < len(xy) and b - a <= 10:
            t = np.arange(1, b - a + 1)[:, None] / (b - a + 1)
            xy[a:b] = xy[a - 1] + (xy[b] - xy[a - 1]) * t
    return xy, seen


def contact_runs(players, ball, teams):
    bv, pv = velocity(ball, 2), velocity(players, 2)
    dist = np.linalg.norm(players - ball[:, None], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    near = dist.argmin(axis=1)
    rows = np.arange(len(ball))
    dmin = dist[rows, near]
    side = teams[near]
    other = np.where(teams[None] != side[:, None], dist, np.inf).min(axis=1)
    relative = np.linalg.norm(bv - pv[rows, near], axis=1)
    dv = np.zeros(len(ball))
    dv[2:-2] = np.linalg.norm(bv[4:] - bv[:-4], axis=1)
    dv = np.nan_to_num(dv)
    kick = (dv >= 2.0) & (dmin <= 2.0) & (dv >= maximum_filter1d(dv, 5))
    carry = (dmin <= 1.5) & (relative <= 1.5)
    with np.errstate(invalid='ignore'):
        clear = other - dmin >= 0.5
    contact = np.where((carry | kick) & clear, side, -1)
    out = []
    for team in (0, 1):
        mask = contact == team
        for a, b in zip(*runs(~mask)):
            if a and b < len(mask) and b - a <= 3:
                mask[a:b] = True
        for a, b in zip(*runs(mask)):
            if b - a >= 8 or kick[a:b].any():
                out.append((int(a), int(b), team))
    return sorted(out)


def owners(contacts, n, alpha):
    """Switch alpha of the way from the old release to the new first contact."""
    own = np.full(n, -1, dtype=np.int8)
    prev = None
    for a, b, team in contacts:
        if prev is None:
            own[a:b] = team
        else:
            _, end, old = prev
            if old == team:
                own[end:b] = team
            else:
                cut = int(round(end + alpha * (max(a, end) - end)))
                own[end:cut] = old
                own[cut:b] = team
        if prev is None or b >= prev[1]:
            prev = (a, b, team)
    if prev is not None:
        own[prev[1]:] = prev[2]
    return own


def infer(inputs, p):
    if not 0 <= p.contact_alpha <= 1:
        raise ValueError('contact_alpha must be between 0 and 1')
    def build():
        out = np.full(inputs.n, -1, dtype=np.int8)
        scale = np.asarray(inputs.atd['pitch'])
        for _, first, last in inputs.blocks:
            sl = slice(first, last + 1)
            ball, seen = ball_track(inputs.atd['ball'][sl, :2] * scale)
            players = inputs.atd['players'][sl, :, :2] * scale
            contacts = contact_runs(players, ball, inputs.atd['teams'])
            own = owners(contacts, len(ball), p.contact_alpha)
            if p.contact_window >= 0:
                near = np.minimum(since_last(seen), until_next(seen))
                own[near > p.contact_window] = -1
            out[sl] = own
        return out
    return inputs._get('contact_owner', (p.contact_alpha, p.contact_window), build)


def refine(inputs, p, labels, emit, bonus, loose_allowed=None):
    """Second decode: change H/A with the contact vote; optionally keep every D frame.

    Recentring preserves the strongest alive emission on each frame. Without it an
    ownership vote would also create unintended evidence that the ball is alive.
    Optional video eligibility opens loose-state entries with the same directional
    event bonus and team-switch cost as the control-state entries.
    """
    from .hsmm import Model
    own = infer(inputs, p)
    updated = emit.copy()
    updated[own == 0, :2] += p.w_contact
    updated[own == 1, 2:4] += p.w_contact
    updated[:, :4] -= (updated[:, :4].max(axis=1) - emit[:, :4].max(axis=1))[:, None]
    if p.contact_lock_dead:
        dead = labels == 'D'
        updated[dead, :4] = -1e6
        updated[~dead, 4] = -1e6
    cost = p.switch_cost if p.contact_switch_cost < 0 else p.contact_switch_cost
    if p.contact_local_switch:
        bonus = bonus.copy()
        reduction = max(0.0, p.switch_cost - cost)
        bonus[own == 0, 0] += reduction
        bonus[own == 1, 1] += reduction
        cost = p.switch_cost
    if loose_allowed is not None:
        if loose_allowed.shape != (inputs.n, 2):
            raise ValueError('loose-state eligibility must have shape (frames, 2)')
        bonus = np.column_stack([bonus, np.where(loose_allowed, bonus[:, :2], -np.inf)])
    labels, states = Model(replace(p, switch_cost=cost),
                           loose_switch_cost=cost if loose_allowed is not None else None).decode(
        updated, bonus, inputs.blocks, inputs.n)
    if p.contact_min_frames > 0:
        labels = tidy(labels, inputs.blocks, p.contact_min_frames)
        # These states are diagnostics for the final labels, after island cleanup.
        states[(labels == 'H') & (states >= 2)] = 0
        states[(labels == 'A') & (states < 2)] = 2
    return labels, states


def tidy(labels, blocks, minimum):
    """Neo's short-spell cleanup, restricted to H/A/H or A/H/A inside one half.

    A short spell next to D is retained: cleanup cannot move a restart or stoppage.
    """
    out = labels.copy()
    for _, first, last in blocks:
        for _ in range(2):
            changes = np.r_[first, np.flatnonzero(
                out[first + 1:last + 1] != out[first:last]) + first + 1, last + 1]
            for a, b in zip(changes[:-1], changes[1:]):
                if (a > first and b <= last and b - a < minimum
                        and out[a] in ('H', 'A')
                        and out[a - 1] == out[b] and out[b] in ('H', 'A')):
                    out[a:b] = out[b]
    return out
