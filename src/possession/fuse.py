"""Turn evidence, shape probabilities, tags and events into decoder inputs.

Emissions are log-likelihood style scores over [H_ctrl, H_loose, A_ctrl, A_loose, D]; bonuses are
per-frame reductions of transition costs [to H, to A, into D, D to H, D to A].
"""

import numpy as np

from .evidence import sigmoid

HOME, AWAY = 0, 1
TO_H, TO_A, DEAD_IN, H_OUT, A_OUT = range(5)


def dead_score(ev, shape_q, events, p):
    n = len(ev["known"])
    z = np.full(n, p.d_bias, dtype=np.float64)
    seen = ev["seen"]
    beyond = np.nan_to_num(ev["beyond"], nan=-99.0)
    z += p.w_out * (seen & (beyond > p.out_margin))
    carried = ~seen & (np.nan_to_num(ev["last_seen_beyond"], nan=-99.0) > p.out_margin)
    z += p.w_out_carry * carried
    slow = np.clip((p.slow_speed - ev["player_speed_smooth"]) / p.slow_scale, -2.0, 2.0)
    z += p.w_slow * slow
    long_missing = ev["missing_run"] > p.dead_gap
    z += p.w_missing_slow * long_missing * np.maximum(slow, 0.0)
    z += p.w_missing * long_missing
    placed = ev["still_run"] >= p.placed_min_frames
    z += p.w_still * placed
    if shape_q is not None:
        qd = np.clip(shape_q[:, 2], 1e-3, 1 - 1e-3)
        z += p.w_shape_dead * np.log(qd / (1 - qd))
    for start, stop, conf, _ in events["dead"]:
        z[start:stop] += p.w_hold * conf
    for f, _, conf, _ in events["restarts"]:
        z[f:f + p.post_restart_frames] -= p.w_post_restart * conf
    for f in events["tag_starts"]:
        z[max(0, f - p.tag_dead_frames):f + p.tag_dead_frames] += p.w_tag_dead
    inside = beyond <= 0.0
    moving = seen & inside & (ev["ball_speed"] > p.alive_ball_speed)
    z -= p.w_alive * moving
    if "whistles" in events:
        speed = ev["player_speed_smooth"]
        for f in events["whistles"][0]:
            if p.w_whistle_dead:
                z[f:f + p.whistle_hold] += p.w_whistle_dead
            if p.w_whistle_confirmed:
                # play stops at the whistle but players and ball take seconds to slow down; if they
                # do slow down soon after, the stoppage started at the whistle, not at the slowdown
                win = speed[f:f + p.whistle_confirm_window]
                slow = np.flatnonzero(win < p.whistle_confirm_speed)
                if len(slow):
                    z[f + p.whistle_lag:f + slow[0] + 1] += p.w_whistle_confirmed
    return z


def nearest_owner(ev, p):
    """(N, 2) proximity-only ownership: that team's nearest slot is close to the ball and clearly
    nearer than the other team's. Unlike ctrl it does not ask the ball to move with the player,
    so it fires at the reception rather than once the receiver is carrying the ball. 0 where the
    ball is not known; interpolated ball frames are trusted as in ctrl."""
    d = np.where(np.isfinite(ev["dist"]), ev["dist"], 99.0)
    prox = sigmoid((p.near_radius - d) / p.ctrl_softness)
    lead = np.clip((d[:, ::-1] - d) / p.near_margin, 0.0, 1.0)
    near = prox * lead
    near[~ev["known"]] = 0.0
    near[ev["known"] & ~ev["seen"]] *= p.filled_trust
    return near


def emissions(ev, shape_q, tags, events, p):
    n = len(ev["known"])
    emit = np.zeros((n, 5), dtype=np.float64)
    known = ev["known"]
    eps = p.ctrl_eps

    c = np.nan_to_num(ev["ctrl"], nan=0.0)
    if p.w_near:
        c = np.maximum(c, p.w_near * nearest_owner(ev, p))
    p_hc = c[:, HOME] * (1 - c[:, AWAY])
    p_ac = c[:, AWAY] * (1 - c[:, HOME])
    p_l = np.clip(1 - p_hc - p_ac, 0.0, 1.0)
    ball = np.stack([np.log(eps + p_hc), np.log(eps + p_l), np.log(eps + p_ac),
                     np.log(eps + p_l)], axis=1) - np.log(eps + 1.0)
    emit[known, :4] += p.w_ball * ball[known]

    if shape_q is not None:
        alive = shape_q[:, 0] + shape_q[:, 1]
        qh = np.where(alive > 0, shape_q[:, 0] / np.maximum(alive, 1e-9), 0.5)
        lh = np.log(p.shape_eps + qh) - np.log(p.shape_eps + 0.5)
        la = np.log(p.shape_eps + 1 - qh) - np.log(p.shape_eps + 0.5)
        w = np.where(known, p.w_shape_known, p.w_shape)
        emit[:, HOME * 2] += w * lh
        emit[:, HOME * 2 + 1] += w * lh
        emit[:, 2] += w * la
        emit[:, 3] += w * la

    if p.w_tag:
        team = np.tanh(tags["team"])
        emit[:, :2] += p.w_tag * team[:, None]
        emit[:, 2:4] -= p.w_tag * team[:, None]

    z = np.clip(dead_score(ev, shape_q, events, p), -p.z_max, p.z_max)
    pd = np.clip(sigmoid(z), 1e-4, 1 - 1e-4)
    emit[:, :4] += p.w_dead * np.log(1 - pd)[:, None]
    emit[:, 4] += p.w_dead * np.log(pd)
    return emit, pd


def add_whistles(evs, graded, p):
    """Referee whistle blasts from the audio track, if they are switched on."""
    if p.w_whistle or p.w_whistle_dead:
        from cv import whistle
        score = whistle.load(len(graded))[:len(graded)]
        evs["whistles"] = whistle.events(score, graded, p.whistle_thresh)
    return evs


def force_pre_kickoff(emit, game, p):
    """Make every non-D state impossible on stretches that are dead by the published timeline or
    by the laws of the game (structure.py): before a kickoff, between a goal and its kick-off."""
    from .structure import dead_intervals
    spans = dead_intervals(game, p)
    if not spans:
        return emit
    emit = emit.copy()
    for lo, hi in spans:
        emit[lo:hi, :4] = -1e6
    return emit


def _add(bonus, col, frame, value, before, after):
    lo, hi = max(0, frame - before), min(len(bonus), frame + after + 1)
    bonus[lo:hi, col] = np.maximum(bonus[lo:hi, col], value)


def bonuses(events, n, p, ev=None):
    bonus = np.zeros((n, 5), dtype=np.float64)
    if p.w_near_switch and ev is not None:
        near = nearest_owner(ev, p)
        near[near < 0.05] = 0.0  # keep the decoder's fast path on frames with nothing nearby
        bonus[:, TO_H] = p.w_near_switch * near[:, HOME]
        bonus[:, TO_A] = p.w_near_switch * near[:, AWAY]
    touch = events["touch"]
    weight = {"turnover": p.w_turnover, "touch": p.w_touch, "duel": p.w_duel,
              "pass": p.w_touch, "deflection": p.w_deflection}
    for f, team, conf, kind in zip(touch["frame"], touch["team"], touch["conf"], touch["kind"]):
        value = weight[kind] * conf
        if value > 0:
            _add(bonus, TO_H if team == HOME else TO_A, f, value, p.touch_before, p.touch_after)
    for f, conf, _ in events["stoppages"]:
        _add(bonus, DEAD_IN, f, p.w_stoppage * conf, p.stop_before, p.stop_after)
    for f in events["shots"]:
        _add(bonus, DEAD_IN, f, p.w_shot_dead, 0, p.shot_window)
    if p.w_whistle and "whistles" in events:
        for f in events["whistles"][0]:
            _add(bonus, DEAD_IN, f, p.w_whistle, p.whistle_before, p.whistle_after)
    for f, team, conf, _ in events["restarts"]:
        value = p.w_restart * conf
        if team in (HOME, -1):
            _add(bonus, H_OUT, f, value, p.restart_before, p.restart_after)
        if team in (AWAY, -1):
            _add(bonus, A_OUT, f, value, p.restart_before, p.restart_after)
    return bonus
