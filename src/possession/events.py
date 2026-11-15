import numpy as np

from .clean import runs

HOME, AWAY = 0, 1


def _peaks(signal, threshold, spacing):
    cand = np.flatnonzero(signal > threshold)
    if not len(cand):
        return cand
    order = cand[np.argsort(-signal[cand], kind="stable")]
    taken = np.zeros(len(signal), dtype=bool)
    keep = []
    for i in order:
        lo, hi = max(0, i - spacing), i + spacing + 1
        if taken[lo:hi].any():
            continue
        taken[i] = True
        keep.append(i)
    return np.array(sorted(keep), dtype=np.int64)


def touches(trk, ev, p):
    vel = trk["ball"]["vel"]
    k = p.touch_k
    dv = np.zeros(len(vel), dtype=np.float32)
    dv[k:-k] = np.hypot(*(vel[2 * k:] - vel[:-2 * k]).T)
    dv = np.nan_to_num(dv)
    dv[~trk["ball"]["seen"]] = 0.0
    frames = _peaks(dv, p.touch_dv, p.touch_spacing)

    dist = ev["dist"][frames]
    nearest = dist.min(axis=1)
    team = np.where(dist[:, HOME] <= dist[:, AWAY], HOME, AWAY)
    close = nearest < p.touch_radius
    frames, team, dist, nearest = frames[close], team[close], dist[close], nearest[close]
    gap = np.abs(dist[:, HOME] - dist[:, AWAY])
    conf = np.exp(-nearest / p.touch_radius) * np.clip(gap / p.touch_margin, 0.0, 1.0)
    duel = dist.max(axis=1) < p.duel_radius
    return {"frame": frames, "team": team, "conf": conf, "dv": dv[frames], "duel": duel}


def sequences(touch, ball_xy, blocks, p):
    n = len(touch["frame"])
    kind = np.array(["touch"] * n, dtype=object)
    half = np.zeros(n, dtype=np.int8)
    for h, first, last in blocks:
        half[(touch["frame"] >= first) & (touch["frame"] <= last)] = h
    for i in range(1, n):
        f0, f1 = touch["frame"][i - 1], touch["frame"][i]
        if half[i] != half[i - 1] or f1 - f0 > p.seq_max_gap:
            continue
        same = touch["team"][i] == touch["team"][i - 1]
        travel = np.hypot(*(ball_xy[f1] - ball_xy[f0])) if not np.isnan(ball_xy[[f0, f1], 0]).any() \
            else 0.0
        if same:
            if travel >= p.pass_min_travel:
                kind[i] = "pass"
            continue
        nxt = i + 1 < n and half[i + 1] == half[i] and touch["frame"][i + 1] - f1 <= p.seq_max_gap
        if nxt and touch["team"][i + 1] == touch["team"][i - 1]:
            kind[i] = "deflection"
        elif touch["duel"][i]:
            kind[i] = "duel"
        else:
            kind[i] = "turnover"
    return kind


def shots(trk, touch, p):
    length, width = trk["pitch"]
    xy, vel = trk["ball"]["xy"], trk["ball"]["vel"]
    out = []
    for f, team in zip(touch["frame"], touch["team"]):
        after = min(f + p.touch_k + 2, len(xy) - 1)
        v = vel[after]
        if np.isnan(v).any() or np.hypot(*v) < p.shot_speed:
            continue
        sign = trk["attack"][f] * (1 if team == HOME else -1)
        goal_x = length if sign > 0 else 0.0
        if abs(goal_x - xy[f, 0]) > p.shot_range:
            continue
        to_goal = np.array([goal_x - xy[f, 0], width / 2.0 - xy[f, 1]])
        cos = np.dot(v, to_goal) / (np.hypot(*v) * np.hypot(*to_goal) + 1e-6)
        if cos > 0.85:
            out.append(f)
    return np.array(out, dtype=np.int64)


def _team_at(dist, f):
    return int(np.argmin(dist[f])) if np.isfinite(dist[f]).any() else -1


def stoppages(trk, ev, tags, p):
    out = []
    seen = trk["ball"]["seen"]
    beyond = np.where(seen, np.nan_to_num(ev["beyond"], nan=-99.0), -99.0)
    over = beyond > p.out_margin
    crossed = beyond > 0.0
    xy, vel = trk["ball"]["xy"], trk["ball"]["vel"]
    length, width = trk["pitch"]
    for start, stop in zip(*runs(over)):
        if stop - start < p.out_min_frames:
            continue
        first = start
        while first > 0 and crossed[first - 1]:
            first -= 1
        x, y = xy[start + int(np.argmax(beyond[start:stop]))]
        goal_line = max(-x, x - length) > max(-y, y - width)
        out.append((first, 1.0, "goal_line" if goal_line else "touchline"))

    for f in np.flatnonzero(seen[:-1] & ~seen[1:]):
        if ev["missing_run"][f + 1] < p.vanish_min_frames or np.isnan(vel[f]).any():
            continue
        nx, ny = xy[f] + vel[f] * p.vanish_horizon
        if nx < 0 or nx > length or ny < 0 or ny > width:
            out.append((int(f), 0.6, "vanish"))

    ps = ev["player_speed_smooth"]
    w = p.decel_window
    drop = np.zeros(len(ps))
    drop[w:-w] = ps[:-2 * w] - ps[2 * w:]
    after = np.zeros(len(ps))
    after[:-w] = ps[w:]
    decel = (drop > p.decel_drop) & (after < p.decel_to)
    for start, stop in zip(*runs(decel)):
        f = start + int(np.argmax(drop[start:stop]))
        out.append((max(0, f - p.decel_lead), 0.5, "whistle"))

    for f in tags["sp_start"]:
        out.append((int(f), 0.4, "tag"))
    out.sort()
    return out


def restarts(trk, ev, p):
    out = []
    speed = ev["ball_speed_smooth"]
    dist = ev["dist"]
    seen = trk["ball"]["seen"]
    n = len(speed)
    rest = seen & (speed < p.rest_speed)
    for start, stop in zip(*runs(rest)):
        if stop - start < p.placed_min_frames or stop >= n - p.kick_within:
            continue
        kick = stop + np.flatnonzero(speed[stop:stop + p.kick_within] > p.restart_kick_speed)
        if not len(kick):
            continue
        calm = ev["player_speed_smooth"][start:stop].mean() < p.placed_calm_speed
        out.append((int(stop), _team_at(dist, stop), 1.0 if calm else 0.6, "placed"))

    beyond = np.where(seen, np.nan_to_num(ev["beyond"], nan=-99.0), -99.0)
    back_in = np.flatnonzero((beyond[:-1] > 0.3) & (beyond[1:] <= 0.0) & seen[:-1] & seen[1:])
    for f in back_in:
        out.append((int(f + 1), _team_at(dist, f), 0.8, "throw"))

    for f in np.flatnonzero(~seen[:-1] & seen[1:]) + 1:
        if ev["missing_run"][f - 1] < p.reappear_min_frames:
            continue
        out.append((int(f), _team_at(dist, f), 0.5, "reappear"))
    out.sort()
    return out


def dead_intervals(stops, starts, p):
    out = []
    frames = np.array([r[0] for r in starts], dtype=np.int64)
    min_gap = {"goal_line": p.hold_goal_line, "touchline": p.hold_touchline}
    for f, conf, kind in stops:
        if kind not in p.hold_kinds.split(","):
            continue
        i = np.searchsorted(frames, f + min_gap.get(kind, p.dead_min_gap))
        if i >= len(frames) or frames[i] > f + p.dead_max_gap:
            continue
        r = starts[i]
        out.append((f, r[0], conf, r[1]))
    return out


def build(trk, ev, tags, blocks, p):
    touch = touches(trk, ev, p)
    touch["kind"] = sequences(touch, trk["ball"]["xy"], blocks, p)
    stops, starts = stoppages(trk, ev, tags, p), restarts(trk, ev, p)
    return {
        "touch": touch,
        "shots": shots(trk, touch, p),
        "stoppages": stops,
        "restarts": starts,
        "dead": dead_intervals(stops, starts, p),
    }
