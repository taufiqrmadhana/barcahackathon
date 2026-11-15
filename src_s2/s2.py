"""S2: a provider-style ownership emulator.

The public GT is "D where the corrected tracking marks the ball dead, else the team the corrected
tracking names as owner". Tracking providers name an owner mechanically: the team of the last
player the ball was in contact with, held through flights and loose phases. This module emulates
that signal directly instead of modelling "controlled possession":

  1. ball   clean the ball track (drop teleports, interpolate short gaps)
  2. touch  contact events: ball close to a player AND (ball velocity changes, or the ball moves
            with the player for a few frames); team = that player's team
  3. owner  last-touch team, held through gaps; a switch needs the new team to touch the ball
  4. dead   out of the pitch, long ball-less stoppages with slow players, collective slowdowns
            with a stationary restart; ends at a restart kick (ball accelerates from a player)
  5. tidy   drop alive islands / dead islands below a minimum length (explainable cleanup)
"""
import json

import numpy as np

from data import FPS, HALVES

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


class Params(dict):
    __getattr__ = dict.__getitem__

    def on(self, key, default=1):
        """Component switch (ablation): `use_*` params, default = the S4 behaviour."""
        return bool(self.get(key, default))

    @classmethod
    def load(cls, path, **over):
        with open(path) as f:
            p = json.load(f)
        p.update(over)
        return cls(p)


# ---------------------------------------------------------------------------------------------- ball
def runs(mask):
    """(start, stop) of True runs."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def smooth(x, y, valid, p):
    """S8: Savitzky-Golay (order 2, window smooth_win frames) over every valid run (0 = off)."""
    sw = int(p.get("smooth_win", 0))
    if sw < 5:
        return x, y
    from scipy.signal import savgol_filter
    sx, sy = x.copy(), y.copy()
    for a, b in runs(valid):
        if b - a >= sw:
            sx[a:b] = savgol_filter(x[a:b], sw, 2, mode="interp")
            sy[a:b] = savgol_filter(y[a:b], sw, 2, mode="interp")
    return sx, sy


def velocity(x, y, m=2):
    """Central difference over +-m frames [m/s] (NaN at the ends / where the position is NaN)."""
    vx = np.full_like(x, np.nan)
    vy = np.full_like(y, np.nan)
    vx[m:-m] = (x[2 * m:] - x[:-2 * m]) / (2 * m) * FPS
    vy[m:-m] = (y[2 * m:] - y[:-2 * m]) / (2 * m) * FPS
    return vx, vy


def kick_span(sigma, p):
    """S8 noise-matched differencing span m (frames): the smallest m >= kick_k for which a velocity
    change of kick_dv is noise_z standard deviations of the noise in dv = |v(t+m) - v(t-m)|, with
    per-component std sigma*sqrt(6)*FPS/(2m) (2nd difference of iid position noise), 2-D: *sqrt(2).
    Clean tracks (sigma ~ 0.003 m) keep m = kick_k = 2: the S7b detector."""
    m = int(p.kick_k)
    if not p.on("kick_span_auto", 0):
        return m
    while m < p.kick_span_max and p.noise_z * sigma * np.sqrt(6) * np.sqrt(2) * FPS / (2 * m) > p.kick_dv:
        m += 1
    return m


def clean_ball(bx, by, p, info=None):
    """Drop teleports (points far from a 5-frame running median), interpolate gaps <= max_gap."""
    seen = ~np.isnan(bx)
    x, y = bx.copy(), by.copy()
    # running median over seen neighbours (+-2), teleport = point > teleport_m away from it
    idx = np.flatnonzero(seen)
    if len(idx) > 5 and p.on("use_teleport"):
        xs, ys = x[idx], y[idx]
        stack_x = np.stack([np.roll(xs, k) for k in range(-2, 3)])
        stack_y = np.stack([np.roll(ys, k) for k in range(-2, 3)])
        mx, my = np.median(stack_x, 0), np.median(stack_y, 0)
        far = np.hypot(xs - mx, ys - my) > p.teleport_m
        x[idx[far]] = np.nan
        y[idx[far]] = np.nan
    seen = ~np.isnan(x)
    valid = seen.copy()
    for a, b in runs(~seen):
        if a == 0 or b == len(x) or b - a > p.max_gap or not p.on("use_interp"):
            continue
        t = np.arange(1, b - a + 1) / (b - a + 1)
        x[a:b] = x[a - 1] + (x[b] - x[a - 1]) * t
        y[a:b] = y[a - 1] + (y[b] - y[a - 1]) * t
        valid[a:b] = True
    if info is not None:
        # noise level of the ball track: robust sigma of the 2nd difference of seen triples
        ok = seen[2:] & seen[1:-1] & seen[:-2]
        d2 = np.r_[(x[2:] - 2 * x[1:-1] + x[:-2])[ok], (y[2:] - 2 * y[1:-1] + y[:-2])[ok]]
        info["sigma"] = float(np.median(np.abs(d2)) / 0.6745 / np.sqrt(6)) if len(d2) else 0.0
    if info is not None and p.get("smooth_mult", 0) > 0:
        m = kick_span(info["sigma"], p)
        if m > p.kick_k:  # noisy track: smooth over ~ the differencing span (clean tracks: untouched)
            p = Params({**p, "smooth_win": int(p.smooth_mult * m + 1)})
    x, y = smooth(x, y, valid, p)
    vx, vy = velocity(x, y)
    return x, y, valid, seen, vx, vy


# --------------------------------------------------------------------------------------- contacts
def contacts(inp, x, y, vx, vy, p, m=None):
    """Per-frame contact team (H/A, -1 none) and per-team nearest distances."""
    teams = inp["teams"]
    px, py = inp["px"], inp["py"]
    dist = np.hypot(px - x[:, None], py - y[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    dH = np.where(teams[None, :] == 0, dist, np.inf).min(1)
    dA = np.where(teams[None, :] == 1, dist, np.inf).min(1)
    near = np.argmin(dist, 1)
    dmin = dist[np.arange(len(x)), near]
    team = teams[near].astype(np.int8)

    # player velocity of the nearest slot (central difference +-m), ball likewise
    k = m if m is not None else p.kick_k
    if k != 2:
        vx, vy = velocity(x, y, k)
    pvx, pvy = velocity(px, py, k)
    r = np.arange(len(x))
    rel = np.hypot(vx - pvx[r, near], vy - pvy[r, near])

    # ball velocity change (a kick / deflection / trap): |v(t+k) - v(t-k)|
    dv = np.zeros(len(x))
    dv[k:-k] = np.hypot(vx[2 * k:] - vx[:-2 * k], vy[2 * k:] - vy[:-2 * k])
    dv = np.nan_to_num(dv)
    # second-closest opponent margin: ambiguous if both teams equally close
    other = np.where(team == 0, dA, dH)
    with np.errstate(invalid="ignore"):
        clear = np.nan_to_num(other - dmin, nan=0.0, posinf=99.0) >= p.clear_margin

    kick = (dv >= p.kick_dv) & (dmin <= p.r_touch)
    # local maximum of dv within +-k
    from scipy.ndimage import maximum_filter1d
    if p.on("use_kick_localmax"):
        kick &= dv >= maximum_filter1d(dv, 2 * k + 1)
    kick_raw = kick.copy()  # restart detection in dead_mask uses this one
    gap_kick = np.zeros(len(x), bool)
    if p.on("use_gap_kick", 0):
        # S8: the kick detector needs the ball velocity BEFORE the strike, which a ball-less gap
        # hides. A ball that reappears (after >= mem_gap_before missing frames) within r_touch of a
        # player and leaves at > restart_speed is that player's touch (a regain or a restart).
        g = int(p.get("mem_gap_before", 25))
        ok = ~np.isnan(x)
        for r0, r1 in runs(ok):
            if r0 < g or ok[r0 - g:r0].any() or r1 - r0 < 6:
                continue
            fwd = np.hypot(x[r0 + 5] - x[r0], y[r0 + 5] - y[r0]) / 5 * FPS
            if fwd >= p.restart_speed and dmin[r0] <= p.r_touch:
                gap_kick[r0] = True
        kick = kick | gap_kick
    if not p.on("use_kick"):
        kick &= False
    carry = (dmin <= p.r_carry) & (np.nan_to_num(rel, nan=99) <= p.carry_rel)
    if not p.on("use_carry"):
        carry &= False
    if not p.on("use_clear"):
        clear = np.ones(len(x), bool)
    c = np.full(len(x), -1, np.int8)
    c[(carry | kick) & clear] = team[(carry | kick) & clear]
    c[np.isinf(dmin)] = -1
    return {"c": c, "kick": kick, "kick_raw": kick_raw, "gap_kick": gap_kick, "carry": carry, "dH": dH, "dA": dA, "dmin": dmin,
            "team": team, "dv": dv, "near": near}


def owner(c, p, n):
    """Last-contact team with confirmation: switch to T when T has >= confirm contact frames
    within the last `window` frames and the current owner has none in the last `quiet` frames."""
    own = np.full(n, -1, np.int8)
    cur = -1
    w, need, quiet = p.window, p.confirm, p.quiet
    cH = np.cumsum(np.r_[0, c == 0])
    cA = np.cumsum(np.r_[0, c == 1])
    for t in range(n):
        lo = max(0, t + 1 - w)
        nH = cH[t + 1] - cH[lo]
        nA = cA[t + 1] - cA[lo]
        if cur == -1:
            if nH >= need or nA >= need:
                cur = 0 if nH >= nA else 1
        else:
            o = 1 - cur
            n_o = nA if o == 1 else nH
            qlo = max(0, t + 1 - quiet)
            n_cur_q = (cH[t + 1] - cH[qlo]) if cur == 0 else (cA[t + 1] - cA[qlo])
            if n_o >= need and n_cur_q == 0:
                cur = o
        own[t] = cur
    return own


def contact_runs(c, kick, p):
    """Runs of same-team contact (gaps <= merge_gap bridged) that are long enough to count:
    >= min_run frames, or containing a kick (a single strike is a touch too)."""
    out = []
    for team in (0, 1):
        m = c == team
        # bridge short holes
        for a, b in runs(~m):
            if 0 < a and b < len(m) and b - a <= p.merge_gap and p.on("use_merge"):
                m[a:b] = True
        for a, b in runs(m):
            if (b - a >= p.min_run or not p.on("use_min_run")
                    or (kick[a:b].any() and p.on("use_kick_exempt"))):
                out.append((a, b, team))
    out.sort()
    return out


def owner_from_runs(cruns, n, p, gap_kick=None):
    """Owner = team of the contact runs; across a flight from team X (released at r) to team Y
    (first contact s) the switch is placed at r + alpha * (s - r): alpha 0 = the receiver owns the
    ball from the release (next-touch convention), alpha 1 = last-touch convention."""
    own = np.full(n, -1, np.int8)
    prev = None
    for a, b, team in cruns:
        if prev is None:
            own[a:b] = team
        else:
            pa, pb, pt = prev
            if team == pt:
                own[pb:b] = team
            else:
                s = max(a, pb)
                cut = int(round(pb + p.alpha * (s - pb)))
                if gap_kick is not None and gap_kick[a] and p.on("use_gap_cut"):
                    cut = s  # S8: a regain after a ball-less gap: the old owner holds until that touch
                own[pb:cut] = pt
                own[cut:b] = team
        # overlapping runs: later start wins from its start
        prev = (a, b, team) if prev is None or b >= prev[1] else prev
    # hold after the last run
    if prev is not None:
        own[prev[1]:] = prev[2]
    return own


# ------------------------------------------------------------------------------------------- dead
def dead_mask(inp, x, y, valid, seen, vx, vy, kick, p, gap_kick=None):
    """Frame-level dead ball from three mechanical signals, then min-duration cleanup.

    out    the (cleaned, seen) ball is beyond a pitch line by > out_margin metres; the stoppage
           lasts until play visibly resumes (players speed up again or a restart kick)
    still  players collectively slow: rolling mean player speed below a threshold (lower while
           the ball is seen, higher while it is missing -- the tactical feed loses the ball mostly
           in stoppages, so a missing ball is itself weak evidence of a dead ball)
    """
    from scipy.ndimage import uniform_filter1d
    n = inp["n"]
    pv = inp["pv"]
    cnt = np.sum(~np.isnan(pv), 1)
    sp = np.where(cnt > 0, np.nansum(pv, 1) / np.maximum(cnt, 1), np.nan)
    sp = np.where(np.isnan(sp), 0.0, sp)
    sm = uniform_filter1d(sp, p.speed_win)
    L, W, m = inp["L"], inp["W"], p.out_margin
    out = seen & ((x < -m) | (x > L + m) | (y < -m) | (y > W + m))
    if not p.on("use_out"):
        out = out & False
    bspeed = np.nan_to_num(np.hypot(vx, vy), nan=0.0)
    still_missing = sm < p.still_missing
    if not p.on("use_still_missing_speed"):
        still_missing = still_missing & False
    if inp.get("shape") is not None and p.get("shape_dead", 0) > 0 and p.on("use_shape_dead"):
        # ball missing: the team-shape model's P(dead) (players only, cross-fitted) replaces the
        # mean-speed rule, which fires in slow live phases (build-up) when the ball is lost
        pdead = uniform_filter1d(inp["shape"][:, 2], p.get("shape_win", 75))
        still_missing = (pdead > p.shape_dead) | (sm < p.get("still_missing_shape", 0))
    if p.on("use_still_long", 0):
        # S7 (151be54): ball missing and the 10 s mean player speed is walking pace (goal
        # celebrations, injuries, substitutions): the shape model has never seen these
        slow = uniform_filter1d(sp, p.get("still_long_win", 250)) < p.get("still_long", 1.0)
        still_missing = still_missing | slow
    still_seen = (sm < p.still_seen) & (uniform_filter1d(bspeed, p.speed_win) < p.ball_still)
    if not p.on("use_still_seen"):
        still_seen = still_seen & False
    if inp.get("shape") is not None and p.get("shape_dead_seen", 0) > 0 and p.on("use_shape_dead_seen"):
        # ball seen: a very confident shape-dead call (players set up for a free kick) also counts
        still_seen |= uniform_filter1d(inp["shape"][:, 2], p.get("shape_win", 75)) > p.shape_dead_seen
    still = np.where(seen, still_seen, still_missing)
    dead = still | out
    # out of play persists until the restart: the first kick (ball strike next to a player) with
    # the ball inside the pitch, after which the ball keeps moving; plus restart_lag frames (the
    # public GT turns alive ~1 s after the restart strike)
    kicks = np.flatnonzero(kick & seen & ~out)
    exits = [(a, b) for a, b in runs(out)]
    hold_max = p.out_hold_max
    if p.on("use_dead_memory", 0):
        # S7b dead memory (idea of Valdi's model_v2.dead_memory, re-implemented): the ball leaves
        # play -> dead until a restart. Differences to the out-hold above: (1) the hold lasts up to
        # mem_hold_max frames, (2) a restart is also "the ball starts moving fast again" (a kick
        # after a ball-less gap has no measurable velocity change), (3) a ball that disappears
        # next to a line for >= mem_lost_gap frames counts as having left play.
        hold_max = p.mem_hold_max
        if p.on("use_lost_exit"):
            Lx, Wy = inp["L"], inp["W"]
            si = np.flatnonzero(seen)
            for a, b in runs(~seen):
                if b - a < p.mem_lost_gap or a == 0 or b >= n:
                    continue
                j = si[np.searchsorted(si, a) - 1] if np.searchsorted(si, a) > 0 else None
                if j is None:
                    continue
                edge = min(x[j], Lx - x[j], y[j], Wy - y[j])
                if edge < p.mem_edge and np.mean(sp[a:min(b, a + p.get("mem_slow_win", 125))]) < p.get("mem_slow", 99):
                    exits.append((a, b))
            exits.sort()
    fast = np.array([], int)
    if p.on("use_dead_memory", 0) and p.on("use_fast_restart"):
        # restart candidates: fast ball motion in a seen run that starts after a ball-less gap
        # (the kick detector needs the ball velocity before the strike, which a gap hides)
        fast_l, mg = [], p.get("mem_gap_before", 25)
        for r0, r1 in runs(seen):
            if r0 == 0 or (~seen[max(0, r0 - mg):r0]).sum() < mg:
                continue
            ff = np.flatnonzero((bspeed[r0:r1] > p.get("mem_fast", 6.0)) & ~out[r0:r1])
            if len(ff):
                fast_l.append(r0 + ff[0])
        fast = np.array(fast_l, int)
    for a, b in exits:
        i = np.searchsorted(kicks, b)
        t = min(n, b + hold_max)
        if not p.on("use_out_hold"):
            t = b  # dead only while the ball is outside
            i = len(kicks)
        while i < len(kicks) and kicks[i] < t:
            k = kicks[i]
            if bspeed[min(n - 1, k + 5)] > p.restart_speed:
                t = min(n, k + (p.restart_lag if p.on("use_restart_lag") else 0))
                break
            i += 1
        if len(fast):
            # ball moving fast again after the exit: restart at the onset of that motion
            j = np.searchsorted(fast, a + p.get("mem_min_delay", 0))
            if j < len(fast) and fast[j] < t:
                f = fast[j]
                g = f
                while g > b and g > f - 50 and seen[g - 1] and bspeed[g - 1] > p.get("mem_rest", 1.0):
                    g -= 1  # onset of the motion (stays inside the seen run: unseen frames stop it)
                t = min(t, g + p.get("mem_lag", 0))
        # fallback: play visibly running again (ball inside and moving, players fast)
        run_again = np.flatnonzero(seen[b:t] & ~out[b:t] & (sm[b:t] >= p.resume_speed)
                                   & (bspeed[b:t] > p.restart_speed))
        if len(run_again):
            t = min(t, b + run_again[0])
        dead[a:t] = True
    if p.on("use_tag_dead", 0):
        # Smart Tagging set pieces: the tag starts when the ball is dead (throw-in, corner, goal
        # kick, free kick); a kick-off after a goal is the restart itself, dead just before it
        for ts, _te, _side, ty in inp.get("set_pieces", []):
            if "Kick Off" in (ty or ""):
                dead[max(0, ts - p.get("tag_ko_pre", 75)):ts] = True
            else:
                dead[max(0, ts - p.get("tag_pre", 50)):ts + p.get("tag_post", 50)] = True
    # cleanup: bridge short alive islands inside stoppages, drop short dead blips
    if p.on("use_min_alive"):
        for a, b in runs(~dead):
            if 0 < a and b < n and b - a < p.min_alive:
                if gap_kick is not None and gap_kick[a:b].any() and not out[a:b].any():
                    continue  # S8: an island that starts with a touch after a ball-less gap is a regain
                dead[a:b] = True
    if p.on("use_min_dead"):
        for a, b in runs(dead):
            if b - a < p.min_dead and not out[a:b].any():
                dead[a:b] = False
    return dead, {"sm": sm, "out": out, "still": still}


def assemble(own, dead, cruns, p, proba=None):
    """Labels from owner + dead. Each alive stretch starts with the restarting team (team of the
    first contact run inside it); flicker shorter than min_spell frames is merged into its
    neighbours."""
    n = len(own)
    lab = own.copy()
    starts = np.array([a for a, _, _ in cruns])
    teams = np.array([t for _, _, t in cruns])
    for a, b in runs(~dead):
        i = np.searchsorted(starts, a)
        has = i < len(starts) and starts[i] < b
        if has and proba is not None and starts[i] - a > p.get("restart_win", 10 ** 9)                 and p.on("use_shape_restart"):
            # first contact far after the restart (ball missing): the restarting team is the one
            # the team-shape model favours over the stretch before that contact
            pr = proba[a:starts[i], :2].mean(0)
            lab[a:starts[i]] = int(pr[1] > pr[0])
        elif has and p.on("use_first_contact"):
            lab[a:starts[i]] = teams[i]
        elif lab[a] == -1:
            lab[a:b] = 0 if a == 0 else lab[a - 1]
    # forward-fill any remaining unknown owner
    for a, b in runs(lab == -1):
        lab[a:b] = lab[a - 1] if a > 0 else 0
    lab[dead] = D
    # merge flicker: alive spells shorter than min_spell take the previous label
    for _ in range(2 if p.on("use_min_spell") else 0):
        ch =np.flatnonzero(np.diff(lab)) + 1
        bounds = np.r_[0, ch, n]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if lab[s] != D and e - s < p.min_spell and s > 0:
                lab[s:e] = lab[s - 1]
    return lab


def predict(inp, p, debug=False):
    if not p.on("use_shape") and inp.get("shape") is not None:
        inp = {**inp, "shape": None}  # ablation: no team-shape model at all
    bx, by = inp["bx"], inp["by"]
    if p.get("noise_sigma", 0) > 0:
        # stress test only (never in a submission): seeded iid Gaussian noise [m] on the seen ball
        rng = np.random.default_rng(int(p.get("noise_seed", 0)))
        bx = bx + rng.normal(0, p.noise_sigma, bx.shape)
        by = by + rng.normal(0, p.noise_sigma, by.shape)
    info = {}
    x, y, valid, seen, vx, vy = clean_ball(bx, by, p, info)
    m = kick_span(info["sigma"], p)
    info["span"] = m
    con = contacts(inp, x, y, vx, vy, p, m)
    cruns = contact_runs(con["c"], con["kick"], p)
    own = shift(owner_from_runs(cruns, inp["n"], p, con["gap_kick"]), p.lag)
    if inp.get("shape") is not None and p.get("shape_gap", 0) > 0 and p.on("use_shape_gap"):
        own = shape_gaps(own, valid, inp["shape"], cruns, p, con["gap_kick"])
    dead, dd = dead_mask(inp, x, y, valid, seen, vx, vy, con["kick_raw"], p, con["gap_kick"])
    lab = assemble(own, dead, cruns, p, inp.get("shape") if p.get("shape_gap", 0) > 0 else None)
    if debug:
        return lab, {"con": con, "cruns": cruns, "own": own, "dead": dead, **dd, "seen": seen}
    return lab


def shift(arr, lag, fill=-1):
    if lag == 0:
        return arr.copy()
    out = np.full_like(arr, fill)
    if lag > 0:
        out[lag:] = arr[:-lag]
    else:
        out[:lag] = arr[-lag:]
    return out


def shape_gaps(own, valid, proba, cruns, p, gap_kick=None):
    """Owner inside ball-missing gaps (>= shape_gap frames) from the team-shape model.

    Inside a gap the contact logic has nothing to say: it holds the last owner X and, if the next
    contact run belongs to the other team Y, cuts at alpha between release and that contact. With
    r_Y = smoothed P(Y) / (P(H) + P(A)) from the shape model (players only):
      X != Y  the switch goes to the start of the final stretch of the gap where r_Y > 0.5 (the
              moment the teams' shapes turned); if the shapes still favour X at the end, keep alpha.
      X == Y  an other-team spell is inserted only where r_other > shape_hi for >= shape_min frames
              (strong, sustained evidence: an A-B-A inside one gap is the over-switching pattern).
    """
    from scipy.ndimage import uniform_filter1d
    q = uniform_filter1d(proba[:, :2].astype(np.float64), p.get("shape_win", 75), axis=0)
    r = q[:, 1] / np.maximum(q.sum(1), 1e-9)
    n = len(own)
    out = own.copy()
    starts = np.array([a for a, _, _ in cruns] or [n])
    teams = np.array([t for _, _, t in cruns] or [-1])
    for a, b in runs(~valid):
        if b - a < p.shape_gap or a == 0:
            continue
        X = own[a - 1]
        j = np.searchsorted(starts, b)
        nxt, Y = (starts[j], teams[j]) if j < len(starts) else (n, X)
        if X < 0 or Y < 0:
            continue
        rY = r if Y == 1 else 1 - r
        if X != Y and gap_kick is not None and p.on("use_gap_cut") and j < len(starts) and gap_kick[starts[j]]:
            continue  # S8: regain after the gap, handled by owner_from_runs
        if X != Y and p.on("use_shape_gap_switch"):
            fav = rY[a:b] > 0.5
            if fav[-1]:
                k = np.flatnonzero(~fav)
                cut = a + (k[-1] + 1 if len(k) else 0)
                out[a:cut] = X
                out[cut:nxt] = Y
        elif X == Y and p.on("use_shape_gap_insert"):
            for s, e in runs(1 - rY[a:b] > p.get("shape_hi", 1.01)):
                if e - s >= p.get("shape_min", 75):
                    out[a + s:a + e] = 1 - X
    return out
