"""Per-frame camera-gated evidence for H / A / D and switch-penalised Viterbi decoder."""

import numpy as np

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dead_memory(feat, blocks, p):
    """1 from the moment the ball leaves play until the first fast restart kick, else 0."""
    n = len(feat["ball_seen"])
    out = np.zeros(n, dtype=np.float32)
    seen = feat["ball_seen"]
    speed = np.where(seen, np.nan_to_num(feat["ball_speed"]), 0.0)
    lost_near_line = (~seen) & (feat["since_seen"] >= p.dead_gap) & (feat["last_edge_m"] < p.edge_margin)
    exits = feat["ball_outside"] | lost_near_line
    restart = seen & (speed > p.restart_speed) & ~feat["ball_outside"]
    for _, first, last in blocks:
        active, since = False, 0
        for t in range(first, last + 1):
            if exits[t]:
                active, since = True, 0
            elif active:
                since += 1
                if (since >= p.restart_min_delay and restart[t]) or since > p.memory_max:
                    active = False
            if active:
                out[t] = 1.0
    return out


def evidence(feat, tags, blocks, p):
    """(N, 3) per-frame scores for H, A, D."""
    n = len(feat["dist"])
    score = np.zeros((n, 3), dtype=np.float64)
    quality = feat["quality"]

    z = np.nan_to_num(feat["z_player_speed"], nan=0.0)
    slow = _sigmoid(-(z - p.z_slow) / p.z_temp)

    known = feat["ball_known"]
    d_home, d_away = feat["dist"][known, 0], feat["dist"][known, 1]
    nearest = np.minimum(d_home, d_away)
    margin = np.clip(np.nan_to_num(d_away - d_home, posinf=1e3, neginf=-1e3) / p.margin_scale,
                     -1.0, 1.0)
    weight = p.w_ball * np.exp(-nearest / p.control_radius)
    weight = np.where(feat["ball_filled"][known], weight * p.w_filled, weight)
    weight *= (1.0 - p.team_gate * slow[known]) * (0.5 + 0.5 * quality[known])
    score[known, H] += weight * margin
    score[known, A] -= weight * margin

    team = np.tanh(tags["team"])
    score[:, H] += p.w_tag * team
    score[:, A] -= p.w_tag * team
    for start in tags["sp_start"]:
        score[max(0, start - p.sp_pre_frames):start + p.sp_dead_frames, D] += p.w_sp_dead
    after = (tags["set_piece"] > 0) & (tags["sp_team"] != 0)
    score[after, H] += p.w_sp_team * tags["sp_team"][after]
    score[after, A] -= p.w_sp_team * tags["sp_team"][after]

    long_missing = feat["missing_run"] > p.dead_gap
    score[:, D] += p.d_bias
    score[feat["ball_outside"], D] += p.w_outside
    score[long_missing, D] += p.w_missing + p.w_missing_slow * slow[long_missing]
    moving = feat["ball_seen"] & (np.nan_to_num(feat["ball_speed"]) > p.alive_ball_speed)
    score[moving, D] -= p.w_alive

    score[:, D] += p.w_slow_z * slow
    cam_still = np.clip((p.cam_still_speed - np.nan_to_num(feat["pan_speed"], nan=0.0))
                        / p.cam_still_speed, 0.0, 1.0)
    score[:, D] += p.w_cam_still * cam_still * slow * quality
    score[:, D] += p.w_memory * dead_memory(feat, blocks, p)
    return score


def viterbi(score, switch_cost, dead_cost):
    """Most likely state path for one contiguous block of frames."""
    cost = np.array([[0.0, switch_cost, dead_cost],
                     [switch_cost, 0.0, dead_cost],
                     [dead_cost, dead_cost, 0.0]])
    n = len(score)
    back = np.zeros((n, 3), dtype=np.int8)
    total = score[0] + np.array([-dead_cost, -dead_cost, 0.0])
    for t in range(1, n):
        options = total[:, None] - cost
        back[t] = options.argmax(axis=0)
        total = options.max(axis=0) + score[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(total.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(score, blocks, p):
    """Labels for the whole video: decoded inside each half, X everywhere else."""
    labels = np.full(len(score), "X", dtype="<U1")
    for _, first, last in blocks:
        path = viterbi(score[first:last + 1], p.switch_cost, p.dead_cost)
        labels[first:last + 1] = LABELS[path]
    return labels
