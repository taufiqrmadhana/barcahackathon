"""Per-frame evidence for H / A / D, decoded into labels with a switch-penalised Viterbi.

The decoder is what turns noisy frame evidence into possessions: changing state costs a fixed
amount, so a team only takes the ball once the evidence for it outweighs that cost. Loose balls
carry little evidence either way, so the last team keeps possession -- the rule CHALLENGE.md asks
for -- without any special case.
"""

import numpy as np

from .hsmm import njit

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


def evidence(feat, tags, p):
    """(N, 3) per-frame scores. Higher means more support for that state."""
    n = len(feat["dist"])
    score = np.zeros((n, 3), dtype=np.float64)

    # nearest player, faded by how far the nearest one is
    known = feat["ball_known"]
    d_home, d_away = feat["dist"][known, 0], feat["dist"][known, 1]
    nearest = np.minimum(d_home, d_away)
    margin = np.clip(np.nan_to_num(d_away - d_home, posinf=1e3, neginf=-1e3) / p.margin_scale,
                     -1.0, 1.0)
    weight = p.w_ball * np.exp(-nearest / p.control_radius)
    weight = np.where(feat["ball_filled"][known], weight * p.w_filled, weight)
    score[known, H] += weight * margin
    score[known, A] -= weight * margin

    # Smart Tagging: attacking team, and restarts
    team = np.tanh(tags["team"])
    score[:, H] += p.w_tag * team
    score[:, A] -= p.w_tag * team
    for start in tags["sp_start"]:
        score[start:start + p.sp_dead_frames, D] += p.w_sp_dead
    after = (tags["set_piece"] > 0) & (tags["sp_team"] != 0)
    score[after, H] += p.w_sp_team * tags["sp_team"][after]
    score[after, A] -= p.w_sp_team * tags["sp_team"][after]

    # dead ball
    long_missing = feat["missing_run"] > p.dead_gap
    slow = np.clip((p.slow_speed - feat["player_speed"]) / p.slow_speed, 0.0, 1.0)
    score[:, D] += p.d_bias
    score[feat["ball_outside"], D] += p.w_outside
    score[long_missing, D] += p.w_missing + p.w_missing_slow * slow[long_missing]
    score[:, D] += p.w_slow * slow
    moving = feat["ball_seen"] & (np.nan_to_num(feat["ball_speed"]) > p.alive_ball_speed)
    score[moving, D] -= p.w_alive
    return score


def viterbi_reference(score, switch_cost, dead_cost):
    """Most likely state path for one contiguous block of frames."""
    cost = np.array([[0.0, switch_cost, dead_cost],
                     [switch_cost, 0.0, dead_cost],
                     [dead_cost, dead_cost, 0.0]])
    n = len(score)
    back = np.zeros((n, 3), dtype=np.int8)
    # Every half starts with a kickoff, i.e. dead.
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


def _viterbi_fast(score, switch_cost, dead_cost):
    cost = np.array([[0.,switch_cost,dead_cost],[switch_cost,0.,dead_cost],
                     [dead_cost,dead_cost,0.]])
    n = len(score)
    back = np.zeros((n,3),dtype=np.int8)
    total = score[0] + np.array([-dead_cost,-dead_cost,0.])
    for t in range(1,n):
        new = np.empty(3)
        for j in range(3):
            best, previous = -np.inf, 0
            for i in range(3):
                value = total[i] - cost[i,j]
                if value > best:
                    best, previous = value, i
            back[t,j] = previous
            new[j] = best + score[t,j]
        total = new
    path = np.empty(n,dtype=np.int8)
    path[-1] = int(total.argmax())
    for t in range(n-1,0,-1):
        path[t-1] = back[t,path[t]]
    return path


viterbi = njit(cache=True)(_viterbi_fast) if njit is not None else viterbi_reference


def decode(score, blocks, p):
    """Labels for the whole video: decoded inside each half, X everywhere else."""
    labels = np.full(len(score), "X", dtype="<U1")
    for _, first, last in blocks:
        path = viterbi(score[first:last + 1], p.switch_cost, p.dead_cost)
        labels[first:last + 1] = LABELS[path]
    return labels
