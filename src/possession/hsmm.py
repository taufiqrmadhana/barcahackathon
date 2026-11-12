"""Step 4: a temporal model over possession states, decoded with Viterbi.

Hidden states:

    H_ctrl  H_loose  A_ctrl  A_loose  D_1 ... D_m

The loose states keep the team of the last possession while the ball travels, bounces or is
contested, so a pass costs almost nothing. Possession can only change by entering the other
team's control state, which needs control evidence for that team, and is cheaper at a frame where
that team touches the ball. Dead ball is a chain of m states, so a stoppage lasts at least m
frames: an explicit minimum duration (the semi-Markov part) that stops a one-second "dead" blip
from splitting a possession in two. Leaving D goes straight into a control state, because a
restart always has a taker.
"""

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

HC, HL, AC, AL = 0, 1, 2, 3
GROUP_LABEL = np.array(["H", "H", "A", "A"])
NEG = -1e9


def _sparse_viterbi(emit, bonus, cost, sources, sizes, bonus_col, d_first, d_last):
    """Visit only legal arcs, preserving the dense decoder's first-index tie rule."""
    n, s = len(emit), len(cost)
    back = np.zeros((n, s), dtype=np.int16)
    total = np.empty(s, dtype=np.float64)
    for j in range(s):
        total[j] = emit[0, min(j, 4)]
    for j in range(d_first, d_last):
        total[j] = NEG
    for t in range(1, n):
        new = np.empty(s, dtype=np.float64)
        for j in range(s):
            best, previous = -np.inf, 0
            for k in range(sizes[j]):
                i = sources[j, k]
                c = cost[i, j]
                col = bonus_col[j, k]
                if col >= 0:
                    c = max(0.0, c - bonus[t, col])
                value = total[i] - c
                if value > best:
                    best, previous = value, i
            back[t, j] = previous
            new[j] = best + emit[t, min(j, 4)]
        total = new
    path = np.empty(n, dtype=np.int16)
    path[-1] = total.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


if njit is not None:
    _sparse_viterbi = njit(cache=True)(_sparse_viterbi)


class Model:
    def __init__(self, p):
        self.m = max(1, int(p.dead_min_frames))
        self.n_states = 4 + self.m
        self.d_first, self.d_last = 4, 4 + self.m - 1
        cost = np.full((self.n_states, self.n_states), np.inf)
        for s in range(4):
            cost[s, s] = 0.0
        cost[HC, HL] = cost[HL, HC] = p.loose_cost
        cost[AC, AL] = cost[AL, AC] = p.loose_cost
        cost[HL, AC] = cost[AL, HC] = p.switch_cost
        cost[HC, AC] = cost[AC, HC] = p.switch_cost + p.tackle_extra
        for s in range(4):
            cost[s, self.d_first] = p.dead_entry_cost
        for i in range(self.d_first, self.d_last):
            cost[i, i + 1] = 0.0
        cost[self.d_last, self.d_last] = 0.0
        cost[self.d_last, HC] = cost[self.d_last, AC] = p.dead_exit_cost
        self.cost = cost
        self.labels = np.concatenate([GROUP_LABEL, np.array(["D"] * self.m)])
        self.sizes = np.isfinite(cost).sum(axis=0)
        self.sources = np.zeros((self.n_states, int(self.sizes.max())), dtype=np.int64)
        self.bonus_col = np.full_like(self.sources, -1)
        for j in range(self.n_states):
            for k, i in enumerate(np.flatnonzero(np.isfinite(cost[:, j]))):
                self.sources[j, k] = i
                if j == HC and i in (AC, AL):
                    self.bonus_col[j, k] = 0
                elif j == AC and i in (HC, HL):
                    self.bonus_col[j, k] = 1
                elif j == self.d_first and i < 4:
                    self.bonus_col[j, k] = 2
                elif i == self.d_last and j in (HC, AC):
                    self.bonus_col[j, k] = 3 if j == HC else 4

    def expand(self, emit):
        """(N, 5) group emissions [HC, HL, AC, AL, D] -> (N, n_states)."""
        return np.hstack([emit[:, :4], np.repeat(emit[:, 4:5], self.m, axis=1)])

    def frame_cost(self, t, bonus):
        cost = self.cost.copy()
        to_h, to_a, dead_in, h_out, a_out = bonus[t]
        if to_h:
            cost[[AC, AL], HC] -= to_h
        if to_a:
            cost[[HC, HL], AC] -= to_a
        if dead_in:
            cost[:4, self.d_first] -= dead_in
        if h_out:
            cost[self.d_last, HC] -= h_out
        if a_out:
            cost[self.d_last, AC] -= a_out
        return np.maximum(cost, 0.0)

    def viterbi(self, emit, bonus):
        if njit is not None:
            return _sparse_viterbi(emit, bonus, self.cost, self.sources, self.sizes,
                                    self.bonus_col, self.d_first, self.d_last)
        return self.viterbi_reference(emit, bonus)

    def viterbi_reference(self, emit, bonus):
        """Best state path for one half. emit (N, 5), bonus (N, 5) cost reductions."""
        e = self.expand(emit)
        n, s = e.shape
        active = np.flatnonzero(np.abs(bonus).sum(axis=1) > 0)
        is_active = np.zeros(n, dtype=bool)
        is_active[active] = True
        base = -self.cost
        back = np.zeros((n, s), dtype=np.int16)
        total = e[0].copy()
        total[self.d_first:self.d_last] = NEG
        for t in range(1, n):
            trans = -self.frame_cost(t, bonus) if is_active[t] else base
            options = total[:, None] + trans
            back[t] = options.argmax(axis=0)
            total = options[back[t], np.arange(s)] + e[t]
        path = np.empty(n, dtype=np.int16)
        path[-1] = int(total.argmax())
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def margins(self, emit, bonus):
        """Per-frame confidence of the best path: its score minus the best score of any path
        with a different label (H / A / D) on that frame. Max-marginals, forward and backward."""
        e = self.expand(emit)
        n, s = e.shape
        is_active = np.abs(bonus).sum(axis=1) > 0
        base = -self.cost
        fwd = np.empty((n, s))
        fwd[0] = e[0]
        fwd[0, self.d_first:self.d_last] = NEG
        for t in range(1, n):
            trans = -self.frame_cost(t, bonus) if is_active[t] else base
            fwd[t] = (fwd[t - 1][:, None] + trans).max(axis=0) + e[t]
        bwd = np.zeros(s)
        groups = [np.flatnonzero(self.labels == lab) for lab in ("H", "A", "D")]
        out = np.empty(n)
        for t in range(n - 1, -1, -1):
            total = fwd[t] + bwd
            best = np.sort([total[g].max() for g in groups])
            out[t] = best[-1] - best[-2]
            if t:
                trans = -self.frame_cost(t, bonus) if is_active[t] else base
                bwd = (trans + (e[t] + bwd)[None, :]).max(axis=1)
        return out

    def decode(self, emit, bonus, blocks, n_frames, margins=False):
        labels = np.full(n_frames, "X", dtype="<U1")
        states = np.full(n_frames, -1, dtype=np.int16)
        margin = np.zeros(n_frames)
        for _, first, last in blocks:
            path = self.viterbi(emit[first:last + 1], bonus[first:last + 1])
            states[first:last + 1] = path
            labels[first:last + 1] = self.labels[path]
            if margins:
                margin[first:last + 1] = self.margins(emit[first:last + 1],
                                                      bonus[first:last + 1])
        if margins:
            return labels, states, margin
        return labels, states
