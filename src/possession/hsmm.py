import numpy as np

HC, HL, AC, AL = 0, 1, 2, 3
GROUP_LABEL = np.array(["H", "H", "A", "A"])
NEG = -1e9


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

    def expand(self, emit):
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
