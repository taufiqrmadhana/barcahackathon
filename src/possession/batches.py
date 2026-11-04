"""Which parts of the video the FILM layer looks at.

Running FILM on 199k full-HD frames is out of reach on a laptop, and most frames sit in the middle
of a possession where v2 is not in doubt. The validation layer therefore works on *batches*:
contiguous frame intervals chosen from the v2 output, so that every label is represented.

    transition   +-pad around every v2 change of label (H<->A, alive<->D): where v2 is most
                 likely to be wrong, and where spellF1 is decided
    doubt        runs where the 1 s mean v2 evidence prefers another state than the decoded one
    anchor       stable windows from the middle of v2 H, A and D runs, so the model sees each
                 label in its ordinary form and not only at its edges
    public       the public ground-truth window, which supplies the training labels

The union is merged into intervals and sampled every `stride` frames.
"""

from dataclasses import dataclass

import numpy as np

LABEL_CODE = {"H": 0, "A": 1, "D": 2}


@dataclass
class BatchConfig:
    pad: int = 75              # frames either side of a v2 transition (3 s)
    doubt_window: int = 25     # frames, smoothing of v2 evidence for the doubt test
    doubt_min: int = 10        # shortest doubt run kept
    doubt_pad: int = 25
    anchors_per_label: int = 40
    anchor_len: int = 100      # 4 s
    stride: int = 5            # FILM sample every 5th frame (5 Hz)
    seed: int = 7


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _moving_mean(x, w):
    kernel = np.ones(w) / w
    return np.stack([np.convolve(x[:, k], kernel, mode="same") for k in range(x.shape[1])], 1)


def select(labels, score, blocks, public=None, cfg=BatchConfig()):
    """Returns (mask over all frames, dict reason -> mask)."""
    n = len(labels)
    codes = np.array([LABEL_CODE.get(l, -1) for l in labels])
    in_half = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        in_half[first:last + 1] = True
    reasons = {k: np.zeros(n, dtype=bool) for k in ("transition", "doubt", "anchor", "public")}
    rng = np.random.default_rng(cfg.seed)

    for _, first, last in blocks:
        seg = codes[first:last + 1]
        # transitions
        for c in np.flatnonzero(np.diff(seg)) + 1 + first:
            reasons["transition"][max(first, c - cfg.pad):min(last + 1, c + cfg.pad)] = True
        # doubt: smoothed evidence prefers another state than the decoded one
        smooth = _moving_mean(score[first:last + 1], cfg.doubt_window)
        chosen = smooth[np.arange(len(seg)), seg]
        doubt = smooth.max(axis=1) > chosen + 1e-9
        for a, b in _runs(doubt):
            if b - a >= cfg.doubt_min:
                reasons["doubt"][max(first, first + a - cfg.doubt_pad):
                                 min(last + 1, first + b + cfg.doubt_pad)] = True
        # anchors: middles of long runs of each label
    for code in range(3):
        runs = [(a, b) for a, b in _runs((codes == code) & in_half)
                if b - a >= cfg.anchor_len + 2 * cfg.pad]
        if not runs:
            continue
        pick = rng.choice(len(runs), size=min(cfg.anchors_per_label, len(runs)), replace=False)
        for i in pick:
            a, b = runs[i]
            mid = (a + b) // 2
            reasons["anchor"][mid - cfg.anchor_len // 2:mid + cfg.anchor_len // 2] = True

    if public is not None:
        reasons["public"] = public & in_half

    mask = np.zeros(n, dtype=bool)
    for m in reasons.values():
        mask |= m
    return mask & in_half, reasons


def intervals(mask):
    return [(int(a), int(b) - 1) for a, b in _runs(mask)]


def samples(mask, stride):
    """Sample frames: every `stride`-th frame of each interval, always including its last frame."""
    out = []
    for first, last in intervals(mask):
        s = np.arange(first, last + 1, stride)
        if s[-1] != last:
            s = np.append(s, last)
        out.append(s)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def describe(mask, reasons, labels):
    lines = ["batches: {} intervals, {} frames ({:.1%} of the halves' frames)".format(
        len(intervals(mask)), int(mask.sum()), mask.sum() / max(1, (labels != "X").sum()))]
    for k, m in reasons.items():
        lines.append("  {:<10} {:>7} frames".format(k, int(m.sum())))
    lab = labels[mask]
    lines.append("  v2 labels inside: " + "  ".join(
        "{} {:.1%}".format(c, (lab == c).mean()) for c in "HAD"))
    return "\n".join(lines)
