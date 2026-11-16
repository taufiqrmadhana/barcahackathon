"""Conservative temporal agreement inside one independently decoded video fold.

These operations use predictions and conditional team probabilities, never labels
from an evaluation file. Both callers must decode the same held-out head before
public slices are stitched; a stitched auxiliary sequence would leak across folds.
"""
import numpy as np


def overlay_dead_mask(primary, phase, mask):
    """Take phase D decisions while retaining established alive-team labels.

    Newly exposed alive frames use the phase branch's team. This pointwise
    operation runs inside each matching held-out head and preserves X/coverage.
    """
    out = primary.copy()
    valid = mask & np.isin(primary, ['H', 'A', 'D']) & np.isin(phase, ['H', 'A', 'D'])
    out[valid & (phase == 'D')] = 'D'
    removed = valid & (primary == 'D') & (phase != 'D')
    out[removed] = phase[removed]
    return out


def _edges(labels, first, last):
    return np.r_[first, np.flatnonzero(
        labels[first + 1:last + 1] != labels[first:last]) + first + 1, last + 1]


def remove_oscillations(labels, proba, mask, blocks, maximum=100):
    """Merge a short H/A/H triplet when pooled neighbour confidence is stronger.

All three spells must be short and have complete video coverage. Both neighbours
must support their shared team above chance. One pass selects non-overlapping
triplets by confidence gain, so a merge cannot trigger a chain of further merges.
"""
    out = labels.copy()
    if maximum <= 0:
        return out
    for _, first, last in blocks:
        edges = _edges(labels, first, last)
        proposals = []
        for i in range(1, len(edges) - 2):
            left, start, stop, right = map(int, edges[i - 1:i + 3])
            middle, neighbour = labels[start], labels[left]
            if (middle not in ('H', 'A') or neighbour not in ('H', 'A')
                    or labels[stop] != neighbour or middle == neighbour
                    or max(start - left, stop - start, right - stop) > maximum
                    or not mask[left:right].all()):
                continue
            q = proba[left:right, :2]
            alive = q.sum(axis=1)
            if not (np.isfinite(q).all() and (alive > 0).all()):
                continue
            conditional = q / alive[:, None]
            side = 0 if middle == 'H' else 1
            a, b = start - left, stop - left
            own = conditional[a:b, side].mean()
            before = conditional[:a, 1 - side].mean()
            after = conditional[b:, 1 - side].mean()
            pooled = (before * a + after * (right - stop)) / (a + right - stop)
            if min(before, after) > .5 and pooled > own:
                proposals.append((pooled - own, left, start, stop, right, neighbour))
        used = np.zeros(len(labels), dtype=bool)
        for _, left, start, stop, right, team in sorted(proposals, reverse=True):
            if not used[left:right].any():
                out[start:stop] = team
                used[left:right] = True
    return out


def extend_agreed_spells(labels, auxiliary, mask, blocks, maximum=100, minimum_iou=.5):
    """Use the union of two overlapping short spells of the same team.

The union must remain short, contain only alive primary labels, have complete
coverage of both models, and meet the specified IoU. This adjusts existing short
spells; it cannot invent an isolated possession or move a D/X or half boundary.
"""
    if maximum <= 0 or not 0 < minimum_iou <= 1:
        raise ValueError('spell consensus needs positive length and IoU in (0, 1]')
    out = labels.copy()
    for _, first, last in blocks:
        def short_spells(values):
            edges = _edges(values, first, last)
            return [(int(a), int(b), values[a]) for a, b in zip(edges[:-1], edges[1:])
                    if values[a] in ('H', 'A') and b - a <= maximum]
        proposals = []
        for start, stop, team in short_spells(labels):
            for a, b, other in short_spells(auxiliary):
                if other != team:
                    continue
                intersection = min(stop, b) - max(start, a)
                left, right = min(start, a), max(stop, b)
                if (intersection <= 0 or right - left > maximum
                        or intersection / (right - left) < minimum_iou
                        or not mask[left:right].all()
                        or not np.isin(labels[left:right], ('H', 'A')).all()):
                    continue
                proposals.append((intersection / (right - left), start, stop, a, b, team))
        used = np.zeros(len(labels), dtype=bool)
        for _, start, stop, a, b, team in sorted(proposals, reverse=True):
            left, right = min(start, a), max(stop, b)
            if not used[left:right].any():
                out[left:right] = team
                used[left:right] = True
    return out
