"""v2 glue: TXT + XML metadata + homography + Smart Tagging in, labels out.

    metadata (XML) ─┐
    ATD (TXT) ──────┼──► features_v2 ──┐
    homography ─────┘                  ├──► model_v2.evidence ──► Viterbi per half ──► labels
    Smart Tagging ──► tagging.priors ──┘
"""

import numpy as np

from . import atd as atd_io
from . import features_v2, homography, metadata, model_v2, tagging
from .pipeline import write, public_gt, score  # noqa: F401  (re-exported)
from games import GAME
# pyrefly: ignore[missing-import]
import possession_metrics as pm


class Inputs:
    def __init__(self, game=GAME, data_dir=None, refresh=False):
        self.game = game
        self.meta = metadata.load(game, data_dir, "tactical")
        self.atd = atd_io.load(game, data_dir, refresh)
        if len(self.atd["teams"]) != len(self.meta.slot_team) or \
                (self.atd["teams"] != self.meta.slot_team).any():
            raise ValueError("TXT slot count / teams disagree with the XML metadata")
        self.atd["teams"] = self.meta.slot_team
        self.atd["pitch"] = self.meta.pitch_m
        self.tags = tagging.priors(game, data_dir)
        self.blocks = self.meta.blocks
        self._data_dir = data_dir
        self._camera = {}
        self._features = {}

    def camera(self, err_scale, pan_window):
        key = (err_scale, pan_window)
        if key not in self._camera:
            self._camera[key] = homography.Camera(
                self.game, self._data_dir, "tactical", self.meta.pitch_m, self.meta.fps,
                ball_norm=self.atd["ball"][:, :2], err_scale=err_scale, pan_window=pan_window)
        return self._camera[key]

    def features(self, p):
        key = (p.max_gap, p.err_scale, p.pan_window)
        if key not in self._features:
            cam = self.camera(p.err_scale, p.pan_window)
            self._features[key] = features_v2.build(self.atd, self.meta, cam, p.max_gap)
        return self._features[key]


def predict(inputs, params):
    feat = inputs.features(params)
    score_ = model_v2.evidence(feat, inputs.tags, inputs.blocks, params)
    return model_v2.decode(score_, inputs.blocks, params)


def match_summary(labels, blocks, fps=25):
    """Whole-match sanity numbers that need no ground truth: possessions, durations, D share."""
    codes = pm.label_codes(labels)
    rows = []
    for half, first, last in blocks:
        seg = codes[first:last + 1]
        change = np.flatnonzero(np.diff(seg)) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(seg)]))
        kinds = seg[starts]
        poss = (kinds == 0) | (kinds == 1)
        dur = (ends - starts)[poss] / float(fps)
        rows.append({
            "half": half, "possessions": int(poss.sum()),
            "mean_s": float(dur.mean()) if len(dur) else 0.0,
            "share_H": float((seg == 0).mean()), "share_A": float((seg == 1).mean()),
            "share_D": float((seg == 2).mean()),
        })
    return rows
