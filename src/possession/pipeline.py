"""Data in, labels out. Shared by the CLI and the tuner."""

import os

import numpy as np
import pandas as pd

from . import atd as atd_io
from . import features, model, tagging
from .paths import ROOT
from games import GAME, half_video_blocks, source_frames
import possession_metrics as pm

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))


class Inputs:
    """Loaded data plus feature sets cached by max_gap, so tuning only rebuilds what changed."""

    def __init__(self, game=GAME, data_dir=None, refresh=False):
        self.game = game
        self.atd = atd_io.load(game, data_dir, refresh)
        self.tags = tagging.priors(game, data_dir)
        self.blocks = half_video_blocks(game)
        self._features = {}

    def features(self, max_gap):
        if max_gap not in self._features:
            self._features[max_gap] = features.build(self.atd, max_gap)
        return self._features[max_gap]


def predict(inputs, params):
    feat = inputs.features(params.max_gap)
    score = model.evidence(feat, inputs.tags, params)
    return model.decode(score, inputs.blocks, params)


def write(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    table = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    problem = pm.validate_submission(table, source_frames("tactical"))
    if problem:
        raise ValueError("invalid submission: " + problem)
    table.to_csv(path, index=False, lineterminator="\n")
    return path


def public_gt():
    return pm.label_codes(pm.read_label_csv(PUBLIC_GT))


def score(labels, gt_codes=None):
    gt_codes = public_gt() if gt_codes is None else gt_codes
    return pm.evaluate(gt_codes, pm.label_codes(labels))
