"""v3 glue: v2 predictions, validated by a FILM frame-interpolation layer on selected batches.

    v2 (TXT + XML + homography + Smart Tagging) ──► evidence, labels
            │
            ▼
    batches (transitions, doubt, per-label anchors, public window)
            │
            ▼
    tactical video ──► FILM(t-2, t+2) vs real t, ball-centred crops ──► residual features (5 Hz)
            │
            ▼
    model zoo, contiguous-block CV on the public labels ──► best model ──► p(H, A, D)
            │
            ▼
    v2 evidence + w * log p on batch frames ──► Viterbi per half ──► v3 labels
"""

import numpy as np

from . import batches as batch_sel
from . import film_features, model_v2, validator
from .pipeline_v2 import Inputs as InputsV2, match_summary, public_gt, score, write  # noqa: F401
import possession_metrics as pm


class Inputs(InputsV2):
    """v2 inputs plus the v2 state the validation layer reads."""

    def v2_state(self, params):
        feat = self.features(params)
        evidence = model_v2.evidence(feat, self.tags, self.blocks, params)
        labels = model_v2.decode(evidence, self.blocks, params)
        memory = model_v2.dead_memory(feat, self.blocks, params)
        return feat, evidence, labels, memory


def public_mask(gt_codes):
    return gt_codes <= 2  # H, A, D; X (and anything else) is not a training label


def select_batches(inputs, params, gt_codes, cfg=batch_sel.BatchConfig()):
    _, evidence, labels, _ = inputs.v2_state(params)
    mask, reasons = batch_sel.select(labels, evidence, inputs.blocks, public_mask(gt_codes), cfg)
    return mask, reasons, labels


def make_extractor(inputs, params, kind="film"):
    from . import film
    feat = inputs.features(params)
    cam = inputs.camera(params.err_scale, params.pan_window)
    return film_features.Extractor(inputs.atd, feat, cam, film.load(kind), film.load("naive"))


class Layer:
    """Everything the validation layer needs, built once and reused by the ablation."""

    def __init__(self, inputs, params, gt_codes, store, batch_cfg=batch_sel.BatchConfig(),
                 fusion=validator.FusionConfig()):
        self.inputs, self.params, self.fusion = inputs, params, fusion
        self.feat, self.evidence, self.labels, memory = inputs.v2_state(params)
        self.n = len(self.labels)
        self.gt = gt_codes
        mask, self.reasons = batch_sel.select(self.labels, self.evidence, inputs.blocks,
                                              public_mask(gt_codes), batch_cfg)
        frames, values = store.table()
        keep = mask[frames] if len(frames) else np.zeros(0, dtype=bool)
        self.sample_frames = frames[keep]
        values = {k: v[keep] for k, v in values.items()}
        # only frames whose batch actually has FILM samples take part
        covered = np.zeros(self.n, dtype=bool)
        intervals = batch_sel.intervals(mask)
        self.intervals = []
        for first, last in intervals:
            s = self.sample_frames[(self.sample_frames >= first) & (self.sample_frames <= last)]
            if len(s) >= 2:
                a, b = int(s[0]), int(s[-1])
                covered[a:b + 1] = True
                self.intervals.append((a, b))
        self.mask = covered
        self.batch_mask = mask
        cols = validator.sample_table(self.sample_frames, values, self.intervals, fusion)
        video = validator.frame_video_features(self.sample_frames, cols, self.intervals, self.n)
        context = validator.v2_context(self.feat, self.evidence, self.labels, inputs.blocks, memory)
        self.columns = dict(context, **video)

        train_frames = self.sample_frames[public_mask(gt_codes)[self.sample_frames]]
        self.train_frames = train_frames
        self.y = gt_codes[train_frames].astype(int)

    # ------------------------------------------------------------------ fitting

    def proba(self, names, factory, n_folds=5, oof=True):
        """p(H, A, D) on every covered frame: out-of-fold inside the public window if oof."""
        x_train = validator.matrix(self.columns, names, self.train_frames)
        rows = np.flatnonzero(self.mask)
        x_all = validator.matrix(self.columns, names, rows)
        proba = np.full((self.n, 3), np.nan)
        full = factory().fit(x_train, self.y)
        proba[rows] = validator._proba3(full, x_all)
        if oof:
            pub = rows[public_mask(self.gt)[rows]]
            lo, hi = self.train_frames.min(), self.train_frames.max()
            edges = np.quantile(self.train_frames, np.linspace(0, 1, n_folds + 1))
            folds = validator.contiguous_folds(self.train_frames, n_folds)
            frame_fold = np.clip(np.searchsorted(edges, np.clip(pub, lo, hi), side="right") - 1,
                                 0, n_folds - 1)
            for f in range(n_folds):
                train = validator.purged_train(self.train_frames, folds, f)
                m = factory().fit(x_train[train], self.y[train])
                sel = pub[frame_fold == f]
                if len(sel):
                    proba[sel] = validator._proba3(m, validator.matrix(self.columns, names, sel))
        return proba

    def decode(self, proba, weight=None):
        cfg = self.fusion if weight is None else validator.FusionConfig(
            weight=weight, p_floor=self.fusion.p_floor, peak_window=self.fusion.peak_window)
        fused = validator.fuse(self.evidence, proba, self.mask, cfg)
        return model_v2.decode(fused, self.inputs.blocks, self.params)

    def evaluate(self, labels, bootstrap=0):
        return pm.evaluate(self.gt, pm.label_codes(labels), bootstrap_n=bootstrap)
