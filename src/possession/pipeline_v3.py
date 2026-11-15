import os
import numpy as np

from . import batches as batch_sel
from . import film_features, hsmm, model_v2, pipeline as loop_pipeline, validator
from .config import Params
from .paths import SRC
from .pipeline_v2 import Inputs as InputsV2, match_summary, public_gt, score, write
from games import GAME, source_frames
import possession_metrics as pm

LOOP_PARAMS_PATH = os.path.join(SRC, "params.json")


class Inputs(InputsV2, loop_pipeline.Inputs):
    def __init__(self, game=GAME, data_dir=None, refresh=False):
        InputsV2.__init__(self, game=game, data_dir=data_dir, refresh=refresh)
        self.n = source_frames("tactical", game)
        self.graded = np.zeros(self.n, dtype=bool)
        for _, first, last in self.blocks:
            self.graded[first:last + 1] = True
        self._cache = {}
        self._loop_state = {}

    def v2_state(self, params):
        feat = self.features(params)
        evidence = model_v2.evidence(feat, self.tags, self.blocks, params)
        labels = model_v2.decode(evidence, self.blocks, params)
        memory = model_v2.dead_memory(feat, self.blocks, params)
        return feat, evidence, labels, memory

    def loop_state(self, loop_params=None):
        p = loop_params or Params.load(LOOP_PARAMS_PATH if os.path.isfile(LOOP_PARAMS_PATH) else None)
        key = repr(p)
        if key not in self._loop_state:
            labels, dbg = loop_pipeline.predict(self, p, debug=True)
            model = hsmm.Model(p)
            _, _, margin = model.decode(dbg["emit"], dbg["bonus"], self.blocks, self.n, margins=True)
            emit = dbg["emit"]
            score3 = np.column_stack([
                np.maximum(emit[:, 0], emit[:, 1]),
                np.maximum(emit[:, 2], emit[:, 3]),
                emit[:, 4],
            ])
            dbg["labels"] = labels
            dbg["margin"] = margin
            dbg["score3"] = score3
            dbg["params"] = p
            self._loop_state[key] = dbg
        return self._loop_state[key]


def public_mask(gt_codes):
    return gt_codes <= 2


def select_batches(inputs, params, gt_codes, cfg=batch_sel.BatchConfig(), loop_params=None):
    _, v2_ev, v2_lab, _ = inputs.v2_state(params)
    loop = inputs.loop_state(loop_params)
    mask_v2, reasons_v2 = batch_sel.select(v2_lab, v2_ev, inputs.blocks, public_mask(gt_codes), cfg)
    mask_loop, reasons_loop = batch_sel.select(loop["labels"], loop["score3"], inputs.blocks,
                                               public_mask(gt_codes), cfg)
    mask = mask_v2 | mask_loop
    reasons = {k: reasons_v2[k] | reasons_loop[k] for k in reasons_v2}
    return mask, reasons, loop["labels"]


def make_extractor(inputs, params, kind="film", **film_kwargs):
    from . import film
    feat = inputs.features(params)
    cam = inputs.camera(params.err_scale, params.pan_window)
    return film_features.Extractor(inputs.atd, feat, cam, film.load(kind, **film_kwargs), film.load("naive"))


class Layer:
    def __init__(self, inputs, params, gt_codes, store, batch_cfg=batch_sel.BatchConfig(),
                 fusion=validator.FusionConfig(), loop_params=None):
        self.inputs, self.params, self.fusion = inputs, params, fusion
        self.feat, self.v2_evidence, self.v2_labels, memory = inputs.v2_state(params)
        self.loop = inputs.loop_state(loop_params)
        self.loop_params = self.loop["params"]
        self.hsmm_model = hsmm.Model(self.loop_params)
        self.evidence = self.loop["score3"]
        self.labels = self.loop["labels"]
        self.n = len(self.labels)
        self.gt = gt_codes
        mask, self.reasons, _ = select_batches(inputs, params, gt_codes, batch_cfg, self.loop_params)
        frames, values = store.table()
        keep = mask[frames] if len(frames) else np.zeros(0, dtype=bool)
        self.sample_frames = frames[keep]
        values = {k: v[keep] for k, v in values.items()}
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
        context_v2 = validator.v2_context(self.feat, self.v2_evidence, self.v2_labels, inputs.blocks, memory)
        context_loop = validator.loop_context(self.loop)
        self.columns = {}
        self.columns.update(context_v2)
        self.columns.update(context_loop)
        self.columns.update(video)

        train_frames = self.sample_frames[public_mask(gt_codes)[self.sample_frames]]
        self.train_frames = train_frames
        self.y = gt_codes[train_frames].astype(int)

    def proba(self, names, factory, n_folds=5, oof=True):
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

    def decode(self, proba, weight=None, decoder="loop"):
        cfg = self.fusion if weight is None else validator.FusionConfig(
            weight=weight, p_floor=self.fusion.p_floor, peak_window=self.fusion.peak_window)
        if decoder == "v2":
            fused = validator.fuse(self.v2_evidence, proba, self.mask, cfg)
            return model_v2.decode(fused, self.inputs.blocks, self.params)
        fused_emit = validator.fuse_hsmm(self.loop["emit"], proba, self.mask, cfg)
        labels, _ = self.hsmm_model.decode(fused_emit, self.loop["bonus"], self.inputs.blocks, self.n)
        return labels

    def evaluate(self, labels, bootstrap=0):
        return pm.evaluate(self.gt, pm.label_codes(labels), bootstrap_n=bootstrap)
