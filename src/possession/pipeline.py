import hashlib
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import atd as atd_io
from . import batches as batch_sel
from . import clean, evidence, events, features, film_features, fuse, homography, hsmm, metadata, model, shape, tagging, validator
from .config import Params
from .paths import CACHE_DIR, ROOT, SRC
from games import GAME, source_frames
import possession_metrics as pm

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
LOOP_PARAMS_PATH = os.path.join(SRC, "params.json")
CODES = {"H": 0, "A": 1, "D": 2}

CLEAN_KEYS = ("player_margin", "max_player_speed", "vel_k", "ball_spike_speed",
              "ball_island_len", "ball_island_gap", "max_gap")
EVIDENCE_KEYS = ("ctrl_radius", "ctrl_softness", "sync_free", "sync_scale", "filled_trust",
                 "corner_zone", "still_speed", "slow_window")


def _key(params, names):
    values = asdict(params)
    return tuple(values[k] for k in names)


class Inputs:
    def __init__(self, game=GAME, data_dir=None, refresh=False):
        self.game = game
        self.meta = metadata.load(game, data_dir, "tactical")
        self.atd = atd_io.load(game, data_dir, refresh)
        if len(self.atd["teams"]) != len(self.meta.slot_team) or (
            self.atd["teams"] != self.meta.slot_team
        ).any():
            raise ValueError("TXT slot count / teams disagree with the XML metadata")
        self.atd["teams"] = self.meta.slot_team
        self.atd["pitch"] = self.meta.pitch_m
        self.tags = tagging.priors(game, data_dir)
        self.blocks = self.meta.blocks
        self.n = source_frames("tactical", game)
        self.graded = np.zeros(self.n, dtype=bool)
        for _, first, last in self.blocks:
            self.graded[first:last + 1] = True
        self._data_dir = data_dir
        self._camera = {}
        self._features = {}
        self._cache = {}
        self._loop_state = {}

    def _get(self, name, key, build):
        full = (name,) + key
        if full not in self._cache:
            self._cache[full] = build()
        return self._cache[full]

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
            self._features[key] = features.build(self.atd, self.meta, cam, p.max_gap)
        return self._features[key]

    def tracking(self, p):
        return self._get("trk", _key(p, CLEAN_KEYS),
                         lambda: clean.build(self.atd, self.blocks, p))

    def evidence(self, p):
        key = _key(p, CLEAN_KEYS + EVIDENCE_KEYS)
        return self._get("ev", key, lambda: evidence.build(self.tracking(p), p))

    def shape_features(self, p):
        key = _key(p, CLEAN_KEYS + EVIDENCE_KEYS)
        return self._get("shape_x", key,
                         lambda: shape.features(self.tracking(p), self.evidence(p)))

    def shape_proba(self, p, labels, train):
        codes = np.array([CODES.get(c, -1) for c in labels], dtype=np.int8)
        X, names = self.shape_features(p)
        digest = hashlib.sha1(
            codes.tobytes()
            + train.tobytes()
            + repr(
                _key(
                    p,
                    CLEAN_KEYS
                    + EVIDENCE_KEYS
                    + ("shape_folds", "shape_guard", "shape_stride", "shape_iter"),
                )
            ).encode()
            + ",".join(names).encode()
        ).hexdigest()[:16]
        path = os.path.join(CACHE_DIR, "shape_{}_{}.npy".format(self.game, digest))
        if os.path.isfile(path):
            return np.load(path)
        q = shape.fit_predict(X, codes, train & (codes >= 0), self.graded, p)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, q)
        return q

    def v2_state(self, params):
        feat = self.features(params)
        ev = model.evidence(feat, self.tags, self.blocks, params)
        labels = model.decode(ev, self.blocks, params)
        memory = model.dead_memory(feat, self.blocks, params)
        return feat, ev, labels, memory

    def loop_state(self, loop_params=None):
        p = loop_params or Params.load(LOOP_PARAMS_PATH if os.path.isfile(LOOP_PARAMS_PATH) else None)
        key = repr(p)
        if key not in self._loop_state:
            labels, dbg = predict_loop(self, p, debug=True)
            hsmm_model = hsmm.Model(p)
            _, _, margin = hsmm_model.decode(dbg["emit"], dbg["bonus"], self.blocks, self.n, margins=True)
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


def shape_training_mask(inputs, ev, labels, p, margin=None):
    mask = inputs.graded & ev["seen"]
    if p.shape_conf > 0 and margin is not None:
        mask &= margin >= p.shape_conf
    change = np.zeros(len(labels), dtype=bool)
    change[1:] = labels[1:] != labels[:-1]
    near = np.convolve(change.astype(np.int8), np.ones(51, dtype=np.int8), mode="same") > 0
    return mask & ~near


def predict_loop(inputs, p, debug=False):
    trk = inputs.tracking(p)
    ev = inputs.evidence(p)
    evs = events.build(trk, ev, inputs.tags, inputs.blocks, p)
    evs["tag_starts"] = inputs.tags["sp_start"]
    hsmm_model = hsmm.Model(p)
    bonus = fuse.bonuses(evs, inputs.n, p)

    want = p.shape_conf > 0
    emit1, _ = fuse.emissions(ev, None, inputs.tags, evs, p)
    labels1, _, *margin = hsmm_model.decode(emit1, bonus, inputs.blocks, inputs.n, margins=want)

    labels = labels1
    rounds = max(1, p.shape_rounds)
    for k in range(rounds):
        train = shape_training_mask(inputs, ev, labels, p, margin[0] if margin else None)
        q = inputs.shape_proba(p, labels, train)
        emit2, p_dead = fuse.emissions(ev, q, inputs.tags, evs, p)
        labels, states, *margin = hsmm_model.decode(
            emit2, bonus, inputs.blocks, inputs.n, margins=want and k < rounds - 1
        )
    if not debug:
        return labels
    return labels, {
        "trk": trk, "ev": ev, "events": evs, "bonus": bonus, "emit": emit2,
        "p_dead": p_dead, "shape": q, "labels1": labels1, "states": states, "train": train,
    }


def public_mask(gt_codes):
    return gt_codes <= 2


def select_batches(inputs, params, gt_codes, cfg=batch_sel.BatchConfig(), loop_params=None):
    _, v2_ev, v2_lab, _ = inputs.v2_state(params)
    loop = inputs.loop_state(loop_params)
    mask_v2, reasons_v2 = batch_sel.select(v2_lab, v2_ev, inputs.blocks, public_mask(gt_codes), cfg)
    mask_loop, reasons_loop = batch_sel.select(
        loop["labels"], loop["score3"], inputs.blocks, public_mask(gt_codes), cfg
    )
    mask = mask_v2 | mask_loop
    reasons = {k: reasons_v2[k] | reasons_loop[k] for k in reasons_v2}
    return mask, reasons, loop["labels"]


def make_extractor(inputs, params, kind="film", **film_kwargs):
    from . import film
    feat = inputs.features(params)
    cam = inputs.camera(params.err_scale, params.pan_window)
    return film_features.Extractor(
        inputs.atd, feat, cam, film.load(kind, **film_kwargs), film.load("naive")
    )


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
            frame_fold = np.clip(
                np.searchsorted(edges, np.clip(pub, lo, hi), side="right") - 1, 0, n_folds - 1
            )
            for f in range(n_folds):
                train = validator.purged_train(self.train_frames, folds, f)
                m = factory().fit(x_train[train], self.y[train])
                sel = pub[frame_fold == f]
                if len(sel):
                    proba[sel] = validator._proba3(m, validator.matrix(self.columns, names, sel))
        return proba

    def decode(self, proba, weight=None, decoder="loop"):
        cfg = self.fusion if weight is None else validator.FusionConfig(
            weight=weight, p_floor=self.fusion.p_floor, peak_window=self.fusion.peak_window,
            v2_blend=self.fusion.v2_blend, conf_pow=self.fusion.conf_pow)
        if decoder == "v2":
            fused = validator.fuse(self.v2_evidence, proba, self.mask, cfg)
            return model.decode(fused, self.inputs.blocks, self.params)
        fused_emit = validator.fuse_hsmm(
            self.loop["emit"], proba, self.mask, cfg, v2_evidence=self.v2_evidence
        )
        labels, _ = self.hsmm_model.decode(fused_emit, self.loop["bonus"], self.inputs.blocks, self.n)
        return labels

    def evaluate(self, labels, bootstrap=0):
        return pm.evaluate(self.gt, pm.label_codes(labels), bootstrap_n=bootstrap)


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
