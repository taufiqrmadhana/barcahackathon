"""Valdi's video/context validator on top of the current Taufiq + Neo decoder.

The public portion of the output is stitched from independently decoded held-out
folds. A fold's labels never enter any classifier used to decode that fold, including
neighbouring frames in its Viterbi pass. Outside the public window the all-public
model is used. Cached video features are local, never submission labels.
"""
from dataclasses import asdict, replace
import hashlib
from pathlib import Path

import joblib
import numpy as np

from . import batches, contact, hsmm, model_v2, pipeline_v2, validator
from .config_valdi import ParamsV2
from .film_features import Store
from .paths import SRC

VERSION = 1
CURRENT = ('current_team', 'current_dead', 'current_H', 'current_A', 'current_D',
           'current_shape_H', 'current_shape_D')
FEATURE_SETS = {
    'context': validator.V2_COLS + CURRENT,
    'flow': validator.V2_COLS + CURRENT + validator.FLOW_COLS + validator.NAIVE_COLS,
    'film': validator.V2_COLS + CURRENT + validator.FLOW_COLS + validator.FILM_COLS,
    'all': validator.V2_COLS + CURRENT + validator.FLOW_COLS + validator.NAIVE_COLS + validator.FILM_COLS,
}


def covered_intervals(frames, mask, max_gap=25):
    """No feature interpolation across an unsampled hole or between half boundaries."""
    return batches.covered_intervals(frames,mask,max_gap)


def features(inputs, p, labels, emit, shape, kind='film', legacy_params=None):
    legacy = pipeline_v2.Inputs(inputs.game)
    # In a masked experiment the context must see the caller's masked ATD.
    legacy.atd = dict(inputs.atd)
    vp = ParamsV2(**legacy_params) if legacy_params else ParamsV2.load(Path(SRC, 'params_v2.json'))
    feat = legacy.features(vp)
    ve = model_v2.evidence(feat, legacy.tags, legacy.blocks, vp)
    vl = model_v2.decode(ve, legacy.blocks, vp)
    memory = model_v2.dead_memory(feat, legacy.blocks, vp)
    columns = validator.v2_context(feat, ve, vl, legacy.blocks, memory)
    score = np.column_stack([emit[:, :2].max(axis=1), emit[:, 2:4].max(axis=1), emit[:, 4]])
    columns.update(current_team=score[:, 0] - score[:, 1],
                   current_dead=score[:, 2] - score[:, :2].max(axis=1),
                   current_H=(labels == 'H').astype(float),
                   current_A=(labels == 'A').astype(float),
                   current_D=(labels == 'D').astype(float),
                   current_shape_H=shape[:, 0] / np.maximum(shape[:, :2].sum(axis=1), 1e-9),
                   current_shape_D=shape[:, 2])
    store = Store(kind, inputs.game)
    frames, values = store.table()
    # Select without reading a ground-truth label, including throughout inference.
    # All available samples can supply stable anchors, not just transition batches.
    intervals = covered_intervals(frames, inputs.graded)
    mask = np.zeros(inputs.n, dtype=bool)
    for first, last in intervals:
        mask[first:last + 1] = True
    cols = validator.sample_table(frames, values, intervals,
                                  validator.FusionConfig(peak_window=0))
    video = validator.frame_video_features(frames, cols, intervals, inputs.n)
    # Native extraction may mix 5 Hz public samples with 1 Hz samples elsewhere.
    # Define peak windows in frames rather than a variable number of samples.
    from scipy.ndimage import maximum_filter1d
    for first, last in intervals:
        sl = slice(first, last + 1)
        for prefix in ('film', 'naive'):
            video[prefix + '_ball_peak'][sl] = maximum_filter1d(
                video[prefix + '_res_ball'][sl], 21, mode='nearest')
        video['film_team_peak_diff'][sl] = (
            maximum_filter1d(video['film_near_home'][sl], 21, mode='nearest')
            - maximum_filter1d(video['film_near_away'][sl], 21, mode='nearest'))
    columns.update(video)
    return columns, mask, frames, asdict(vp)


def factory(name, seed=0):
    if name == 'logreg':
        return validator.model_zoo(seed)['logreg']()
    if name == 'hist_gbm':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7,
            min_samples_leaf=20, learning_rate=.05, l2_regularization=5.,
            early_stopping=False, random_state=seed)
    raise ValueError('unknown hybrid classifier: ' + name)


def fit(columns, sample_frames, gt_codes, names, model_name, kind, legacy_params,
        n_folds=5, gap=125, train_stride=25):
    public = np.flatnonzero(gt_codes < 3)
    if not len(public) or train_stride < 1:
        raise ValueError('supply public labels and a positive training stride')
    grid = np.r_[np.arange(public.min(),public.max()+1,train_stride),public.max()]
    rows = np.intersect1d(sample_frames,grid)
    if len(rows) < 100:
        raise ValueError('extract at least 100 public video samples before fitting')
    x, y = validator.matrix(columns, names, rows), gt_codes[rows]
    edges = np.linspace(public.min(), public.max() + 1, n_folds + 1)
    folds = np.clip(np.searchsorted(edges, rows, side='right') - 1, 0, n_folds - 1)
    if np.any(np.bincount(folds,minlength=n_folds)==0):
        raise ValueError('video training samples must cover every public fold; complete the frame plan')
    heads = []
    for f in range(n_folds):
        train = validator.purged_train(rows, folds, f, gap)
        if np.unique(y[train]).size < 2:
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy='prior')
        else:
            model = factory(model_name)
        model.fit(x[train], y[train])
        heads.append(model)
    full = factory(model_name).fit(x, y)
    return dict(version=VERSION, names=names, model_name=model_name, kind=kind,
        legacy_params=legacy_params, full=full, heads=heads, edges=edges,
        public_start=int(public.min()), public_stop=int(public.max()) + 1,
        purge_frames=gap, training_rows=len(rows), folds=n_folds, train_stride=train_stride)


def decode(inputs, p, baseline, emit, bonus, proba, mask):
    delta = validator.fuse(np.zeros((inputs.n, 3)), proba, mask,
                            validator.FusionConfig(weight=p.hybrid_weight))
    updated = emit + delta[:, [0, 0, 1, 1, 2]]
    # Uncovered video cannot cause a label change through a distant Viterbi choice.
    for lab, cols in [('H', (0, 1)), ('A', (2, 3)), ('D', (4,))]:
        off = ~mask & (baseline != lab)
        for col in cols:
            updated[off, col] = -1e6
    if p.w_contact:
        labels = contact.refine(inputs, replace(p, contact_lock_dead=p.hybrid_lock_dead),
                                 baseline, updated, bonus)[0]
    else:
        if p.hybrid_lock_dead:
            updated[baseline == 'D', :4] = -1e6
            updated[baseline != 'D', 4] = -1e6
        labels = hsmm.Model(p).decode(updated, bonus, inputs.blocks, inputs.n)[0]
    labels[~mask] = baseline[~mask]
    return labels


def refine(inputs, p, baseline, emit, bonus, shape, artifact, prepared=None):
    if artifact['version'] != VERSION:
        raise ValueError('hybrid checkpoint schema differs')
    prepared = prepared or features(inputs, p, baseline, emit, shape,
                                     artifact['kind'], artifact['legacy_params'])
    columns, mask, _, _ = prepared
    rows = np.flatnonzero(mask)
    names = artifact['names']
    x = validator.matrix(columns, names, rows)
    def predict(model):
        q = np.full((inputs.n, 3), np.nan)
        q[rows] = validator._proba3(model, x)
        return decode(inputs, p, baseline, emit, bonus, q, mask)
    output = predict(artifact['full'])
    # Independent whole-half decoding per held-out model prevents decoding neighbours
    # with a model trained on the held-out labels and bleeding those labels back in.
    public = np.arange(artifact['public_start'], artifact['public_stop'])
    ids = np.clip(np.searchsorted(artifact['edges'], public, side='right') - 1,
                  0, artifact['folds'] - 1)
    for f, model in enumerate(artifact['heads']):
        labels = predict(model)
        held = public[ids == f]
        output[held] = labels[held]
    return output


def apply(inputs, p, baseline, emit, bonus, shape):
    path = Path(p.hybrid_model)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = inputs._get('hybrid_artifact', (digest,), lambda: joblib.load(path))
    if ('tracking_ball_sha256' in artifact and artifact['tracking_ball_sha256'] !=
            hashlib.sha256(inputs.atd['ball'].tobytes()).hexdigest()):
        raise ValueError('hybrid video crops use different ball tracking; re-extract for masked inputs')
    if 'base_params' in artifact:
        expected = {k:v for k,v in artifact['base_params'].items() if not k.startswith('hybrid_')}
        actual = {k:v for k,v in asdict(p).items() if not k.startswith('hybrid_')}
        if expected != actual:
            raise ValueError('hybrid checkpoint baseline parameters differ; refit for this configuration')
    return refine(inputs, p, baseline, emit, bonus, shape, artifact)
