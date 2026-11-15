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
ENSEMBLE_VERSION = 2
ROUTED_ENSEMBLE_VERSION = 3
ENSEMBLE_VERSIONS = (ENSEMBLE_VERSION, ROUTED_ENSEMBLE_VERSION)
CURRENT = ('current_team', 'current_dead', 'current_H', 'current_A', 'current_D',
           'current_shape_H', 'current_shape_D')
# Discrete decoder outputs and durations can make the video head repeat an early
# mistake. The complementary head keeps the continuous evidence and video instead.
HARD_CONTEXT = ('current_H', 'current_A', 'current_D', 'v2_is_H', 'v2_is_A',
                'v2_is_D', 'v2_since_change', 'v2_until_change')
FEATURE_SETS = {
    'context': validator.V2_COLS + CURRENT,
    'flow': validator.V2_COLS + CURRENT + validator.FLOW_COLS + validator.NAIVE_COLS,
    'film': validator.V2_COLS + CURRENT + validator.FLOW_COLS + validator.FILM_COLS,
    'all': validator.V2_COLS + CURRENT + validator.FLOW_COLS + validator.NAIVE_COLS + validator.FILM_COLS,
}
FEATURE_SETS['film_soft'] = tuple(name for name in FEATURE_SETS['film'] if name not in HARD_CONTEXT)
MOTION = ('h_press_mean_d75', 'a_press_mean_d75', 'h_speed_d75', 'a_speed_d75',
          'h_y_std_s25', 'a_y_std_s25')
FEATURE_SETS['film_motion'] = FEATURE_SETS['film_soft'] + tuple('spatial_' + k for k in MOTION)


def covered_intervals(frames, mask, max_gap=25):
    """No feature interpolation across an unsampled hole or between half boundaries."""
    return batches.covered_intervals(frames,mask,max_gap)


def features(inputs, p, labels, emit, shape, kind='film', legacy_params=None, feature_names=()):
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
    if any(name.startswith('spatial_') for name in feature_names):
        raw, names = inputs.shape_features(p)
        for name in MOTION:
            columns['spatial_' + name] = raw[:, names.index(name)]
    return columns, mask, frames, asdict(vp)


def factory(name, seed=0, names=None, logreg_c=.3):
    def build_logreg():
        return validator.model_zoo(seed)['logreg']().set_params(logisticregression__C=logreg_c)
    if name in ('logreg', 'logreg_blend') and (not np.isfinite(logreg_c) or logreg_c <= 0):
        raise ValueError('logistic C must be finite and positive')
    if name == 'logreg':
        return build_logreg()
    if name == 'logreg_blend':
        if names is None:
            raise ValueError('feature names are required for the blended classifier')
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import VotingClassifier
        from sklearn.pipeline import make_pipeline
        indices = [i for i, column in enumerate(names) if column not in HARD_CONTEXT]
        if not indices:
            raise ValueError('the blended classifier needs continuous or video features')
        soft = make_pipeline(ColumnTransformer([('soft', 'passthrough', indices)]), build_logreg())
        return VotingClassifier([('original', build_logreg()), ('soft', soft)],
                                voting='soft', weights=[1., 3.])
    if name == 'hist_gbm':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7,
            min_samples_leaf=20, learning_rate=.05, l2_regularization=5.,
            early_stopping=False, random_state=seed)
    raise ValueError('unknown hybrid classifier: ' + name)


def fit(columns, sample_frames, gt_codes, names, model_name, kind, legacy_params,
        n_folds=5, gap=125, train_stride=25, logreg_c=.3):
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
            model = factory(model_name, names=names, logreg_c=logreg_c)
        model.fit(x[train], y[train])
        heads.append(model)
    full = factory(model_name, names=names, logreg_c=logreg_c).fit(x, y)
    return dict(version=VERSION, names=names, model_name=model_name, kind=kind,
        legacy_params=legacy_params, full=full, heads=heads, edges=edges,
        public_start=int(public.min()), public_stop=int(public.max()) + 1,
        purge_frames=gap, training_rows=len(rows), folds=n_folds, train_stride=train_stride,
        logreg_c=logreg_c if model_name in ('logreg', 'logreg_blend') else None)


def decode(inputs, p, baseline, emit, bonus, proba, mask):
    delta = probability_evidence(proba, mask, p.hybrid_weight, p.hybrid_dead_weight)
    updated = emit + delta[:, [0, 0, 1, 1, 2]]
    # Uncovered video cannot cause a label change through a distant Viterbi choice.
    for lab, cols in [('H', (0, 1)), ('A', (2, 3)), ('D', (4,))]:
        off = ~mask & (baseline != lab)
        for col in cols:
            updated[off, col] = -1e6
    lock_dead = p.hybrid_lock_dead
    if lock_dead and p.hybrid_dead_margin > 0:
        editable = dead_flexibility(baseline, mask, inputs.blocks,
                                    p.hybrid_dead_margin, p.dead_min_frames)
        dead = baseline == 'D'
        updated[dead & ~editable, :4] = -1e6
        updated[~dead & ~editable, 4] = -1e6
        lock_dead = 0
    allowed = (loose_eligibility(proba, mask, p.hybrid_loose_threshold)
               if p.hybrid_loose_threshold > 0 else None)
    if p.w_contact:
        labels = contact.refine(inputs, replace(p, contact_lock_dead=lock_dead),
                                 baseline, updated, bonus, loose_allowed=allowed)[0]
    else:
        if lock_dead:
            updated[baseline == 'D', :4] = -1e6
            updated[baseline != 'D', 4] = -1e6
        if allowed is not None:
            bonus = np.column_stack([bonus, np.where(allowed, bonus[:, :2], -np.inf)])
        labels = hsmm.Model(p, loose_switch_cost=p.switch_cost if allowed is not None else None).decode(
            updated, bonus, inputs.blocks, inputs.n)[0]
    labels[~mask] = baseline[~mask]
    if p.hybrid_prune_max_frames > 0:
        labels = prune_unconfirmed(labels, proba, mask, inputs.blocks,
                                   p.hybrid_prune_max_frames,
                                   p.hybrid_prune_min_confidence)
    return labels


def probability_evidence(proba, mask, team_weight, dead_weight=-1.):
    """Separate conditional team odds from alive/dead evidence.

    P(team) = P(alive) * P(team|alive). Separate weights can calibrate those two
    tasks independently. The shared-weight route retains the original arithmetic.
    """
    cfg = validator.FusionConfig(weight=team_weight)
    zero = np.zeros_like(proba, dtype=float)
    if dead_weight < 0 or dead_weight == team_weight:
        return validator.fuse(zero, proba, mask, cfg)
    if not np.isfinite(dead_weight):
        raise ValueError('dead evidence weight must be finite')
    q = np.clip(proba, cfg.p_floor, 1. - cfg.p_floor)
    alive = q[:, :2].sum(axis=1, keepdims=True)
    evidence = np.empty_like(zero)
    evidence[:, :2] = team_weight * np.log(q[:, :2] / alive) + dead_weight * np.log(alive)
    evidence[:, 2] = dead_weight * np.log(q[:, 2])
    evidence -= evidence.mean(axis=1, keepdims=True)
    valid = mask & np.isfinite(evidence).all(axis=1)
    zero[valid] = evidence[valid]
    return zero


def loose_eligibility(proba, mask, threshold):
    """A video-supported last touch may precede sustained control of the ball.

    Eligibility only opens a transition; the usual emissions, switch cost and
    duration constraints still decide the path. NaNs and absent video abstain.
    """
    if not 0 < threshold <= 1:
        raise ValueError('loose-switch confidence must be in (0, 1]')
    alive = proba[:, :2].sum(axis=1)
    valid = mask & np.isfinite(proba).all(axis=1) & (alive > 0)
    conditional = np.zeros((len(proba), 2), dtype=float)
    np.divide(proba[:, :2], alive[:, None], out=conditional, where=valid[:, None])
    return valid[:, None] & (conditional > threshold)


def dead_flexibility(baseline, mask, blocks, maximum, minimum_dead):
    """Editable edges of inferred D/alive phases, preserving a fixed phase centre.

    D phases retain a centre of min(existing length, minimum_dead) frames, so an
    OOF splice cannot shrink them below that length. Alive phases keep a centre
    frame, so adjustments cannot merge neighbouring stoppages across a short restart.
    """
    editable = np.zeros(len(baseline), dtype=bool)
    maximum = max(0, int(maximum))
    for _, first, last in blocks:
        dead = baseline[first:last + 1] == 'D'
        edges = np.r_[first, np.flatnonzero(dead[1:] != dead[:-1]) + first + 1, last + 1]
        for a, b in zip(edges[:-1], edges[1:]):
            sides = int(a > first) + int(b <= last)
            keep = max(1, int(minimum_dead)) if baseline[a] == 'D' else 1
            width = min(maximum, max(0, b - a - keep) // sides) if sides else 0
            if a > first:
                editable[a:a + width] = True
            if b <= last:
                editable[b - width:b] = True
    return editable & mask


def prune_unconfirmed(labels, proba, mask, blocks, maximum=100, minimum=.5):
    """Merge a short H/A/H or A/H/A island when the classifier prefers its neighbours.

    Confidence is conditional on the ball being alive. D, X, uncovered frames and
    half boundaries are never changed. This runs inside each independently decoded
    fold, before splicing: its confidence cannot come from another fold's model.
    """
    if maximum <= 0:
        return labels
    if not 0 <= minimum <= 1:
        raise ValueError('minimum spell confidence must be between zero and one')
    out = labels.copy()
    for _, first, last in blocks:
        changes = np.r_[first, np.flatnonzero(
            out[first + 1:last + 1] != out[first:last]) + first + 1, last + 1]
        for a, b in zip(changes[:-1], changes[1:]):
            if (a == first or b > last or b - a > maximum
                    or out[a] not in ('H', 'A') or out[a - 1] != out[b]
                    or out[b] not in ('H', 'A') or not mask[a:b].all()):
                continue
            alive = proba[a:b, :2].sum(axis=1)
            if not (np.isfinite(proba[a:b, :2]).all() and (alive > 0).all()):
                continue
            side = 0 if out[a] == 'H' else 1
            confidence = proba[a:b, side] / alive
            if confidence.mean() < minimum:
                out[a:b] = out[b]
    return out


def combine_artifacts(members, weights, team_router=None):
    """Combine aligned held-out heads; a member cannot borrow another fold's labels."""
    weights = np.asarray(weights, dtype=float)
    if (len(members) < 2 or weights.shape != (len(members),)
            or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('supply at least two models and finite nonnegative weights')
    reference = members[0]
    timeline = ('public_start','public_stop','folds','purge_frames','train_stride')
    for member in members:
        if member['version'] != VERSION:
            raise ValueError('ensemble members must be ordinary hybrid checkpoints')
        if (any(member[k] != reference[k] for k in timeline)
                or not np.array_equal(member['edges'], reference['edges'])
                or len(member['heads']) != reference['folds']):
            raise ValueError('ensemble members must share the same purged time folds')
    out = dict(version=ENSEMBLE_VERSION, model_name='ensemble', members=members,
               weights=weights / weights.sum(), edges=reference['edges'],
               **{k:reference[k] for k in timeline})
    if team_router is not None:
        if set(team_router) != {'expert', 'primary_max', 'expert_min'}:
            raise ValueError('team router needs expert, primary_max and expert_min')
        expert = team_router['expert']
        if (not isinstance(expert, (int, np.integer)) or isinstance(expert, bool)
                or not 0 <= expert < len(members) or weights[expert] != 0):
            raise ValueError('team expert must reference a zero-weight ensemble member')
        if not .5 <= team_router['primary_max'] <= team_router['expert_min'] <= 1.:
            raise ValueError('team router confidence thresholds must satisfy .5 <= primary <= expert <= 1')
        out['team_router'] = dict(team_router)
        out['version'] = ROUTED_ENSEMBLE_VERSION
    for key in ('tracking_ball_sha256','base_params'):
        if any(key in member for member in members):
            if not all(key in member for member in members):
                raise ValueError('ensemble provenance is incomplete: ' + key)
            def comparable(value):
                return ({k:v for k,v in value.items() if not k.startswith('hybrid_')}
                        if key == 'base_params' else value)
            if any(comparable(member[key]) != comparable(reference[key]) for member in members):
                raise ValueError('ensemble members use different baseline inputs: ' + key)
            out[key] = reference[key]
    return out


def prepare_features(inputs, p, baseline, emit, shape, artifact):
    if artifact['version'] == VERSION:
        return features(inputs,p,baseline,emit,shape,artifact['kind'],artifact['legacy_params'],
                        artifact['names'])
    if artifact['version'] in ENSEMBLE_VERSIONS:
        return [prepare_features(inputs,p,baseline,emit,shape,m) for m in artifact['members']]
    raise ValueError('hybrid checkpoint schema differs')


def route_teams(primary, expert, mask, primary_max=.8, expert_min=.9):
    """Use a confident expert only for uncertain conditional H/A; preserve P(D).

    Missing/invalid expert evidence abstains. The primary's confidence and all
    probability values outside the gate remain unchanged, including dead-ball odds.
    """
    if not .5 <= primary_max <= expert_min <= 1.:
        raise ValueError('team router confidence thresholds must satisfy .5 <= primary <= expert <= 1')
    valid = (mask & np.isfinite(primary).all(axis=1) & np.isfinite(expert).all(axis=1)
             & (primary >= 0).all(axis=1) & (expert >= 0).all(axis=1))
    alive, other_alive = primary[:, :2].sum(axis=1), expert[:, :2].sum(axis=1)
    valid &= (alive > 0) & (other_alive > 0)
    h = np.divide(primary[:, 0], alive, out=np.full(len(primary), .5), where=valid)
    other = np.divide(expert[:, 0], other_alive, out=np.full(len(primary), .5), where=valid)
    gate = valid & (np.maximum(h, 1-h) < primary_max) & (np.maximum(other, 1-other) > expert_min)
    out = primary.copy()
    out[gate, 0] = (1-primary[gate, 2])*other[gate]
    out[gate, 1] = (1-primary[gate, 2])*(1-other[gate])
    return out


def refine(inputs, p, baseline, emit, bonus, shape, artifact, prepared=None):
    router = None
    expert = expert_view = None
    if artifact['version'] == VERSION:
        members, weights = [artifact], np.array([1.])
        views = [prepared if prepared is not None else prepare_features(
            inputs,p,baseline,emit,shape,artifact)]
    elif artifact['version'] in ENSEMBLE_VERSIONS:
        all_members = artifact['members']
        checked = combine_artifacts(all_members, artifact['weights'], artifact.get('team_router'))
        if checked['version'] != artifact['version']:
            raise ValueError('ensemble schema and team router metadata differ')
        all_weights = checked['weights']
        router = checked.get('team_router')
        if prepared is not None and (not isinstance(prepared,list) or len(prepared) != len(all_members)):
            raise ValueError('ensemble inference needs one prepared feature view per member')
        active = np.flatnonzero(all_weights > 0)
        members = [all_members[i] for i in active]
        weights = all_weights[active]
        views = ([prepared[i] for i in active] if prepared is not None else
                 [prepare_features(inputs,p,baseline,emit,shape,m) for m in members])
        if router is not None:
            expert = all_members[router['expert']]
            expert_view = (prepared[router['expert']] if prepared is not None else
                           prepare_features(inputs,p,baseline,emit,shape,expert))
    else:
        raise ValueError('hybrid checkpoint schema differs')
    mask = np.logical_and.reduce([view[1] for view in views])
    rows = np.flatnonzero(mask)
    if not len(rows):
        return baseline.copy()
    matrices = [validator.matrix(view[0],member['names'],rows)
                for view,member in zip(views,members)]
    if expert is not None:
        expert_mask = mask & expert_view[1]
        expert_rows = np.flatnonzero(expert_mask)
        expert_x = validator.matrix(expert_view[0], expert['names'], expert_rows)
    def predict(models, expert_model=None):
        q = np.full((inputs.n, 3), np.nan)
        if len(models) == 1:
            q[rows] = validator._proba3(models[0],matrices[0])
        else:
            q[rows] = sum(weight * validator._proba3(model,x)
                          for weight,model,x in zip(weights,models,matrices))
        if expert_model is not None and len(expert_rows):
            other = np.full((inputs.n, 3), np.nan)
            other[expert_rows] = validator._proba3(expert_model, expert_x)
            q = route_teams(q, other, expert_mask, router['primary_max'], router['expert_min'])
        return decode(inputs,p,baseline,emit,bonus,q,mask)
    output = predict([member['full'] for member in members], expert['full'] if expert else None)
    # Independent whole-half decoding per held-out model prevents decoding neighbours
    # with a model trained on the held-out labels and bleeding those labels back in.
    reference = members[0]
    public = np.arange(reference['public_start'], reference['public_stop'])
    ids = np.clip(np.searchsorted(reference['edges'], public, side='right') - 1,
                  0, reference['folds'] - 1)
    for f in range(reference['folds']):
        labels = predict([member['heads'][f] for member in members], expert['heads'][f] if expert else None)
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
