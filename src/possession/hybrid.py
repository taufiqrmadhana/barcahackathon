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

from . import batches, contact, hsmm, model_v2, pipeline_v2, spell_consensus, validator, video_provenance
from .config_valdi import ParamsV2
from .film_features import Store
from .paths import SRC

VERSION = 1
ENSEMBLE_VERSION = 2
ROUTED_ENSEMBLE_VERSION = 3
MULTI_ROUTED_ENSEMBLE_VERSION = 4
TEMPORAL_ENSEMBLE_VERSION = 5
DEAD_ROUTED_ENSEMBLE_VERSION = 6
PHASE_ENSEMBLE_VERSION = 7
ENSEMBLE_VERSIONS = (ENSEMBLE_VERSION, ROUTED_ENSEMBLE_VERSION,
                     MULTI_ROUTED_ENSEMBLE_VERSION, TEMPORAL_ENSEMBLE_VERSION,
                     DEAD_ROUTED_ENSEMBLE_VERSION, PHASE_ENSEMBLE_VERSION)
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
FEATURE_SETS['film_dead_motion'] = tuple(k for k in FEATURE_SETS['film_motion']
    if k.startswith(('film_', 'flow_', 'spatial_')) or k in
    ('ball_seen', 'ball_filled', 'since_seen', 'quality', 'players_in_view', 'pan_speed'))
from .broadcast import FEATURES as BROADCAST_FEATURES
FEATURE_SETS['broadcast_dead'] = BROADCAST_FEATURES


def covered_intervals(frames, mask, max_gap=25):
    """No feature interpolation across an unsampled hole or between half boundaries."""
    return batches.covered_intervals(frames,mask,max_gap)


def features(inputs, p, labels, emit, shape, kind='film', legacy_params=None, feature_names=()):
    legacy = pipeline_v2.Inputs(inputs.game)
    # In a masked experiment the context must see the caller's masked ATD.
    legacy.atd = dict(inputs.atd)
    vp = ParamsV2(**legacy_params) if legacy_params else ParamsV2.load(Path(SRC, 'params_v2.json'))
    video_provenance.describe(inputs, p, kind, asdict(vp))
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
    if any(name.startswith('broadcast_') for name in feature_names):
        from . import broadcast
        extra, coverage, _ = broadcast.features(inputs, p)
        columns.update(extra)
        mask &= coverage
        frames = frames[coverage[frames]]
    if any(name.startswith('spatial_') for name in feature_names):
        raw, names = inputs.shape_features(p)
        for name in MOTION:
            columns['spatial_' + name] = raw[:, names.index(name)]
    return columns, mask, frames, asdict(vp)


def factory(name, seed=0, names=None, logreg_c=.3):
    def build_logreg():
        return validator.model_zoo(seed)['logreg']().set_params(logisticregression__C=logreg_c)
    if name in ('logreg', 'logreg_blend', 'dead_logreg') and (not np.isfinite(logreg_c) or logreg_c <= 0):
        raise ValueError('logistic C must be finite and positive')
    if name == 'logreg':
        return build_logreg()
    if name == 'dead_logreg':
        from .dead_head import DeadBallClassifier
        return DeadBallClassifier(C=logreg_c)
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
        n_folds=5, gap=125, train_stride=25, logreg_c=.3, strict_fold_edges=False):
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
        train = ((rows < edges[f]-gap) | (rows >= edges[f+1]+gap) if strict_fold_edges else
                 validator.purged_train(rows, folds, f, gap))
        if np.unique(y[train]).size < 2:
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy='prior')
        else:
            model = factory(model_name, names=names, logreg_c=logreg_c)
        model.fit(x[train], y[train])
        heads.append(model)
    full = factory(model_name, names=names, logreg_c=logreg_c).fit(x, y)
    artifact = dict(version=VERSION, names=names, model_name=model_name, kind=kind,
        legacy_params=legacy_params, full=full, heads=heads, edges=edges,
        public_start=int(public.min()), public_stop=int(public.max()) + 1,
        purge_frames=gap, training_rows=len(rows), folds=n_folds, train_stride=train_stride,
        logreg_c=logreg_c if model_name in ('logreg', 'logreg_blend', 'dead_logreg') else None)
    if strict_fold_edges:
        artifact['strict_fold_edges'] = True
    return artifact


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


def combine_artifacts(members, weights, team_router=None, team_routers=None, temporal=None,
                      dead_router=None, phase_router=None):
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
    if team_router is not None and team_routers is not None:
        raise ValueError('supply one router or an ordered router list, not both')
    if team_routers is not None and (not isinstance(team_routers, (list, tuple)) or len(team_routers) < 2):
        raise ValueError('an ordered router list needs at least two experts')
    routers = team_routers if team_routers is not None else ([team_router] if team_router is not None else [])
    seen = set()
    for router in routers:
        if set(router) != {'expert', 'primary_max', 'expert_min'}:
            raise ValueError('team router needs expert, primary_max and expert_min')
        expert = router['expert']
        if (not isinstance(expert, (int, np.integer)) or isinstance(expert, bool)
                or not 0 <= expert < len(members) or weights[expert] != 0):
            raise ValueError('team expert must reference a zero-weight ensemble member')
        if expert in seen:
            raise ValueError('ordered team experts must be distinct')
        seen.add(expert)
        if not .5 <= router['primary_max'] <= router['expert_min'] <= 1.:
            raise ValueError('team router confidence thresholds must satisfy .5 <= primary <= expert <= 1')
    if team_router is not None:
        out['team_router'] = dict(team_router)
        out['version'] = ROUTED_ENSEMBLE_VERSION
    elif team_routers is not None:
        out['team_routers'] = [dict(router) for router in routers]
        out['version'] = MULTI_ROUTED_ENSEMBLE_VERSION
    if temporal is not None:
        required = {'oscillation_max_frames', 'consensus_weights',
                    'consensus_max_frames', 'consensus_min_iou'}
        if set(temporal) not in (required, required | {'consensus_dead_weight'}):
            raise ValueError('temporal consensus metadata differs')
        for key, minimum in [('oscillation_max_frames', 0), ('consensus_max_frames', 1)]:
            value = temporal[key]
            if (not isinstance(value, (int, np.integer)) or isinstance(value, bool)
                    or value < minimum):
                raise ValueError('temporal spell lengths must be nonnegative integers')
        auxiliary = np.asarray(temporal['consensus_weights'], dtype=float)
        if (auxiliary.shape != weights.shape or not np.isfinite(auxiliary).all()
                or (auxiliary < 0).any() or auxiliary.sum() <= 0):
            raise ValueError('supply finite nonnegative consensus weights for every member')
        if not 0 < temporal['consensus_min_iou'] <= 1:
            raise ValueError('consensus IoU must be in (0, 1]')
        dead_weight = temporal.get('consensus_dead_weight', 0.)
        if not 0 <= dead_weight <= 1:
            raise ValueError('consensus dead probability weight must be in [0, 1]')
        out['temporal'] = dict(temporal, consensus_weights=auxiliary / auxiliary.sum(),
                               consensus_dead_weight=dead_weight)
        out['version'] = TEMPORAL_ENSEMBLE_VERSION
    if dead_router is not None:
        if set(dead_router) != {'expert', 'weight'}:
            raise ValueError('dead router needs expert and weight')
        expert = dead_router['expert']
        if (not isinstance(expert, (int, np.integer)) or isinstance(expert, bool)
                or not 0 <= expert < len(members) or weights[expert] != 0):
            raise ValueError('dead expert must reference a zero-weight ensemble member')
        if not 0 <= dead_router['weight'] <= 1:
            raise ValueError('dead expert weight must be in [0, 1]')
        if temporal is not None and out['temporal']['consensus_weights'][expert] != 0:
            raise ValueError('dead expert must not supply auxiliary team probabilities')
        out['dead_router'] = dict(dead_router)
        out['version'] = DEAD_ROUTED_ENSEMBLE_VERSION
    if phase_router is not None:
        if temporal is None or dead_router is None:
            raise ValueError('phase refinement requires the temporal and dead-head baseline')
        if set(phase_router) != {'expert', 'weight', 'dead_margin', 'keep_teams'}:
            raise ValueError('phase router metadata differs')
        expert = phase_router['expert']
        if (not isinstance(expert, (int, np.integer)) or isinstance(expert, bool)
                or not 0 <= expert < len(members) or weights[expert] != 0
                or expert == dead_router['expert'] or expert in seen
                or out['temporal']['consensus_weights'][expert] != 0):
            raise ValueError('phase expert must be a separate zero-weight member')
        margin = phase_router['dead_margin']
        if (not isinstance(margin, (int, np.integer)) or isinstance(margin, bool) or margin < 1
                or not isinstance(phase_router['keep_teams'], bool)
                or not 0 <= phase_router['weight'] <= 1):
            raise ValueError('phase refinement needs a positive margin and a weight in [0, 1]')
        out['phase_router'] = dict(phase_router)
        out['version'] = PHASE_ENSEMBLE_VERSION
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
        video_provenance.check(inputs, p, artifact)
        if any(name.startswith('broadcast_') for name in artifact['names']):
            from . import broadcast
            broadcast.check(inputs, p, artifact)
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


def blend_dead_probability(primary, auxiliary, mask, weight):
    """Average only P(D), preserving primary conditional H/A odds exactly.

    The auxiliary is another same-fold prediction. Missing evidence, all-dead
    primary rows with undefined H/A odds, and uncovered frames abstain.
    """
    if not 0 <= weight <= 1:
        raise ValueError('auxiliary dead probability weight must be in [0, 1]')
    out = primary.copy()
    if weight == 0:
        return out
    alive = primary[:, :2].sum(axis=1)
    valid = (mask & np.isfinite(primary).all(axis=1) & np.isfinite(auxiliary).all(axis=1)
             & (alive > 0) & (primary[:, 2] >= 0) & (primary[:, 2] <= 1)
             & (auxiliary[:, 2] >= 0) & (auxiliary[:, 2] <= 1))
    dead = (1 - weight) * primary[valid, 2] + weight * auxiliary[valid, 2]
    out[valid, :2] *= ((1 - dead) / alive[valid])[:, None]
    out[valid, 2] = dead
    return out


def refine(inputs, p, baseline, emit, bonus, shape, artifact, prepared=None):
    experts = []
    auxiliary_members, auxiliary_weights, auxiliary_views = [], [], []
    temporal = None
    dead_router, dead_member, dead_view = None, None, None
    phase_router, phase_member, phase_view = None, None, None
    if artifact['version'] == VERSION:
        members, weights = [artifact], np.array([1.])
        views = [prepared if prepared is not None else prepare_features(
            inputs,p,baseline,emit,shape,artifact)]
    elif artifact['version'] in ENSEMBLE_VERSIONS:
        all_members = artifact['members']
        checked = combine_artifacts(all_members, artifact['weights'], artifact.get('team_router'),
                                    artifact.get('team_routers'), artifact.get('temporal'),
                                    artifact.get('dead_router'), artifact.get('phase_router'))
        if checked['version'] != artifact['version']:
            raise ValueError('ensemble schema and routing/temporal metadata differ')
        all_weights = checked['weights']
        routers = checked.get('team_routers', [checked['team_router']] if 'team_router' in checked else [])
        if prepared is not None and (not isinstance(prepared,list) or len(prepared) != len(all_members)):
            raise ValueError('ensemble inference needs one prepared feature view per member')
        feature_views = {}
        def view_for(i):
            if i not in feature_views:
                feature_views[i] = (prepared[i] if prepared is not None else
                                    prepare_features(inputs,p,baseline,emit,shape,all_members[i]))
            return feature_views[i]
        active = np.flatnonzero(all_weights > 0)
        members = [all_members[i] for i in active]
        weights = all_weights[active]
        views = [view_for(i) for i in active]
        for router in routers:
            expert = all_members[router['expert']]
            expert_view = view_for(router['expert'])
            experts.append((router, expert, expert_view))
        temporal = checked.get('temporal')
        if temporal is not None:
            auxiliary_weights = temporal['consensus_weights']
            active_auxiliary = np.flatnonzero(auxiliary_weights > 0)
            auxiliary_members = [all_members[i] for i in active_auxiliary]
            auxiliary_weights = auxiliary_weights[active_auxiliary]
            auxiliary_views = [view_for(i) for i in active_auxiliary]
        dead_router = checked.get('dead_router')
        if dead_router is not None:
            dead_member = all_members[dead_router['expert']]
            dead_view = view_for(dead_router['expert'])
        phase_router = checked.get('phase_router')
        if phase_router is not None:
            phase_member = all_members[phase_router['expert']]
            phase_view = view_for(phase_router['expert'])
    else:
        raise ValueError('hybrid checkpoint schema differs')
    mask = np.logical_and.reduce([view[1] for view in views])
    rows = np.flatnonzero(mask)
    if not len(rows):
        return baseline.copy()
    matrices = [validator.matrix(view[0],member['names'],rows)
                for view,member in zip(views,members)]
    expert_data = []
    for router, expert, expert_view in experts:
        expert_mask = mask & expert_view[1]
        expert_rows = np.flatnonzero(expert_mask)
        expert_x = validator.matrix(expert_view[0], expert['names'], expert_rows)
        expert_data.append((router, expert_mask, expert_rows, expert_x))
    if temporal is not None:
        auxiliary_mask = np.logical_and.reduce([view[1] for view in auxiliary_views])
        auxiliary_rows = np.flatnonzero(auxiliary_mask)
        auxiliary_matrices = [validator.matrix(view[0], member['names'], auxiliary_rows)
                              for view, member in zip(auxiliary_views, auxiliary_members)]
    if dead_router is not None:
        dead_mask = mask & dead_view[1]
        dead_rows = np.flatnonzero(dead_mask)
        dead_x = validator.matrix(dead_view[0], dead_member['names'], dead_rows)
    if phase_router is not None:
        phase_mask = mask & phase_view[1]
        phase_rows = np.flatnonzero(phase_mask)
        phase_x = validator.matrix(phase_view[0], phase_member['names'], phase_rows)
    def predict(models, expert_models, auxiliary_models, dead_model, phase_model):
        q = np.full((inputs.n, 3), np.nan)
        if len(models) == 1:
            q[rows] = validator._proba3(models[0],matrices[0])
        else:
            q[rows] = sum(weight * validator._proba3(model,x)
                          for weight,model,x in zip(weights,models,matrices))
        for expert_model, (router, expert_mask, expert_rows, expert_x) in zip(expert_models, expert_data):
            if not len(expert_rows):
                continue
            other = np.full((inputs.n, 3), np.nan)
            other[expert_rows] = validator._proba3(expert_model, expert_x)
            q = route_teams(q, other, expert_mask, router['primary_max'], router['expert_min'])
        if temporal is not None:
            auxiliary_q = np.full((inputs.n, 3), np.nan)
            if len(auxiliary_rows):
                auxiliary_q[auxiliary_rows] = sum(weight * validator._proba3(model,x)
                    for weight,model,x in zip(auxiliary_weights,auxiliary_models,auxiliary_matrices))
            q = blend_dead_probability(q, auxiliary_q, mask & auxiliary_mask,
                                       temporal['consensus_dead_weight'])
        if dead_router is not None and len(dead_rows) and dead_router['weight'] > 0:
            other = np.full((inputs.n, 3), np.nan)
            other[dead_rows] = validator._proba3(dead_model, dead_x)
            q = blend_dead_probability(q, other, dead_mask, dead_router['weight'])
        def finish(probabilities, params):
            labels = decode(inputs,params,baseline,emit,bonus,probabilities,mask)
            if temporal is not None:
                labels = spell_consensus.remove_oscillations(
                    labels, probabilities, mask, inputs.blocks, temporal['oscillation_max_frames'])
                auxiliary_labels = decode(inputs,params,baseline,emit,bonus,auxiliary_q,auxiliary_mask)
                labels = spell_consensus.extend_agreed_spells(
                    labels, auxiliary_labels, mask & auxiliary_mask, inputs.blocks,
                    temporal['consensus_max_frames'], temporal['consensus_min_iou'])
            return labels
        labels = finish(q, p)
        if phase_router is not None and len(phase_rows) and phase_router['weight'] > 0:
            other = np.full((inputs.n, 3), np.nan)
            other[phase_rows] = validator._proba3(phase_model, phase_x)
            valid = (phase_mask & np.isfinite(other).all(1) & (other[:, 2] >= 0)
                     & (other[:, 2] <= 1) & np.isfinite(q).all(1) & (q[:, :2].sum(1) > 0))
            if valid.any():
                changed = blend_dead_probability(q, other, valid, phase_router['weight'])
                phase = finish(changed, replace(p, hybrid_dead_margin=phase_router['dead_margin']))
                labels = (spell_consensus.overlay_dead_mask(labels, phase, mask)
                          if phase_router['keep_teams'] else phase)
        return labels
    output = predict([member['full'] for member in members], [expert['full'] for _,expert,_ in experts],
                     [member['full'] for member in auxiliary_members],
                     dead_member['full'] if dead_member is not None else None,
                     phase_member['full'] if phase_member is not None else None)
    # Independent whole-half decoding per held-out model prevents decoding neighbours
    # with a model trained on the held-out labels and bleeding those labels back in.
    reference = members[0]
    public = np.arange(reference['public_start'], reference['public_stop'])
    ids = np.clip(np.searchsorted(reference['edges'], public, side='right') - 1,
                  0, reference['folds'] - 1)
    for f in range(reference['folds']):
        labels = predict([member['heads'][f] for member in members],
                         [expert['heads'][f] for _,expert,_ in experts],
                         [member['heads'][f] for member in auxiliary_members],
                         dead_member['heads'][f] if dead_member is not None else None,
                         phase_member['heads'][f] if phase_member is not None else None)
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
