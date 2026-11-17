"""Screen additional broadcast evidence against a frozen phase ensemble.

Every candidate uses matching purged time heads and whole-half decoding before
public stitching. All scores reuse the public labels for method selection;
they are not independent test results. Per-frame outputs stay in the cache.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import broadcast, clean, hybrid, pipeline, spell_consensus, validator
from possession.config import Params
from possession.paths import ROOT


def temporal_features(columns, quality, blocks):
    """Control history and +/-1 s context, with no crossing of half boundaries."""
    result = dict(columns)
    n = len(quality)
    for team in ('home', 'away'):
        control = (quality & (columns['broadcast_dist_'+team] < 1.5)
                   & (columns['broadcast_relative_'+team] < 1.5))
        for a, b in zip(*clean.runs(control)):
            if b-a < 8:
                control[a:b] = False
        before, after = np.full(n, 125., np.float32), np.full(n, 125., np.float32)
        for _, a, b in blocks:
            before[a:b+1] = np.minimum(clean.since_last(control[a:b+1]), 125.)/25
            after[a:b+1] = np.minimum(clean.until_next(control[a:b+1]), 125.)/25
        result['broadcast_since_'+team+'_control'] = before
        result['broadcast_until_'+team+'_control'] = after
    for key in ('team_distance', 'contact_team', 'dist_home', 'dist_away',
                'relative_home', 'relative_away'):
        value = columns['broadcast_'+key]
        for lag in (-25, 25):
            shifted = np.full(n, np.nan)
            for _, a, b in blocks:
                low, high = max(a, a-lag), min(b+1, b+1-lag)
                shifted[low:high] = value[low+lag:high+lag]
            result[f'broadcast_{key}_lag{lag}'] = shifted
    return result


MOTION_KEYS = ('naive_res_mean', 'naive_res_p99', 'naive_res_ball',
               'naive_res_home', 'naive_res_away', 'naive_near_home',
               'naive_near_away', 'film_cnt_home', 'film_cnt_away',
               'flow_cam', 'flow_act', 'diff_global')


def motion_features(path, inputs, quality, provenance):
    metadata = json.loads(path.with_suffix('.json').read_text())
    if broadcast.digest(path) != metadata.get('cache_sha256'):
        raise ValueError('motion extraction is incomplete or modified')
    if metadata['inputs']['broadcast'] != provenance:
        raise ValueError('motion extraction uses different geometry')
    plan = Path(ROOT, 'src/cv/frames/valdi_film_dense.txt')
    if broadcast.digest(plan) != metadata['inputs']['plan_sha256']:
        raise ValueError('motion extraction uses a different sampling plan')
    native = np.load(path)
    frames = native['frames']
    intervals = hybrid.covered_intervals(frames, inputs.graded & quality)
    coverage = np.zeros(inputs.n, bool)
    result = {}
    keys = MOTION_KEYS
    if metadata['inputs'].get('interpolator') == 'film':
        from possession.film_features import FEATURES
        keys = FEATURES
    for key in keys:
        value = np.full(inputs.n, np.nan)
        for a, b in intervals:
            rows = np.arange(a, b+1)
            take = (frames >= a) & (frames <= b)
            value[rows] = np.interp(rows, frames[take], native[key][take])
            coverage[rows] = True
        result['broadcast_motion_'+key] = value
    return result, coverage, frames, metadata


class Probe:
    def __init__(self, params, folds):
        self.p = Params.load(params)
        self.inputs = pipeline.Inputs()
        self.base, debug = pipeline.predict(self.inputs, replace(self.p, hybrid_model=''), debug=True)
        self.emit, self.bonus, self.shape = (debug[k] for k in ('emit', 'bonus', 'shape'))
        self.gt = pipeline.public_gt()
        original = joblib.load(self.p.hybrid_model)
        if original['version'] != hybrid.PHASE_ENSEMBLE_VERSION:
            raise ValueError('this screen requires a phase ensemble')
        self.views = hybrid.prepare_features(self.inputs, self.p, self.base, self.emit,
                                              self.shape, original)
        models = []
        for member, (columns, _, frames, legacy) in zip(original['members'], self.views):
            models.append(member if folds == member['folds'] else hybrid.fit(
                columns, frames, self.gt, member['names'], member['model_name'],
                member['kind'], legacy, n_folds=folds, gap=member['purge_frames'],
                train_stride=member['train_stride'], logreg_c=member.get('logreg_c', .3),
                strict_fold_edges=member.get('strict_fold_edges', False)))
        self.a = hybrid.combine_artifacts(models, original['weights'],
            original.get('team_router'), original.get('team_routers'),
            original['temporal'], original['dead_router'], original['phase_router'])
        self.mask = np.logical_and.reduce([v[1] for v, w in zip(self.views, self.a['weights']) if w > 0])
        cfg = self.a['temporal']
        auxiliary_mask = np.logical_and.reduce([v[1] for v, w in zip(self.views, cfg['consensus_weights']) if w > 0])
        if not np.array_equal(auxiliary_mask, self.mask):
            raise ValueError('this screen requires primary/auxiliary coverage to match')
        self.public = np.arange(self.a['public_start'], self.a['public_stop'])
        self.held = np.clip(np.searchsorted(self.a['edges'], self.public, side='right')-1, 0, folds-1)
        packs = [self.probabilities([m['full']]+m['heads'], v[0], m['names'], v[1])
                 for m, v in zip(models, self.views)]
        primary = sum(w*q for w, q in zip(self.a['weights'], packs) if w > 0)
        for router in self.a.get('team_routers', []):
            i = router['expert']
            primary = np.array([hybrid.route_teams(q, e, self.mask & self.views[i][1],
                                router['primary_max'], router['expert_min'])
                                for q, e in zip(primary, packs[i])])
        self.auxiliary = sum(w*q for w, q in zip(cfg['consensus_weights'], packs) if w > 0)
        primary = np.array([hybrid.blend_dead_probability(q, aux, self.mask,
                            cfg['consensus_dead_weight']) for q, aux in zip(primary, self.auxiliary)])
        dead = self.a['dead_router']
        self.primary = np.array([hybrid.blend_dead_probability(q, e,
            self.mask & self.views[dead['expert']][1], dead['weight'])
            for q, e in zip(primary, packs[dead['expert']])])
        phase = self.a['phase_router']
        if not phase['keep_teams']:
            raise ValueError('this screen expects pointwise phase fusion')
        self.phase, self.phase_mask = packs[phase['expert']], self.mask & self.views[phase['expert']][1]
        self.base_heads = [self.finish(q, aux, self.p) for q, aux in zip(self.primary, self.auxiliary)]
        self.reference = hybrid.refine(self.inputs, self.p, self.base, self.emit, self.bonus,
                                       self.shape, self.a, self.views)
        np.testing.assert_array_equal(self.render(), self.reference)

    def probabilities(self, models, columns, names, coverage):
        rows = np.flatnonzero(coverage)
        matrix = validator.matrix(columns, names, rows)
        result = []
        for model in models:
            q = np.full((self.inputs.n, 3), np.nan)
            q[rows] = validator._proba3(model, matrix)
            result.append(q)
        return np.array(result)

    def finish(self, q, aux, params):
        cfg = self.a['temporal']
        y = hybrid.decode(self.inputs, params, self.base, self.emit, self.bonus, q, self.mask)
        y = spell_consensus.remove_oscillations(y, q, self.mask, self.inputs.blocks,
                                                cfg['oscillation_max_frames'])
        other = hybrid.decode(self.inputs, params, self.base, self.emit, self.bonus, aux, self.mask)
        return spell_consensus.extend_agreed_spells(y, other, self.mask, self.inputs.blocks,
                                                    cfg['consensus_max_frames'], cfg['consensus_min_iou'])

    def render(self, pack=None, coverage=None, mode='phase', contact_vote=None):
        cfg = self.a['phase_router']
        heads = []
        for i, (q, aux, phase) in enumerate(zip(self.primary, self.auxiliary, self.phase)):
            base = self.base_heads[i]
            phase_mask = self.phase_mask
            if pack is not None and mode == 'phase':
                phase, phase_mask = pack[i], self.mask & coverage
            elif pack is not None:
                e = pack[i]
                if mode == 'uncertain':
                    q = hybrid.route_teams(q, e, self.mask & coverage)
                else:
                    alive, ea = q[:, :2].sum(1), e[:, :2].sum(1)
                    prior = np.divide(q[:, :2], alive[:, None], out=np.full_like(q[:, :2], .5), where=alive[:, None] > 0)
                    ratio = np.divide(e[:, :2], ea[:, None], out=np.full_like(e[:, :2], .5), where=ea[:, None] > 0)
                    gate = self.mask & coverage & np.isfinite(e).all(1) & (alive > 0) & (ea > 0)
                    q = q.copy()
                    if mode == 'mix25':
                        ratio = .75*prior + .25*ratio
                    elif mode == 'contact':
                        gate &= ((ratio.max(1) > .9) & (np.abs(contact_vote) > .5)
                                 & (ratio.argmax(1) == np.where(contact_vote > 0, 0, 1)))
                    else:
                        raise ValueError('unknown team routing mode')
                    q[gate, :2] = (1-q[gate, 2])[:, None]*ratio[gate]
                base = self.finish(q, aux, self.p)
            changed = hybrid.blend_dead_probability(q, phase, phase_mask, cfg['weight'])
            refined = self.finish(changed, aux, replace(self.p, hybrid_dead_margin=cfg['dead_margin']))
            heads.append(spell_consensus.overlay_dead_mask(base, refined, self.mask))
        output = heads[0].copy()
        for i, y in enumerate(heads[1:]):
            held = self.public[self.held == i]
            output[held] = y[held]
        return output

    def fit(self, columns, names, native, coverage, family, alive_only=False, binary=False):
        native = native[coverage[native]]
        if alive_only:
            native = native[self.gt[native] < 2]
        matrix = validator.matrix(columns, names, native)
        target = self.gt[native]
        if binary and family != 'dead_logreg':
            target = np.where(target == 2, 2, 0)
        models = []
        for head in range(-1, self.a['folds']):
            train = (np.ones(len(native), bool) if head < 0 else
                     ((native < self.a['edges'][head]-self.a['purge_frames'])
                      | (native >= self.a['edges'][head+1]+self.a['purge_frames'])))
            model = (hybrid.factory(family, names=names, logreg_c=.3) if np.unique(target[train]).size > 1
                     else DummyClassifier(strategy='prior'))
            model.fit(matrix[train], target[train])
            models.append(model)
        return models, self.probabilities(models, columns, names, coverage), len(native)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', type=Path, default=Path(ROOT, 'src/configs/macro_v9.json'))
    parser.add_argument('--suite', choices=('temporal_team', 'temporal_phase', 'motion', 'missing_phase'), required=True)
    parser.add_argument('--motion-cache', type=Path, default=Path(ROOT, 'cache/broadcast_motion/features.npz'))
    parser.add_argument('--out', type=Path, default=Path(ROOT, 'cache/broadcast_followups'))
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    probe = Probe(args.params, args.folds)
    bb, quality, provenance = broadcast.features(probe.inputs, probe.p)
    coverage = quality & probe.mask
    view = {'temporal_phase': 2, 'temporal_team': 1, 'motion': 0, 'missing_phase': 0}[args.suite]
    columns = dict(probe.views[view][0])
    columns.update(bb)
    extra = temporal_features(bb, coverage, probe.inputs.blocks)
    if args.suite == 'missing_phase':
        extra = dict(bb, broadcast_available=quality.astype(float))
        coverage, native, extraction = probe.mask.copy(), probe.views[0][2], None
        groups = {'geometry': tuple(bb), 'geometry_available': tuple(extra)}
    elif args.suite == 'motion':
        extra, coverage, native, extraction = motion_features(args.motion_cache, probe.inputs, quality, provenance)
        coverage &= probe.mask
        groups = {'motion': tuple(extra), 'geometry_motion': tuple(bb)+tuple(extra),
                  'soft_geometry_motion': hybrid.FEATURE_SETS['film_soft']+tuple(bb)+tuple(extra)}
    else:
        extraction = None
        native = (np.loadtxt(Path(ROOT, 'src/cv/frames/valdi_film_5hz.txt'), dtype=np.int64)
                  if args.suite == 'temporal_team' else probe.views[0][2])
        groups = {'basic': tuple(bb), 'temporal': tuple(extra)}
        groups['soft_temporal' if args.suite == 'temporal_team' else 'motion_temporal'] = (
            hybrid.FEATURE_SETS['film_soft' if args.suite == 'temporal_team' else 'film_dead_motion']+tuple(extra))
    columns.update(extra)
    stride = 5 if args.suite == 'temporal_team' else 25
    native = np.intersect1d(native, np.r_[np.arange(probe.public[0], probe.public[-1]+1, stride), probe.public[-1]])
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / 'baseline.npy', probe.reference)
    results = []
    report = dict(suite=args.suite, folds=args.folds, training_stride=stride,
        selected_on_public=True, same_public_possessions=True, private_score_known=False,
        strict_fold_edges=True, purge_frames=probe.a['purge_frames'],
        baseline=pipeline.score(probe.reference, probe.gt), broadcast_provenance=provenance,
        extraction=extraction, params_sha256=broadcast.digest(args.params),
        model_sha256=broadcast.digest(probe.p.hybrid_model), source_sha256=broadcast.digest(__file__),
        support=int(coverage.sum()), public_support=int(coverage[probe.public].sum()), results=results)
    for group, names in groups.items():
        families = ('dead_logreg',) if args.suite == 'missing_phase' else (
            ('logreg', 'hist_gbm', 'dead_logreg') if args.suite == 'motion' else (
            ('logreg', 'hist_gbm') if args.suite == 'temporal_team' else ('dead_logreg', 'hist_gbm')))
        for family in families:
            binary = args.suite in ('temporal_phase', 'missing_phase') or family == 'dead_logreg'
            models, pack, count = probe.fit(columns, names, native, coverage, family,
                alive_only=args.suite == 'temporal_team', binary=binary)
            key = f'{args.suite}_{group}_{family}'
            joblib.dump(dict(models=models, names=names, model=family, C=.3,
                training_rows=count, stride=stride, strict_fold_edges=True,
                edges=probe.a['edges'], purge_frames=probe.a['purge_frames'],
                baseline_model_sha256=report['model_sha256'], broadcast_provenance=provenance,
                extraction=extraction, experimental=True), args.out / (key+'.joblib'))
            modes = ('phase',) if binary else (('uncertain', 'mix25', 'contact')
                if args.suite == 'temporal_team' else (('uncertain', 'mix25')
                if family == 'logreg' else ('uncertain',)))
            for mode in modes:
                y = probe.render(pack, coverage, mode, bb['broadcast_contact_team'])
                score = pipeline.score(y, probe.gt)
                row = dict(group=group, model=family, mode=mode, features=len(names), training_rows=count,
                    macroF1=score['macro_f1'], spellF1=score['spell_f1_mean'], possessions=score['n_pred_spells'],
                    per_class=score['per_class'], changed_frames=int((y != probe.reference).sum()),
                    public_changes=int((y[probe.public] != probe.reference[probe.public]).sum()))
                results.append(row)
                print(json.dumps(row), flush=True)
                np.save(args.out / f'{key}_{mode}.npy', y)
                (args.out / 'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
