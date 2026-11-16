"""Screen frozen DINOv2 crops against a fold-aligned video ensemble.

PCA, imputation and logistic fitting use training rows only. Each candidate
is decoded per held-out head before public stitching. This screen reuses
public labels for selection and is not independent match validation.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import hybrid, pipeline, spell_consensus, validator
from possession.config import Params
from possession.paths import ROOT


class Probe:
    def __init__(self, params, folds):
        self.p = Params.load(params)
        self.inputs = pipeline.Inputs()
        self.base, dbg = pipeline.predict(self.inputs, replace(self.p, hybrid_model=''), debug=True)
        self.emit, self.bonus, self.shape = dbg['emit'], dbg['bonus'], dbg['shape']
        self.gt = pipeline.public_gt()
        original = joblib.load(self.p.hybrid_model)
        if original['version'] != hybrid.DEAD_ROUTED_ENSEMBLE_VERSION:
            raise ValueError('this screen expects the macro-v6 ensemble')
        self.views = hybrid.prepare_features(self.inputs, self.p, self.base, self.emit, self.shape, original)
        models = []
        for member, (columns, _, frames, legacy) in zip(original['members'], self.views):
            models.append(member if folds == member['folds'] else hybrid.fit(
                columns, frames, self.gt, member['names'], member['model_name'], member['kind'],
                legacy, n_folds=folds, gap=member['purge_frames'], train_stride=member['train_stride'],
                logreg_c=member.get('logreg_c', .3)))
        self.a = hybrid.combine_artifacts(models, original['weights'],
            original.get('team_router'), original.get('team_routers'), original['temporal'], original['dead_router'])
        active = [v[1] for v, w in zip(self.views, self.a['weights']) if w > 0]
        self.mask = np.logical_and.reduce(active)
        if not all(np.array_equal(v[1], self.mask) for v in self.views):
            raise ValueError('this prototype requires aligned video coverage')
        self.allrows = np.flatnonzero(self.mask)
        self.public = np.arange(self.a['public_start'], self.a['public_stop'])
        self.held = np.clip(np.searchsorted(self.a['edges'], self.public, side='right') - 1, 0, folds-1)
        grid = np.r_[np.arange(self.public[0], self.public[-1]+1, self.a['train_stride']), self.public[-1]]
        self.rows = np.intersect1d(self.views[0][2], grid)
        self.trainfold = np.clip(np.searchsorted(self.a['edges'], self.rows, side='right')-1, 0, folds-1)
        self.target = self.gt[self.rows]
        self.members = []
        for member, view in zip(models, self.views):
            matrix = validator.matrix(view[0], member['names'], self.allrows)
            self.members.append(self.probabilities([member['full']] + member['heads'], matrix))
        self.primary = self.route(sum(w*q for w, q in zip(self.a['weights'], self.members) if w > 0))
        self.auxiliary = sum(w*q for w, q in zip(self.a['temporal']['consensus_weights'], self.members) if w > 0)
        self.dead = self.members[self.a['dead_router']['expert']]
        reference = hybrid.refine(self.inputs, self.p, self.base, self.emit, self.bonus, self.shape,
                                  self.a, self.views)
        np.testing.assert_array_equal(self.render(self.primary, self.auxiliary), reference)
        self.reference = reference

    def probabilities(self, models, matrix):
        result = []
        for model in models:
            q = np.full((self.inputs.n, 3), np.nan)
            q[self.allrows] = validator._proba3(model, matrix)
            result.append(q)
        return np.array(result)

    def route(self, pack):
        out = pack.copy()
        for router in self.a.get('team_routers', []):
            out = np.array([hybrid.route_teams(q, e, self.mask, router['primary_max'], router['expert_min'])
                            for q, e in zip(out, self.members[router['expert']])])
        return out

    def render(self, primary, auxiliary):
        cfg = self.a['temporal']
        output = None
        for head, (q, aux, dead) in enumerate(zip(primary, auxiliary, self.dead)):
            q = hybrid.blend_dead_probability(q, aux, self.mask, cfg['consensus_dead_weight'])
            q = hybrid.blend_dead_probability(q, dead, self.mask, self.a['dead_router']['weight'])
            labels = hybrid.decode(self.inputs, self.p, self.base, self.emit, self.bonus, q, self.mask)
            labels = spell_consensus.remove_oscillations(labels, q, self.mask, self.inputs.blocks,
                                                        cfg['oscillation_max_frames'])
            support = hybrid.decode(self.inputs, self.p, self.base, self.emit, self.bonus, aux, self.mask)
            labels = spell_consensus.extend_agreed_spells(labels, support, self.mask, self.inputs.blocks,
                                                         cfg['consensus_max_frames'], cfg['consensus_min_iou'])
            if head == 0:
                output = labels
            else:
                held = self.public[self.held == head-1]
                output[held] = labels[held]
        return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--params', default=str(Path(ROOT, 'src/configs/macro_v6.json')))
    ap.add_argument('--cache', type=Path, default=Path(ROOT, 'cache/improve_nov17/dino_5hz.npz'))
    ap.add_argument('--out', type=Path, default=Path(ROOT, 'cache/improve_nov17/dino_bench'))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--components', type=int, nargs='+', default=[16, 32])
    ap.add_argument('--C', type=float, default=.3)
    ap.add_argument('--pool', choices=['cls', 'ball', 'above', 'pools'], default='cls')
    ap.add_argument('--crops', choices=['128', '320', 'both'], nargs='+', default=['128', '320', 'both'])
    args = ap.parse_args()
    native = np.load(args.cache)
    metadata = json.loads(args.cache.with_suffix('.json').read_text())
    frames, embeddings = native['frames'], native['embeddings']
    if args.pool in ('ball', 'above'):
        embeddings = native['patch_pools'][:, :, 0 if args.pool == 'ball' else 1]
    elif args.pool == 'pools':
        embeddings = native['patch_pools'].reshape(len(frames), 2, -1)
    if hashlib.sha256(Path(ROOT, 'src/cv/frames/valdi_film_5hz.txt').read_bytes()).hexdigest() != metadata['frame_plan_sha256']:
        raise ValueError('feature cache uses a different frame plan')
    np.testing.assert_array_equal(frames, np.loadtxt(Path(ROOT, 'src/cv/frames/valdi_film_5hz.txt'), dtype=np.int64))
    probe = Probe(args.params, args.folds)
    for key, value in [('raw_ball_sha256', probe.inputs.atd['ball']),
                       ('crop_ball_sha256', probe.inputs.ball_atd(probe.p)['ball'])]:
        if hashlib.sha256(value.tobytes()).hexdigest() != metadata[key]:
            raise ValueError('feature crop provenance differs: ' + key)
    intervals = hybrid.covered_intervals(frames, probe.inputs.graded)
    coverage = np.zeros(probe.inputs.n, bool)
    for a, b in intervals:
        coverage[a:b+1] = True
    if (probe.mask & ~coverage).any():
        raise ValueError('DINO frame plan is missing primary video coverage')
    crops = []
    for crop_index in range(2):
        dimension = embeddings.shape[-1]
        values = np.full((len(probe.allrows), dimension), np.nan, np.float32)
        for a, b in intervals:
            take = (probe.allrows >= a) & (probe.allrows <= b)
            source = (frames >= a) & (frames <= b)
            for column in range(dimension):
                values[take, column] = np.interp(probe.allrows[take], frames[source],
                                                embeddings[source, crop_index, column])
        if not np.isfinite(values).all():
            raise ValueError('invalid interpolated DINO features')
        crops.append(values)
    names = probe.a['members'][0]['names']
    context = validator.matrix(probe.views[0][0], names, probe.allrows)
    train_indices = np.searchsorted(probe.allrows, probe.rows)
    np.testing.assert_array_equal(probe.allrows[train_indices], probe.rows)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for crop in args.crops:
        visual = crops[0] if crop == '128' else crops[1] if crop == '320' else np.column_stack(crops)
        allx = np.column_stack([context, visual])
        x = allx[train_indices]
        for components in args.components:
            models = []
            for head in range(-1, args.folds):
                train = (np.ones(len(probe.rows), bool) if head < 0 else
                         validator.purged_train(probe.rows, probe.trainfold, head, probe.a['purge_frames']))
                visual_steps = (PCA(n_components=components, whiten=True, svd_solver='full', random_state=0)
                                if components else StandardScaler())
                preprocessing = ColumnTransformer([
                    ('context', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), list(range(len(names)))),
                    ('visual', make_pipeline(SimpleImputer(strategy='median'), visual_steps),
                     list(range(len(names), allx.shape[1])))])
                model = make_pipeline(preprocessing, LogisticRegression(C=args.C, max_iter=3000))
                model.fit(x[train], probe.target[train])
                models.append(model)
            key = f'f{args.folds}_{args.pool}_crop{crop}_pc{components}_C{args.C}'
            joblib.dump(dict(models=models, crop=crop, pca_components=components, C=args.C,
                             extraction=metadata, pool=args.pool, context_names=names, folds=args.folds,
                             edges=probe.a['edges'], purge_frames=probe.a['purge_frames'],
                             model_name='experimental_dino_logreg'), args.out / f'{key}.joblib')
            pack = probe.probabilities(models, allx)
            for mode in ['expert', 'replace_dense', 'auxiliary']:
                if mode == 'expert':
                    primary = np.array([hybrid.route_teams(q, e, probe.mask) for q, e in zip(probe.primary, pack)])
                elif mode == 'replace_dense':
                    primary = probe.route(.75*pack + .25*probe.members[1])
                else:
                    primary = probe.primary
                auxiliary = .5*probe.members[0]+.5*pack if mode == 'auxiliary' else probe.auxiliary
                labels = probe.render(primary, auxiliary)
                score = pipeline.score(labels, probe.gt)
                result = dict(folds=args.folds, pool=args.pool, crop=crop, pca_components=components, C=args.C, mode=mode,
                    macroF1=score['macro_f1'], spellF1=score['spell_f1_mean'], possessions=score['n_pred_spells'],
                    per_class=score['per_class'], changed_frames=int((labels != probe.reference).sum()))
                results.append(result)
                print(json.dumps(result), flush=True)
                np.save(args.out / f'{key}_{mode}.npy', labels)
                (args.out / 'results.json').write_text(json.dumps(dict(selected_on_public=True,
                    same_public_possessions=True, private_score_known=False,
                    baseline=pipeline.score(probe.reference, probe.gt),
                    extraction_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),
                    results=results), indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')


if __name__ == '__main__':
    main()
