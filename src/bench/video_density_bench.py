"""Paired sparse/dense video audit, keeping classifier training at 1 Hz.

All time partitions reuse the same public possessions. This measures sensitivity to
video sampling on those possessions, not performance on independent matches.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import hybrid, pipeline
from possession.config import Params
from possession.film_features import Store
from possession.paths import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--params', default=str(Path(ROOT, 'src/configs/pre_dense.json')))
    ap.add_argument('--kind', default='film_dense')
    ap.add_argument('--folds', nargs='+', type=int, default=[3, 4, 5, 6])
    ap.add_argument('--out', default=str(Path(ROOT, 'cache/improve_nov15/video_density_audit.json')))
    args = ap.parse_args()
    p = Params.load(args.params)
    inputs = pipeline.Inputs()
    base, dbg = pipeline.predict(inputs, replace(p, hybrid_model=''), debug=True)
    original = joblib.load(p.hybrid_model)
    kinds = (original['kind'], args.kind)
    prepared = {kind:hybrid.features(inputs, p, base, dbg['emit'], dbg['shape'],
                kind, original['legacy_params']) for kind in kinds}
    np.testing.assert_array_equal(prepared[kinds[0]][1], prepared[kinds[1]][1])
    gt = pipeline.public_gt()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for folds in args.folds:
        before = None
        for kind in kinds:
            columns, mask, frames, legacy = prepared[kind]
            artifact = (original if kind == kinds[0] and folds == original['folds']
                        else hybrid.fit(columns, frames, gt, original['names'],
                            original['model_name'], kind, legacy, n_folds=folds,
                            gap=original['purge_frames'], train_stride=original['train_stride'],
                            logreg_c=original.get('logreg_c', .3)))
            labels = hybrid.refine(inputs, p, base, dbg['emit'], dbg['bonus'],
                                   dbg['shape'], artifact, prepared[kind])
            if before is None:
                before = labels
            np.testing.assert_array_equal(labels == 'D', before == 'D')
            np.testing.assert_array_equal(labels[~mask], before[~mask])
            score = pipeline.score(labels, gt)
            row = dict(folds=folds, kind=kind, samples=len(frames),
                training_rows=artifact['training_rows'], spellF1=score['spell_f1_mean'],
                macroF1=score['macro_f1'], public_possessions=score['n_pred_spells'],
                transition_f1_1s=score['transitions'][25]['f1'],
                changed_frames=int((labels != before).sum()),
                public_changed_frames=int(((labels != before) & (gt < 3)).sum()))
            rows.append(row)
            print(json.dumps(row), flush=True)
            path.write_text(json.dumps(dict(selection_used_public=True,
                same_public_possessions=True, params=asdict(p),
                feature_cache_sha256={k:hashlib.sha256(Path(Store(k).path).read_bytes()).hexdigest()
                                      for k in kinds}, results=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
