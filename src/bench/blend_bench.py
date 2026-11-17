"""Compare original and blended video heads on the same purged time partitions.

Alternate partitions reuse the same public possessions; they are sensitivity checks,
not independent validation matches. Each head decodes its own whole-half context.
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
from possession.paths import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--params', default=str(Path(ROOT, 'src/configs/pre_blend.json')))
    ap.add_argument('--folds', nargs='+', type=int, default=[3, 4, 5, 6])
    ap.add_argument('--out', default=str(Path(ROOT, 'cache/improve_nov15/blend_audit.json')))
    args = ap.parse_args()
    p = Params.load(args.params)
    inputs = pipeline.Inputs()
    base, dbg = pipeline.predict(inputs, replace(p, hybrid_model=''), debug=True)
    original = joblib.load(p.hybrid_model)
    prepared = hybrid.features(inputs, p, base, dbg['emit'], dbg['shape'],
                               original['kind'], original['legacy_params'])
    columns, mask, frames, legacy = prepared
    gt = pipeline.public_gt()
    rows = []
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    for folds in args.folds:
        before = None
        for name in ('logreg', 'logreg_blend'):
            artifact = (original if folds == original['folds'] and name == 'logreg'
                        else hybrid.fit(columns, frames, gt, original['names'], name,
                            original['kind'], legacy, n_folds=folds,
                            gap=original['purge_frames'], train_stride=original['train_stride'],
                            logreg_c=original.get('logreg_c', .3)))
            labels = hybrid.refine(inputs, p, base, dbg['emit'], dbg['bonus'],
                                   dbg['shape'], artifact, prepared)
            if before is None:
                before = labels
            np.testing.assert_array_equal(labels == 'D', before == 'D')
            np.testing.assert_array_equal(labels[~mask], before[~mask])
            score = pipeline.score(labels, gt)
            row = dict(folds=folds, model=name, spellF1=score['spell_f1_mean'],
                       macroF1=score['macro_f1'], public_possessions=score['n_pred_spells'],
                       per_class=score['per_class'],
                       transition_f1_1s=score['transitions'][25]['f1'],
                       changed_frames=int((labels != before).sum()),
                       public_changed_frames=int(((labels != before) & (gt < 3)).sum()))
            rows.append(row)
            print(json.dumps(row), flush=True)
            path.write_text(json.dumps(dict(selection_used_public=True,
                same_public_possessions=True, params=asdict(p),
                classifier_sha256=hashlib.sha256(Path(p.hybrid_model).read_bytes()).hexdigest(),
                excluded_columns=hybrid.HARD_CONTEXT, weights=[1, 3],
                results=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
