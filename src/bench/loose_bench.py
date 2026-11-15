"""Paired audit of video-gated switches and bounded D edges on purged time folds.

Alternative partitions reuse the same 17 public possessions. No independent-match
or private-score claim follows from this sensitivity check.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
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
    ap.add_argument('--params', default=str(Path(ROOT, 'src/configs/pre_loose.json')))
    ap.add_argument('--folds', nargs='+', type=int, default=[3, 4, 5, 6])
    ap.add_argument('--thresholds', nargs='+', type=float, default=[0., .8])
    ap.add_argument('--dead-margins', nargs='+', type=int, default=[0, 12])
    ap.add_argument('--out', default=str(Path(ROOT, 'cache/improve_nov15_round3/loose_audit.json')))
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
        artifact = (original if folds == original['folds'] else hybrid.fit(columns, frames,
            gt, original['names'], original['model_name'], original['kind'], legacy,
            n_folds=folds, gap=original['purge_frames'], train_stride=original['train_stride'],
            logreg_c=original.get('logreg_c', .3)))
        before = None
        for threshold, margin in itertools.product(sorted(set([0.] + args.thresholds)),
                                                  sorted(set([0] + args.dead_margins))):
            pp = replace(p, hybrid_loose_threshold=threshold, hybrid_dead_margin=margin)
            labels = hybrid.refine(inputs, pp, base, dbg['emit'], dbg['bonus'],
                                   dbg['shape'], artifact, prepared)
            if before is None:
                before = labels
            changed_dead = (labels == 'D') != (before == 'D')
            editable = hybrid.dead_flexibility(base, mask, inputs.blocks, margin, p.dead_min_frames)
            assert not (changed_dead & ~editable).any()
            np.testing.assert_array_equal(labels[~mask], before[~mask])
            score = pipeline.score(labels, gt)
            row = dict(folds=folds, threshold=threshold, dead_margin=margin,
                spellF1=score['spell_f1_mean'], macroF1=score['macro_f1'],
                public_possessions=score['n_pred_spells'],
                changed_frames=int((labels != before).sum()),
                public_changed_frames=int(((labels != before) & (gt < 3)).sum()),
                changed_D_frames=int(changed_dead.sum()))
            rows.append(row)
            print(json.dumps(row), flush=True)
            path.write_text(json.dumps(dict(selection_used_public=True,
                same_public_possessions=True, params=asdict(p),
                classifier_sha256=hashlib.sha256(Path(p.hybrid_model).read_bytes()).hexdigest(),
                results=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
