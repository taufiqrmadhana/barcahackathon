"""Check phase blend weights, regularisation and context subsets against v11.

The other five members and all decoding rules are frozen. Selection still
reuses the same public possessions; these are sensitivity checks only.
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broadcast_followup_bench import Probe
from possession import broadcast, hybrid, pipeline, validator
from possession.dead_head import DeadBallClassifier
from possession.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--out', type=Path, default=Path(ROOT, 'cache/phase_blend_bench'))
    args = parser.parse_args()
    probe = Probe(Path(ROOT, 'src/configs/macro_v10.json'), args.folds)
    index = probe.a['phase_router']['expert']
    columns, coverage, frames, _ = probe.views[index]
    names = hybrid.FEATURE_SETS['broadcast_dead_context']
    columns = dict(columns)
    columns.update(broadcast.availability_context(columns['broadcast_available'].astype(bool), probe.inputs.blocks))
    native = np.intersect1d(frames, np.r_[np.arange(probe.public[0], probe.public[-1]+1, 25), probe.public[-1]])
    native = native[coverage[native]]
    args.out.mkdir(parents=True, exist_ok=True)

    def fit(selected, c):
        matrix = validator.matrix(columns, selected, native)
        models = []
        for head in range(-1, args.folds):
            train = (np.ones(len(native), bool) if head < 0 else
                ((native < probe.a['edges'][head]-125) | (native >= probe.a['edges'][head+1]+125)))
            models.append(DeadBallClassifier(C=c).fit(matrix[train], probe.gt[native[train]]))
        return models, probe.probabilities(models, columns, selected, coverage)

    _, context = fit(names, .3)
    reference = probe.render(.5*probe.phase+.5*context, coverage)
    frozen = json.loads(Path(ROOT, 'src/bench/results/macro_v11_2026-11-17.json').read_text())
    expected = next(r for r in frozen['sensitivity']['results'] if r['folds']==args.folds and r['objective']=='candidate')
    baseline = pipeline.score(reference, probe.gt)
    assert baseline['macro_f1']==expected['macroF1'] and baseline['spell_f1_mean']==expected['spellF1']
    cases = [('context_weight25', .25, None, .3), ('context_weight75', .75, None, .3),
        ('context_C0.1_average50', .5, names, .1),
        ('fraction_only_average50', .5, names[:18]+names[20:], .3),
        ('time_only_average50', .5, names[:20], .3)]
    results = []
    for name, weight, selected, c in cases:
        candidate = context
        if selected:
            models, candidate = fit(selected, c)
            joblib.dump(dict(models=models, names=selected, C=c, folds=args.folds), args.out / (name+'.joblib'))
        labels = probe.render((1-weight)*probe.phase+weight*candidate, coverage)
        score = pipeline.score(labels, probe.gt)
        row = dict(folds=args.folds, case=name, macroF1=score['macro_f1'], spellF1=score['spell_f1_mean'],
            accuracy=score['accuracy'], delta_macro_vs_v11=score['macro_f1']-baseline['macro_f1'],
            public_changes=int((reference[probe.public]!=labels[probe.public]).sum()))
        results.append(row)
        np.save(args.out / (name+'.npy'), labels)
        print(json.dumps(row), flush=True)
        (args.out / 'results.json').write_text(json.dumps(dict(results=results,
            same_public_possessions=True, private_score_known=False,
            source_sha256=broadcast.digest(__file__)), indent=2)+'\n')


if __name__ == '__main__':
    main()
