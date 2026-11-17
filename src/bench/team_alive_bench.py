"""Screen an additional H/A specialist trained only on alive public rows.

All training and preprocessing exclude the target fold and a 125-frame purge.
The existing confidence gate preserves P(D), and complete halves are decoded
before public stitching. These experiments reuse the public labels for selection.
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broadcast_followup_bench import Probe
from possession import broadcast, pipeline, validator
from possession.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', type=Path, default=Path(ROOT, 'src/configs/macro_v11.json'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--out', type=Path, default=Path(ROOT, 'cache/team_alive_bench'))
    args = parser.parse_args()
    probe = Probe(args.params, args.folds)
    args.out.mkdir(parents=True, exist_ok=True)
    reference = pipeline.score(probe.reference, probe.gt)
    report = dict(folds=args.folds, same_public_possessions=True, private_score_known=False,
        baseline_macroF1=reference['macro_f1'], baseline_spellF1=reference['spell_f1_mean'],
        source_sha256=broadcast.digest(__file__), model_sha256=broadcast.digest(probe.p.hybrid_model),
        purge_frames=125, strict_fold_edges=True, primary_max=.8, expert_min=.9, results=[])
    specs = [(f'{label}_C{c:g}', index, c, False) for label,index in [('raw',0),('motion',2),('fused',3)]
             for c in [.3,1.,3.]] + [('motion_C1_balanced',2,1.,True)]
    for name,index,c,balanced in specs:
        columns,coverage,frames,_ = probe.views[index]
        names = probe.a['members'][index]['names']
        native = np.intersect1d(frames,np.r_[np.arange(probe.public[0],probe.public[-1]+1,25),probe.public[-1]])
        native = native[coverage[native] & (probe.gt[native]<2)]
        x = validator.matrix(columns,names,native)
        models = []
        for head in range(-1,args.folds):
            train = (np.ones(len(native),bool) if head<0 else
                ((native<probe.a['edges'][head]-125)|(native>=probe.a['edges'][head+1]+125)))
            y = probe.gt[native[train]]
            model = validator.model_zoo(0)['logreg']().set_params(logisticregression__C=c,
                logisticregression__class_weight='balanced' if balanced else None)
            if len(np.unique(y))<2:model=DummyClassifier(strategy='prior')
            models.append(model.fit(x[train],y))
        pack = probe.probabilities(models,columns,names,coverage)
        labels = probe.render(pack,coverage,mode='uncertain')
        score = pipeline.score(labels,probe.gt)
        truth = np.array(['H','A','D','X'])[probe.gt]
        public = probe.gt<3
        before,after=probe.reference==truth,labels==truth
        row = dict(case=name,features=len(names),training_rows=len(native),C=c,balanced=balanced,
            macroF1=score['macro_f1'],spellF1=score['spell_f1_mean'],accuracy=score['accuracy'],
            possessions=score['n_pred_spells'],changed_frames=int((labels!=probe.reference).sum()),
            public_changes=int((public&(labels!=probe.reference)).sum()),
            public_errors_fixed=int((public&~before&after).sum()),
            public_correct_regressed=int((public&before&~after).sum()))
        report['results'].append(row)
        np.save(args.out/(name+'.npy'),labels)
        joblib.dump(dict(models=models,names=names,C=c,balanced=balanced,folds=args.folds),args.out/(name+'.joblib'))
        print(json.dumps(row),flush=True)
        (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
