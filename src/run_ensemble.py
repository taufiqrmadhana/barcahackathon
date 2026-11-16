"""Combine existing video checkpoints with identical purged time folds.

Probabilities are combined before each independent whole-half decode. This writes
held-out public predictions and an all-public refit only outside that window.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import sys

import joblib

sys.path.insert(0,str(Path(__file__).resolve().parent))
from possession import hybrid,pipeline
from possession.config import Params
from possession.paths import ROOT
from run import parse_overrides,report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models',nargs='+',required=True)
    ap.add_argument('--weights',nargs='+',type=float,required=True)
    ap.add_argument('--team-expert',type=int,action='append',
                    help='zero-weight H/A expert index; repeat in priority order')
    ap.add_argument('--primary-max',type=float,default=.8)
    ap.add_argument('--expert-min',type=float,default=.9)
    ap.add_argument('--dead-expert',type=int,help='zero-weight P(D) expert index')
    ap.add_argument('--dead-expert-weight',type=float,default=.5)
    ap.add_argument('--consensus-weights',nargs='+',type=float,
                    help='auxiliary mixture for agreeing short spells, using the same held-out heads')
    ap.add_argument('--consensus-max-frames',type=int,default=100)
    ap.add_argument('--consensus-min-iou',type=float,default=.5)
    ap.add_argument('--consensus-dead-weight',type=float,default=0.,
                    help='blend auxiliary P(D) without changing primary conditional H/A odds')
    ap.add_argument('--oscillation-max-frames',type=int,default=0)
    ap.add_argument('--params',default=str(Path(ROOT,'src/params.json')))
    ap.add_argument('--set',nargs='+')
    ap.add_argument('--out',required=True)
    ap.add_argument('--name',default='macro_ensemble')
    args=ap.parse_args()
    members=[joblib.load(path) for path in args.models]
    if any('base_params' not in m or 'tracking_ball_sha256' not in m for m in members):
        raise ValueError('fit members with run_hybrid.py to record baseline provenance')
    routers = [dict(expert=i,primary_max=args.primary_max,expert_min=args.expert_min)
               for i in (args.team_expert or [])]
    router = routers[0] if len(routers)==1 else None
    multiple = routers if len(routers)>1 else None
    temporal = (dict(consensus_weights=args.consensus_weights,
                     consensus_max_frames=args.consensus_max_frames,
                     consensus_min_iou=args.consensus_min_iou,
                     consensus_dead_weight=args.consensus_dead_weight,
                     oscillation_max_frames=args.oscillation_max_frames)
                if args.consensus_weights is not None else None)
    if (args.oscillation_max_frames or args.consensus_dead_weight) and temporal is None:
        raise ValueError('temporal processing requires --consensus-weights')
    dead_router = (dict(expert=args.dead_expert,weight=args.dead_expert_weight)
                   if args.dead_expert is not None else None)
    artifact=hybrid.combine_artifacts(members,args.weights,router,multiple,temporal,dead_router)
    p=Params.load(args.params,**parse_overrides(args.set))
    physical=lambda value:{k:v for k,v in value.items() if not k.startswith('hybrid_')}
    if physical(artifact['base_params'])!=physical(asdict(p)):
        raise ValueError('ensemble members use a different physical baseline')
    inputs=pipeline.Inputs()
    if artifact['tracking_ball_sha256']!=hashlib.sha256(inputs.atd['ball'].tobytes()).hexdigest():
        raise ValueError('ensemble crops use different ball tracking')
    base,dbg=pipeline.predict(inputs,replace(p,hybrid_model=''),debug=True)
    labels=hybrid.refine(inputs,p,base,dbg['emit'],dbg['bonus'],dbg['shape'],artifact)
    result=pipeline.score(labels)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    artifact['member_sha256']=[hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in args.models]
    joblib.dump(artifact,path)
    pipeline.write(str(Path(ROOT,'evaluation/results',args.name+'.csv')),labels)
    (path.parent/'result.json').write_text(json.dumps(dict(objective='macroF1',selected_on_public=True,
        models=args.models,weights=args.weights,team_router=router,team_routers=multiple,temporal=temporal,
        dead_router=dead_router,
        macroF1=result['macro_f1'],
        spellF1=result['spell_f1_mean'],params=asdict(p)),indent=2)+'\n')
    report(result)


if __name__=='__main__':
    main()
