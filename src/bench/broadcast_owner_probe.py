"""Run the frozen external ownership model on aligned broadcast tracking.

This diagnostic reads no possession GT. Outputs stay in the ignored cache.
The supported recording frames are cleaned with the external model's original
configuration before its team and directed-boundary heads are evaluated.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession.paths import ROOT
from games import asset_paths, source_frames
from possession import atd, broadcast, clean, ownership, pipeline
from possession.config import Params


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--params',default=str(Path(ROOT,'src/configs/macro_v8.json')))
    ap.add_argument('--out',type=Path,default=Path(ROOT,'cache/improve_nov17_owner'))
    args=ap.parse_args()
    inputs=pipeline.Inputs();p=Params.load(args.params)
    _,quality,metadata=broadcast.features(inputs,p)
    cache=Path(broadcast.CACHE_DIR,'broadcast',f'{inputs.game}_features.npz')
    mapping=np.load(cache)['mapping']
    paths=asset_paths(inputs.game,inputs.data_dir,source='broadcast')
    teams=atd.slot_teams(paths['atd_meta'])
    players,ball=atd._parse(paths['atd_raw'],len(teams),source_frames('broadcast',inputs.game))
    source=np.clip(mapping,0,len(ball)-1)
    mapped_players,mapped_ball=players[source].copy(),ball[source].copy()
    mapped_players[players[source,:,2]>12.]=np.nan
    mapped_players[~quality],mapped_ball[~quality]=np.nan,np.nan
    tracking=clean.build(dict(players=mapped_players,ball=mapped_ball,teams=teams,
        names=atd.team_names(paths['atd_meta']),pitch=inputs.atd['pitch']),inputs.blocks,p)
    artifact=joblib.load(p.ownership_model)
    if artifact['cleaning']!={key:asdict(p)[key] for key in ownership.CLEAN_KEYS}:
        raise ValueError('external model cleaning differs from the probe configuration')
    owner,transition,support=ownership.proba(artifact,ownership.features(tracking,inputs.blocks))
    support &= quality
    owner[~support],transition[~support]=.5,0.
    if not (np.isfinite(owner).all() and np.isfinite(transition).all()):
        raise ValueError('external model produced invalid predictions')
    np.testing.assert_allclose(owner.sum(1),1.)
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'broadcast_owner.npz',owner=owner,transition=transition,
        support=support,quality=quality,ball_seen=tracking['ball']['seen'],ball_speed=tracking['ball']['speed'])
    manifest=dict(version=1,public_gt_used=False,broadcast_provenance=metadata,
        ownership_model_sha256=hashlib.sha256(Path(p.ownership_model).read_bytes()).hexdigest(),
        train_games=artifact['train_games'],features=len(artifact['names']),cleaning=artifact['cleaning'],
        supported_frames=int(support.sum()),public_supported_frames=int(support[11185:18685].sum()),
        public_confident_frames=int(((owner[11185:18685].max(1)>.9)&support[11185:18685]).sum()))
    (args.out/'broadcast_owner.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='broadcast_provenance'}))


if __name__=='__main__':
    main()
