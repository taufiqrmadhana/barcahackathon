"""Extract aligned broadcast residuals/flow without reading possession labels.

The existing tactical frame plan supplies samples. A public pilot uses only the
checkpoint's declared time window and native 1 Hz grid, never target labels.
Crop centres use cleaned/fused tactical ball geometry in the broadcast camera.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import asset_paths, source_frames
from possession import atd, broadcast, features_v2, film, film_features, homography, pipeline
from possession.config import Params
from possession.paths import ROOT
from possession.video import Video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', type=Path, default=Path(ROOT, 'src/configs/macro_v9.json'))
    parser.add_argument('--frames', type=Path, default=Path(ROOT, 'src/cv/frames/valdi_film_dense.txt'))
    parser.add_argument('--out', type=Path, default=Path(ROOT, 'cache/broadcast_motion/features.npz'))
    parser.add_argument('--interpolator', choices=('naive', 'film'), default='naive')
    parser.add_argument('--public-pilot', action='store_true')
    parser.add_argument('--film-model', type=Path, help='local FILM SavedModel directory')
    parser.add_argument('--chunk', type=int, default=24)
    args = parser.parse_args()
    if args.chunk < 1:
        raise ValueError('chunk must be positive')
    if args.interpolator == 'film' and not args.film_model:
        raise ValueError('supply the local FILM SavedModel so its files can be fingerprinted')
    p, inputs = Params.load(args.params), pipeline.Inputs()
    _, quality, provenance = broadcast.features(inputs, p)
    mapping = np.load(Path(ROOT, 'cache/broadcast/BAR-ATH_features.npz'))['mapping']
    paths = asset_paths(inputs.game, inputs.data_dir, source='broadcast')
    plan = np.unique(np.loadtxt(args.frames, dtype=np.int64, ndmin=1))
    if (plan < 0).any() or (plan >= inputs.n).any():
        raise ValueError('sample plan is outside the tactical recording')
    frames = plan[quality[plan] & (mapping[plan] >= film_features.K)
                  & (mapping[plan] < source_frames('broadcast')-film_features.K)]
    pilot = None
    if args.public_pilot:
        checkpoint = joblib.load(p.hybrid_model)
        first, stop = checkpoint['public_start'], checkpoint['public_stop']
        frames = np.intersect1d(frames, np.r_[np.arange(first, stop, 25), stop-1])
        pilot = dict(first=first, stop=stop, stride=25)
    model_hashes = None
    if args.film_model:
        files = sorted(path for path in args.film_model.rglob('*') if path.is_file())
        if not files:
            raise ValueError('FILM model folder is empty')
        model_hashes = {str(path.relative_to(args.film_model)): broadcast.digest(path) for path in files}
    code = ('film_features.py', 'film.py', 'video.py', 'features_v2.py', 'homography.py')
    signature = dict(version=1, interpolator=args.interpolator,
        algorithm='naive +/-2 frame residual and Farneback flow, existing Extractor'
                  if args.interpolator == 'naive' else 'FILM +/-2 frame residual and Farneback flow',
        broadcast=provenance, video_sha256=broadcast.digest(paths['video']),
        plan_sha256=broadcast.digest(args.frames), public_labels_used=False,
        public_pilot=pilot, model_files=model_hashes, features=list(film_features.FEATURES),
        crop=list(film_features.CROP), code_sha256={name: broadcast.digest(Path(ROOT, 'src/possession', name))
                                                 for name in code})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = args.out.with_suffix('.json')
    previous = None
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        if metadata['inputs'] != signature:
            raise ValueError('existing extraction inputs differ; choose a new output')
        if args.out.exists():
            if broadcast.digest(args.out) != metadata.get('cache_sha256'):
                raise ValueError('existing feature cache is modified or incomplete')
            previous = np.load(args.out)
    elif args.out.exists():
        raise ValueError('existing cache has no manifest')
    stored = np.zeros(0, np.int64) if previous is None else previous['frames']
    values = {key: np.zeros(0, np.float32) if previous is None else previous[key]
              for key in film_features.FEATURES}
    if len(np.unique(stored)) != len(stored) or np.setdiff1d(stored, frames).size:
        raise ValueError('existing cache has unexpected sample indices')
    todo = np.setdiff1d(frames, stored)
    print(json.dumps(dict(planned=len(frames), remaining=len(todo), interpolator=args.interpolator)), flush=True)
    if not len(todo):
        return
    teams = atd.slot_teams(paths['atd_meta'])
    players, ball = atd._parse(paths['atd_raw'], len(teams), source_frames('broadcast'))
    scale = np.asarray(inputs.atd['pitch'])
    xy = players[:, :, :2].copy()
    xy[(players[:, :, 2] > 12.) | ~np.isfinite(xy).all(2)
       | (xy[:, :, 0] < -5/scale[0]) | (xy[:, :, 0] > 1+5/scale[0])
       | (xy[:, :, 1] < -5/scale[1]) | (xy[:, :, 1] > 1+5/scale[1])] = np.nan
    centre, blocks = np.full((len(ball), 2), np.nan), []
    tracking = inputs.tracking(p)
    for half, first, last in inputs.blocks:
        targets = np.arange(first, last+1)
        sources = mapping[targets]
        valid = sources >= 0
        centre[sources[valid]] = tracking['ball']['xy'][targets[valid]]/scale
        blocks.append((half, int(sources[valid].min()), int(sources[valid].max())))
    last, _ = features_v2.forward_fill(centre, np.isfinite(centre).all(1), blocks)
    feat = dict(ball_xy=centre*scale, last_ball=last)
    camera = homography.Camera(inputs.game, inputs.data_dir, source='broadcast',
                                pitch_m=scale, ball_norm=ball[:, :2])
    video = Video(inputs.game, inputs.data_dir, source='broadcast')
    if tuple(camera.image_size) != tuple(video.size):
        raise ValueError('broadcast camera projection and video resolution disagree')
    interpolator = (film.NaiveInterpolator() if args.interpolator == 'naive'
                    else film.FilmInterpolator(url=str(args.film_model.resolve())))
    extractor = film_features.Extractor(dict(players=xy, pitch=scale, teams=teams), feat,
        camera, interpolator, film.NaiveInterpolator(), video)
    started = time.monotonic()
    try:
        for start in range(0, len(todo), args.chunk):
            target = todo[start:start+args.chunk]
            rows = extractor.run(mapping[target])
            stored = np.r_[stored, target]
            for key in values:
                values[key] = np.r_[values[key], rows[key]]
            order = np.argsort(stored, kind='stable')
            stored, values = stored[order], {key: value[order] for key, value in values.items()}
            temporary = args.out.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, frames=stored, **values)
            temporary.replace(args.out)
            metadata = dict(inputs=signature, cache_sha256=broadcast.digest(args.out),
                frames=len(stored), complete=len(stored) == len(frames),
                finite={key: int(np.isfinite(value).sum()) for key, value in values.items()})
            sidecar.write_text(json.dumps(metadata, indent=2)+'\n')
            print(json.dumps(dict(done=min(start+args.chunk, len(todo)), total=len(todo),
                                 elapsed=time.monotonic()-started)), flush=True)
    finally:
        video.close()


if __name__ == '__main__':
    main()
