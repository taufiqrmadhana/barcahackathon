"""Extract FILM at raw or fused ball centres, without reading possession labels.

Use a separate cache for each crop source. Existing undocumented feature rows are
never appended to: mixing crop coordinates invalidates the comparison.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from possession import film, film_features, pipeline, pipeline_v2, video_provenance
from possession.config import Params
from possession.config_valdi import ParamsV2
from possession.paths import SRC


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--cache-kind', required=True)
    ap.add_argument('--params', default=str(Path(SRC, 'params.json')))
    ap.add_argument('--legacy-params', default=str(Path(SRC, 'params_v2.json')))
    ap.add_argument('--ball-source', help='raw, or a fused cache name; default: physical config source')
    ap.add_argument('--chunk', type=int, default=8)
    args = ap.parse_args()
    if args.chunk < 1:
        raise ValueError('chunk must be positive')
    p = Params.load(args.params)
    source = p.ball_source if args.ball_source is None else args.ball_source
    if source == 'raw':
        source = ''
    physical = pipeline.Inputs()
    inputs = pipeline_v2.Inputs()
    if (not np.array_equal(physical.atd['teams'], inputs.atd['teams'])
            or not np.allclose(physical.atd['pitch'], inputs.atd['pitch'])):
        raise ValueError('physical and video tracking coordinate systems differ')
    inputs.atd = physical.ball_atd(replace(p, ball_source=source))
    vp = ParamsV2.load(args.legacy_params)
    frames = np.unique(np.loadtxt(args.frames, dtype=np.int64, ndmin=1))
    if not len(frames) or (frames < 2).any() or (frames >= inputs.meta.n_frames-2).any():
        raise ValueError('sample frames and their +/-2 neighbours must be inside the video')
    expected = video_provenance.metadata(physical, p, args.cache_kind, asdict(vp), source)
    meta_path = video_provenance.metadata_path(args.cache_kind, physical.game)
    store = film_features.Store(args.cache_kind, physical.game)
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != expected:
            raise ValueError('cache uses different crop coordinates; choose a new cache kind')
    elif len(store.frames):
        raise ValueError('existing cache has no crop metadata; choose a new cache kind')
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(expected, indent=2)+'\n')
    missing = store.missing(frames)
    print(f'{len(frames)} planned, {len(missing)} to extract; crop source: {source or "raw"}', flush=True)
    if not len(missing):
        return
    feat = inputs.features(vp)
    cam = inputs.camera(vp.err_scale, vp.pan_window)
    extractor = film_features.Extractor(inputs.atd, feat, cam, film.load('film'), film.load('naive'))
    film_features.extract(extractor, store, frames, chunk=args.chunk)


if __name__ == '__main__':
    main()
