"""Bind an aligned video cache to its crop coordinates and fitted checkpoint."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .paths import CACHE_DIR


def store_path(kind, game):
    return Path(CACHE_DIR, f'{game}_v3_{kind}_features.npz')


def metadata_path(kind, game):
    return Path(str(store_path(kind, game)) + '.json')


def metadata(inputs, p, kind, legacy_params, ball_source):
    ball = inputs.ball_atd(replace(p, ball_source=ball_source))['ball']
    return dict(version=1, kind=kind, ball_source=ball_source,
                raw_ball_sha256=hashlib.sha256(inputs.atd['ball'].tobytes()).hexdigest(),
                crop_ball_sha256=hashlib.sha256(ball.tobytes()).hexdigest(),
                crop_params={k:legacy_params[k] for k in ('max_gap', 'err_scale', 'pan_window')})


def describe(inputs, p, kind, legacy_params):
    """Old raw caches have no sidecar. Aligned caches must match their declared source."""
    path = metadata_path(kind, inputs.game)
    if not path.exists():
        return None
    found = json.loads(path.read_text())
    keys = {'version', 'kind', 'ball_source', 'raw_ball_sha256', 'crop_ball_sha256', 'crop_params'}
    if (not isinstance(found, dict) or set(found) != keys or found.get('version') != 1
            or not isinstance(found['ball_source'], str)):
        raise ValueError('video crop metadata schema differs')
    expected = metadata(inputs, p, kind, legacy_params, found['ball_source'])
    if found != expected:
        raise ValueError('video crop coordinates or extraction parameters differ; re-extract in a new cache')
    return dict(found, feature_store_sha256=hashlib.sha256(
        store_path(kind, inputs.game).read_bytes()).hexdigest())


def check(inputs, p, artifact):
    observed = describe(inputs, p, artifact['kind'], artifact['legacy_params'])
    if observed != artifact.get('video_provenance'):
        raise ValueError('video cache provenance differs from the fitted checkpoint; refit')
