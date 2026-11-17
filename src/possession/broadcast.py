"""Align the second recording and derive geometric evidence without reading GT."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate

from games import asset_paths, source_frames
from . import atd, clean, contact, homography
from .paths import CACHE_DIR

FEATURES = ('broadcast_quality', 'broadcast_player_match', 'broadcast_players',
    'broadcast_ball_seen', 'broadcast_since_ball', 'broadcast_ball_speed',
    'broadcast_contact_team', 'broadcast_dist_home', 'broadcast_relative_home',
    'broadcast_player_speed_home', 'broadcast_count_home', 'broadcast_dist_away',
    'broadcast_relative_away', 'broadcast_player_speed_away', 'broadcast_count_away',
    'broadcast_team_distance', 'broadcast_ball_disagreement')
CONTEXT_FEATURES = ('broadcast_available_since', 'broadcast_available_until',
                    'broadcast_available_fraction25', 'broadcast_available_fraction125')


def availability_context(available, blocks):
    """GT-free, per-half context of short versus sustained tracking failures."""
    result = {}
    for key, function in [('since', clean.since_last), ('until', clean.until_next)]:
        value = np.full(len(available), np.nan)
        for _, first, last in blocks:
            value[first:last+1] = np.minimum(function(available[first:last+1]), 125.)/25
        result['broadcast_available_'+key] = value
    for width in (25, 125):
        value = np.full(len(available), np.nan)
        for _, first, last in blocks:
            value[first:last+1] = uniform_filter1d(available[first:last+1].astype(float), width, mode='nearest')
        result[f'broadcast_available_fraction{width}'] = value
    return result


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def align(tactical, tactical_seen, ball, seen, blocks, scale):
    """Find independent per-half offsets from robust 5-minute trajectory matches.

    Replays and cuts can produce bad anchors. Accept at least three low-error
    anchors per half, with no more than five frames of offset disagreement.
    Geometry is shared by both provided recordings; reflected matches abstain.
    """
    windows = []
    broadcast = ball[::25, :2]
    bm = seen[::25].astype(float)
    for half, first, last in blocks:
        for start in range(first, last+1, 7500):
            stop = min(last+1, start+7500)
            if stop-start < 2500:
                start = max(first, last+1-7500)
            window = np.arange(start, stop)
            samples = window[::25]
            a = np.nan_to_num(tactical[samples]*scale)
            am = tactical_seen[samples].astype(float)
            coarse = []
            for flip_x, flip_y in [(False, False), (True, False), (False, True), (True, True)]:
                b = broadcast.copy()
                if flip_x:
                    b[:, 0] = 1-b[:, 0]
                if flip_y:
                    b[:, 1] = 1-b[:, 1]
                b = np.nan_to_num(b*scale)
                count = correlate(bm, am, mode='valid', method='fft')
                cost = (correlate((b*b).sum(1)*bm, am, mode='valid', method='fft')
                        + correlate(bm, (a*a).sum(1)*am, mode='valid', method='fft'))
                for axis in range(2):
                    cost -= 2*correlate(b[:, axis]*bm, a[:, axis]*am, mode='valid', method='fft')
                scores = np.maximum(0, cost)/np.maximum(count, 1)
                scores[count < max(40, .2*len(samples))] = np.inf
                for _ in range(2):
                    peak = int(scores.argmin())
                    coarse.append((float(scores[peak]), int(peak*25-start), flip_x, flip_y))
                    scores[max(0, peak-20):peak+21] = np.inf
            candidates = []
            for _, offset, flip_x, flip_y in sorted(coarse)[:3]:
                best = None
                for delta in range(offset-35, offset+36):
                    mapped = window+delta
                    inside = (mapped >= 0) & (mapped < len(ball))
                    target, source = window[inside], mapped[inside]
                    valid = tactical_seen[target] & seen[source]
                    if valid.sum() < max(200, .1*len(window)):
                        continue
                    xy = ball[source[valid], :2].copy()
                    if flip_x:
                        xy[:, 0] = 1-xy[:, 0]
                    if flip_y:
                        xy[:, 1] = 1-xy[:, 1]
                    distance = np.linalg.norm((xy-tactical[target[valid]])*scale, axis=1)
                    metric = float(np.sqrt(np.mean(np.minimum(distance, 20.)**2)))
                    row = dict(offset=int(delta), flip_x=flip_x, flip_y=flip_y,
                        clipped_rmse_m=metric, median_error_m=float(np.median(distance)),
                        p90_error_m=float(np.quantile(distance, .9)), overlap=int(valid.sum()))
                    if best is None or metric < best['clipped_rmse_m']:
                        best = row
                if best is not None:
                    candidates.append(best)
            candidates.sort(key=lambda value: value['clipped_rmse_m'])
            windows.append(dict(half=int(half), first=int(start), last=int(stop), candidates=candidates))
    mapping = np.full(len(tactical), -1, np.int64)
    accepted = []
    for half, first, last in blocks:
        anchors = [w['candidates'][0] for w in windows if w['half'] == half and w['candidates']
                   and w['candidates'][0]['median_error_m'] < 2.
                   and w['candidates'][0]['overlap'] > 250]
        if (len(anchors) < 3 or any(a['flip_x'] or a['flip_y'] for a in anchors)):
            raise ValueError('broadcast alignment has insufficient consistent geometric anchors')
        offsets = np.array([a['offset'] for a in anchors])
        if np.ptp(offsets) > 5:
            raise ValueError('broadcast clock is not stable within this half')
        offset = int(np.rint(np.median(offsets)))
        target = np.arange(first, last+1)
        mapped = target+offset
        valid = (mapped >= 0) & (mapped < len(ball))
        mapping[target[valid]] = mapped[valid]
        accepted.append(dict(half=int(half), offset=offset, anchors=len(anchors),
                             offset_range=int(np.ptp(offsets))))
    return mapping, accepted, windows


def _build(inputs, p, players, ball, teams, error, valid):
    scale = np.asarray(inputs.atd['pitch'])
    tactical = inputs.atd['ball'][:, :2]
    tactical_seen = (np.isfinite(tactical).all(1) & (tactical[:, 0] > -.1)
        & (tactical[:, 0] < 1.1) & (tactical[:, 1] > -.1) & (tactical[:, 1] < 1.1))
    good = (valid & (error < 1.) & np.isfinite(ball[:, :2]).all(1)
        & (ball[:, 0] > -.1) & (ball[:, 0] < 1.1) & (ball[:, 1] > -.1) & (ball[:, 1] < 1.1))
    mapping, alignment, windows = align(tactical, tactical_seen, ball, good, inputs.blocks, scale)
    source = np.clip(mapping, 0, len(ball)-1)
    bp = players[source, :, :2].astype(np.float32)*scale
    bb = ball[source, :2].astype(np.float32)*scale
    tracking = inputs.tracking(p)
    tp, tt = tracking['players_xy'], tracking['teams']
    quality = (mapping >= 0) & valid[source] & (error[source] < 1.)
    bp[(players[source, :, 2] > 12.) | ~np.isfinite(bp).all(2)
       | (bp[:, :, 0] < -5) | (bp[:, :, 0] > scale[0]+5)
       | (bp[:, :, 1] < -5) | (bp[:, :, 1] > scale[1]+5)] = np.nan
    match = np.full(inputs.n, np.nan, np.float32)
    bc, tc = np.isfinite(bp).all(2).sum(1), np.isfinite(tp).all(2).sum(1)
    for first in range(0, inputs.n, 1000):
        last = min(first+1000, inputs.n)
        distances = []
        for side in (0, 1):
            b, t = bp[first:last, teams == side], tp[first:last, tt == side]
            d = np.linalg.norm(b[:, :, None]-t[:, None, :], axis=-1)
            d = np.nan_to_num(d, nan=np.inf).min(2)
            d[~np.isfinite(d)] = np.nan
            distances.append(d)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            match[first:last] = np.nanmedian(np.column_stack(distances), axis=1)
    quality &= (match < 1.5) & (bc >= 5) & (tc >= 8)
    seen = (quality & np.isfinite(bb).all(1) & (bb[:, 0] > -.5)
            & (bb[:, 0] < scale[0]+.5) & (bb[:, 1] > -.5) & (bb[:, 1] < scale[1]+.5))
    bb[~seen], bp[~quality] = np.nan, np.nan
    bv, pv = np.full_like(bb, np.nan), np.full_like(bp, np.nan)
    age, vote = np.full(inputs.n, 1000.), np.full(inputs.n, -1, np.int8)
    for _, first, last in inputs.blocks:
        sl = slice(first, last+1)
        bv[sl], pv[sl] = clean.velocity(bb[sl], 2), clean.velocity(bp[sl], 2)
        age[sl] = np.minimum(clean.since_last(seen[sl]), clean.until_next(seen[sl]))
        contacts = contact.contact_runs(bp[sl], bb[sl], teams)
        owners = contact.owners(contacts, last-first+1, .5)
        owners[age[sl] > 125] = -1
        vote[sl] = owners
    columns = dict(broadcast_quality=1/(1+error[source]), broadcast_player_match=match,
        broadcast_players=bc.astype(float), broadcast_ball_seen=seen.astype(float),
        broadcast_since_ball=age, broadcast_ball_speed=np.linalg.norm(bv, axis=1),
        broadcast_contact_team=np.where(vote < 0, 0., np.where(vote == 0, 1., -1.)))
    for side, name in ((0, 'home'), (1, 'away')):
        distance = np.linalg.norm(bp[:, teams == side]-bb[:, None], axis=2)
        distance = np.nan_to_num(distance, nan=np.inf)
        nearest = distance.argmin(1)
        minimum = distance[np.arange(inputs.n), nearest]
        velocity = pv[:, teams == side][np.arange(inputs.n), nearest]
        minimum[~np.isfinite(minimum)] = np.nan
        columns['broadcast_dist_'+name] = minimum
        columns['broadcast_relative_'+name] = np.linalg.norm(velocity-bv, axis=1)
        columns['broadcast_player_speed_'+name] = np.linalg.norm(velocity, axis=1)
        columns['broadcast_count_'+name] = np.isfinite(bp[:, teams == side]).all(2).sum(1).astype(float)
    columns['broadcast_team_distance'] = columns['broadcast_dist_away']-columns['broadcast_dist_home']
    columns['broadcast_ball_disagreement'] = np.linalg.norm(bb-tracking['ball']['xy'], axis=1)
    for values in columns.values():
        values[~quality] = np.nan
    return columns, quality, mapping, alignment, windows


def features(inputs, p):
    data_dir = getattr(inputs, 'data_dir', None)
    paths = asset_paths(inputs.game, data_dir, source='broadcast')
    physical = {k: v for k, v in asdict(p).items() if not k.startswith('hybrid_')}
    signature = dict(version=1, sources={k: digest(paths[k]) for k in ('atd_meta', 'atd_raw', 'homography')},
        tactical_ball_sha256=hashlib.sha256(inputs.atd['ball'].tobytes()).hexdigest(),
        tactical_players_sha256=hashlib.sha256(inputs.atd['players'].tobytes()).hexdigest(),
        fused_ball_sha256=hashlib.sha256(inputs.ball_atd(p)['ball'].tobytes()).hexdigest(),
        pitch=list(inputs.atd['pitch']), blocks=[list(b) for b in inputs.blocks], params=physical)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()

    def build():
        folder = Path(CACHE_DIR, 'broadcast')
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / f'{inputs.game}_features.npz'
        sidecar = file.with_suffix('.json')
        if file.exists() and sidecar.exists():
            metadata = json.loads(sidecar.read_text())
            if metadata.get('inputs') == signature:
                if digest(file) != metadata['feature_cache_sha256']:
                    raise ValueError('broadcast feature cache was modified')
                cached = np.load(file)
                return {k: cached[k] for k in FEATURES}, cached['mask'], metadata
        if tuple(atd.team_names(paths['atd_meta'])) != tuple(inputs.atd['names']):
            raise ValueError('broadcast and tactical team metadata differ')
        teams = atd.slot_teams(paths['atd_meta'])
        players, ball = atd._parse(paths['atd_raw'], len(teams), source_frames('broadcast', inputs.game))
        _, error, valid = homography._read(inputs.game, data_dir, 'broadcast')
        columns, mask, mapping, alignment, windows = _build(inputs, p, players, ball, teams, error, valid)
        temporary = Path(str(file)+'.tmp.npz')
        np.savez_compressed(temporary, mask=mask, mapping=mapping, **columns)
        temporary.replace(file)
        metadata = dict(version=1, inputs=signature, alignment=alignment,
            feature_cache_sha256=digest(file), public_labels_used=False, columns=list(FEATURES))
        sidecar.write_text(json.dumps(metadata, indent=2)+'\n')
        (folder / f'{inputs.game}_alignment.json').write_text(json.dumps(windows, indent=2)+'\n')
        return columns, mask, metadata

    return inputs._get('broadcast_features', (key,), build)


def provenance(inputs, p, names=()):
    _, available, metadata = features(inputs, p)
    if not any(name in CONTEXT_FEATURES for name in names):
        return metadata
    columns = availability_context(available, inputs.blocks)
    values = np.column_stack([columns[name] for name in CONTEXT_FEATURES])
    return dict(version=2, geometry=metadata, availability_context=dict(version=1,
        names=list(CONTEXT_FEATURES), cap_frames=125, windows_frames=[25,125],
        per_half=True, values_sha256=hashlib.sha256(values.tobytes()).hexdigest()))


def check(inputs, p, artifact):
    if provenance(inputs, p, artifact['names']) != artifact.get('broadcast_provenance'):
        raise ValueError('broadcast evidence differs from the fitted checkpoint; refit')
