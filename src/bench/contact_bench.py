"""Compare contact refinement with the same baseline, D mask and inputs.

Public stress tests use round-one cross-fitted shape: the public fold never trains
on its own ball-derived labels. Original ATD AND video-fused tracks are masked;
ATD-centred detections cannot supply hidden coordinates to the contact arm.
External tests use existing leave-one-game-out ownership and shape checkpoints.
Only aggregate scores are written. Competition ground truth stays local.
"""
import argparse
import copy
import gc
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import contact, pipeline
from possession.config import Params
from possession.paths import ROOT, CACHE_DIR
from bench.ownership_bench import metrics

ROOT = Path(ROOT)
VARIANTS = {
    'baseline': {},
    'contact_no_cleanup': dict(w_contact=1., contact_switch_cost=4.),
    'contact_public': dict(w_contact=1., contact_switch_cost=4., contact_min_frames=12),
    'contact_cautious': dict(w_contact=1., contact_switch_cost=16., contact_min_frames=12),
    'contact_default': dict(w_contact=1., contact_switch_cost=2., contact_min_frames=12,
                            contact_local_switch=1),
}


def variants(inputs, p, baseline, dbg, gt, condition, **meta):
    rows = []
    for name, overrides in VARIANTS.items():
        pp = replace(p, **overrides)
        lab = baseline if not overrides else contact.refine(
            inputs, pp, baseline, dbg['emit'], dbg['bonus'])[0]
        row = dict(variant=name, condition=condition, **meta,
                   **metrics(lab, gt, dbg['ev']['seen'], baseline))
        if overrides and p.contact_lock_dead and row['dead_frames_changed']:
            raise AssertionError('contact refinement changed a locked D mask')
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def public(p, seeds, window, out):
    from possession import events, fuse, hsmm, ownership
    seeds = list(seeds)
    mask_paths = [Path(CACHE_DIR, f'cv/mk_hold_{arm}_s{seed}.npz')
                  for seed in seeds for arm in ('atd', 'fused')]
    if any(not path.exists() for path in mask_paths):
        from lt.masked import Bench
        Bench(seeds=seeds, prefix='mk_hold').prepare()
    mask_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in mask_paths}
    base = pipeline.Inputs()
    p = replace(p, shape_rounds=1, w_contact=0, contact_window=window,
                contact_alpha=.5, contact_lock_dead=1, contact_min_frames=0,
                contact_local_switch=0)
    print('build leak-free round-one shape', flush=True)
    _, full = pipeline.predict(base, p, debug=True)
    q = full['shape']
    gt = np.array(['H', 'A', 'D', 'X'])[pipeline.public_gt()]
    rows = []
    full_fused = base.ball_atd(p)['ball'][:, :2].copy()
    del full
    for seed in [-1] + list(seeds):
        inputs = copy.copy(base)
        inputs._cache = {}
        inputs.atd = dict(base.atd, ball=base.atd['ball'].copy())
        pp = p
        if seed >= 0:
            # These tracks are produced by lt.masked.Bench; hidden pass-1 detections
            # were excluded before fusion. Seeds 6..11 are the previous holdout set.
            raw = np.load(Path(CACHE_DIR, f'cv/mk_hold_atd_s{seed}.npz'))['xy']
            fused = np.load(Path(CACHE_DIR, f'cv/mk_hold_fused_s{seed}.npz'))['xy']
            inputs.atd['ball'][11185:18685, :2] = raw[11185:18685]
            xy = full_fused.copy()
            xy[11185:18685] = fused[11185:18685]
            name = f'contact_hold_fused_s{seed}'
            np.savez(Path(CACHE_DIR, f'cv/{name}.npz'), xy=xy)
            pp = replace(p, ball_source=name)
        ev = inputs.evidence(pp)
        evs = events.build(inputs.tracking(pp), ev, inputs.tags, inputs.blocks, pp)
        evs['tag_starts'] = inputs.tags['sp_start']
        evs = fuse.add_whistles(evs, inputs.graded, pp)
        emit, _ = fuse.emissions(ev, q, inputs.tags, evs, pp)
        emit = fuse.force_pre_kickoff(emit, inputs.game, pp)
        bonus = fuse.bonuses(evs, inputs.n, pp, ev)
        if pp.ownership_model:
            emit, bonus = ownership.apply(emit, bonus, ownership.infer(inputs, pp), pp)
        labels, _ = hsmm.Model(pp).decode(emit, bonus, inputs.blocks, inputs.n)
        dbg = dict(emit=emit, bonus=bonus, ev=ev)
        rows += variants(inputs, pp, labels, dbg, gt,
                         'clean' if seed < 0 else 'transplant', seed=seed)
        inputs._cache.clear()
        gc.collect()
        out.write_text(json.dumps(dict(suite='public', shape_rounds=1,
            selected_on_public=True, params=asdict(p), mask_sha256=mask_hashes,
            variants=VARIANTS, results=rows), indent=2)+'\n')
    return rows


def external(window, out, games, conditions, seeds):
    from bench.ownership_bench import inputs_for
    from bench import metrica_open
    from possession import atd
    folder = ROOT/'cache/ownership_v2/logo'
    original = json.loads((folder/'manifest.json').read_text())['params']
    ball = atd.load()['ball']
    rows = []
    for game in games:
        p = Params(**original)
        p = replace(p, contact_window=window,
            pre_kickoff_dead=0, goal_dead=0,
            ownership_model=str(folder/f'ownership_holdout_{game}.joblib'),
            supervised_shape_model=str(folder/f'shape_holdout_{game}.joblib'))
        assert game not in joblib.load(p.ownership_model)['train_games']
        assert game not in joblib.load(p.supervised_shape_model)['train_games']
        for condition in conditions:
            for seed in ([-1] if condition == 'clean' else seeds):
                print('external', game, condition, seed, flush=True)
                inputs = inputs_for(game, condition, seed, ball)
                baseline, dbg = pipeline.predict(inputs, p, debug=True)
                gt = metrica_open.labels(game, inputs.n)
                rows += variants(inputs, p, baseline, dbg, gt, condition, game=game, seed=seed)
                del inputs, dbg
                gc.collect()
                out.write_text(json.dumps(dict(suite='external', selected_on_public=True,
                    heldout_checkpoints=True, contact_window=window, variants=VARIANTS,
                    results=rows), indent=2)+'\n')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', choices=['public', 'external'], default='public')
    ap.add_argument('--params', default=str(ROOT/'src/configs/ownership_v2.json'))
    ap.add_argument('--window', type=int, default=125)
    ap.add_argument('--seeds', type=int, nargs='+')
    ap.add_argument('--games', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--conditions', nargs='+', default=['clean', 'atd', 'corrupt'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--variants', nargs='+', choices=list(VARIANTS))
    args = ap.parse_args()
    if args.variants:
        chosen = {name: VARIANTS[name] for name in args.variants}
        VARIANTS.clear()
        VARIANTS.update(chosen)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.suite == 'public':
        rows = public(Params.load(args.params), args.seeds or [6,7,8,9,10,11], args.window, out)
    else:
        rows = external(args.window, out, args.games, args.conditions, args.seeds or [0])
    print('means (spell / macro)', flush=True)
    for condition in sorted({r['condition'] for r in rows}):
        for variant in VARIANTS:
            selected = [r for r in rows if r['condition']==condition and r['variant']==variant]
            print(condition, variant, np.mean([[r['spellF1'],r['macroF1']] for r in selected], axis=0), flush=True)


if __name__ == '__main__':
    main()
