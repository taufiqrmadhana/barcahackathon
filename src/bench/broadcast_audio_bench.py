"""Test aligned broadcast whistle evidence with the frozen public-fold protocol."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.ndimage import maximum_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broadcast_followup_bench import Probe
from cv import whistle
from games import asset_paths, source_frames
from possession import broadcast, clean, hybrid, pipeline
from possession.config import Params
from possession.paths import ROOT


def extract(params, folder):
    inputs, p = pipeline.Inputs(), Params.load(params)
    provenance = broadcast.features(inputs, p)[2]
    folder.mkdir(parents=True, exist_ok=True)
    path = asset_paths(inputs.game, inputs.data_dir, 'broadcast')['video']
    video_hash = broadcast.digest(path)
    detector_hash = broadcast.digest(Path(ROOT, 'src/cv/whistle.py'))
    file, sidecar = folder / 'score.npy', folder / 'score.json'
    if file.exists() and sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        if (metadata['video_sha256'] != video_hash or metadata['detector_sha256'] != detector_hash
                or metadata['broadcast'] != provenance or broadcast.digest(file) != metadata['score_sha256']):
            raise ValueError('audio extraction inputs or cached scores differ; choose a new folder')
        return np.load(file), metadata
    original = whistle.RAW
    try:
        whistle.RAW = str(folder / 'audio16k.raw')
        whistle.extract_audio(path)
        score = whistle.compute(source_frames('broadcast'), chunk=2000)
        np.save(file, score)
        metadata = dict(version=1, video_sha256=video_hash,
            score_sha256=broadcast.digest(file), raw_sha256=broadcast.digest(whistle.RAW),
            detector_sha256=detector_hash, broadcast=provenance, public_labels_used=False,
            n_frames=len(score), score_min=float(score.min()), score_max=float(score.max()),
            threshold=p.whistle_thresh)
        sidecar.write_text(json.dumps(metadata, indent=2)+'\n')
        return score, metadata
    finally:
        whistle.RAW = original


def features(score, mapping, inputs, threshold):
    coverage = inputs.graded & (mapping >= 0) & (mapping < len(score))
    aligned = np.full(inputs.n, np.nan)
    aligned[coverage] = score[mapping[coverage]]
    columns = {'broadcast_audio_score': aligned}
    event_frames = []
    for _, a, b in inputs.blocks:
        event_frames.extend(whistle.events(aligned[a:b+1], coverage[a:b+1], threshold)[0]+a)
    on = np.zeros(inputs.n, bool)
    on[event_frames] = True
    for name, width in (('peak25', 25), ('peak125', 125)):
        value = np.full(inputs.n, np.nan)
        for _, a, b in inputs.blocks:
            value[a:b+1] = maximum_filter1d(np.nan_to_num(aligned[a:b+1], nan=-100.), width, mode='nearest')
        columns['broadcast_audio_'+name] = value
    for name, function in (('since', clean.since_last), ('until', clean.until_next)):
        value = np.full(inputs.n, np.nan)
        for _, a, b in inputs.blocks:
            value[a:b+1] = np.minimum(function(on[a:b+1]), 125.)/25
        columns['broadcast_audio_'+name] = value
    return columns, coverage, len(event_frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', type=Path, default=Path(ROOT, 'src/configs/macro_v9.json'))
    parser.add_argument('--audio-cache', type=Path, default=Path(ROOT, 'cache/broadcast_audio'))
    parser.add_argument('--out', type=Path, default=Path(ROOT, 'cache/broadcast_audio_bench'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    score, metadata = extract(args.params, args.audio_cache)
    if args.extract_only:
        print(json.dumps(metadata), flush=True)
        return
    probe = Probe(args.params, args.folds)
    bb, _, provenance = broadcast.features(probe.inputs, probe.p)
    if metadata['broadcast'] != provenance:
        raise ValueError('broadcast geometry changed during audio extraction')
    mapping = np.load(Path(ROOT, 'cache/broadcast/BAR-ATH_features.npz'))['mapping']
    audio, coverage, events = features(score, mapping, probe.inputs, probe.p.whistle_thresh)
    coverage &= probe.mask
    columns = dict(probe.views[2][0])
    columns.update(bb)
    columns.update(audio)
    groups = {'audio': tuple(audio), 'geometry_audio': tuple(bb)+tuple(audio),
              'video_geometry_audio': hybrid.FEATURE_SETS['film_dead_motion']+tuple(bb)+tuple(audio),
              'geometry_control': tuple(bb)}
    grid = np.r_[np.arange(probe.public[0], probe.public[-1]+1, 25), probe.public[-1]]
    native = np.intersect1d(probe.views[0][2], grid)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    report = dict(folds=args.folds, training_stride=25, strict_fold_edges=True,
        purge_frames=probe.a['purge_frames'], selected_on_public=True,
        same_public_possessions=True, private_score_known=False, extraction=metadata,
        baseline=pipeline.score(probe.reference, probe.gt), events=events,
        source_sha256=broadcast.digest(__file__), protocol_sha256=broadcast.digest(Path(__file__).with_name('broadcast_followup_bench.py')),
        support=int(coverage.sum()), public_support=int(coverage[probe.public].sum()), results=results)
    np.save(args.out / 'baseline.npy', probe.reference)
    for group, names in groups.items():
        for family in ('dead_logreg', 'hist_gbm'):
            models, pack, count = probe.fit(columns, names, native, coverage, family, binary=True)
            key = f'{group}_{family}'
            joblib.dump(dict(models=models, names=names, model=family, training_rows=count,
                edges=probe.a['edges'], purge_frames=probe.a['purge_frames'],
                strict_fold_edges=True, extraction=metadata, experimental=True), args.out / (key+'.joblib'))
            labels = probe.render(pack, coverage, 'phase')
            result = pipeline.score(labels, probe.gt)
            row = dict(group=group, model=family, features=len(names), training_rows=count,
                macroF1=result['macro_f1'], spellF1=result['spell_f1_mean'], possessions=result['n_pred_spells'],
                changed_frames=int((labels != probe.reference).sum()),
                public_changes=int((labels[probe.public] != probe.reference[probe.public]).sum()),
                per_class=result['per_class'])
            results.append(row)
            print(json.dumps(row), flush=True)
            np.save(args.out / (key+'.npy'), labels)
            (args.out / 'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
