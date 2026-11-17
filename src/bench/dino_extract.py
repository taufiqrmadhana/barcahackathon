"""Cache frozen DINOv2 features from ball crops, locally and without GT labels.

Run with .venv-cv/bin/python. Prepare the official facebookresearch/dinov2
checkout at the pinned revision before calling this script. Existing caches
resume only when their complete extraction manifest matches.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import homography, pipeline, pipeline_v2
from possession.config import Params
from possession.config_valdi import ParamsV2
from possession.film_features import action_points
from possession.paths import ROOT
from possession.video import Video, crop, crop_box

REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
MODEL_HASH = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=Path, default=Path(ROOT, 'src/cv/frames/valdi_film_5hz.txt'))
    ap.add_argument('--repo', type=Path, default=Path(ROOT, 'cache/models/dinov2_repo'))
    ap.add_argument('--hub', type=Path, default=Path(ROOT, 'cache/models/torch_hub'))
    ap.add_argument('--out', type=Path, default=Path(ROOT, 'cache/improve_nov17/dino_5hz.npz'))
    ap.add_argument('--params', default=str(Path(ROOT, 'src/configs/macro_v6.json')))
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--patch-pools', action='store_true', help='also average tokens near and above the ball')
    args = ap.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('DINOv2 source revision differs from the pinned extractor')
    torch.set_num_threads(args.threads)
    torch.hub.set_dir(str(args.hub.resolve()))
    model = torch.hub.load(str(args.repo), 'dinov2_vits14', source='local', pretrained=True).eval()
    checkpoint = args.hub / 'checkpoints/dinov2_vits14_pretrain.pth'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != MODEL_HASH:
        raise ValueError('DINOv2 weight hash differs')
    frames = np.unique(np.loadtxt(args.frames, dtype=np.int64))
    inputs = pipeline.Inputs()
    params = Params.load(args.params)
    legacy = pipeline_v2.Inputs()
    vp = ParamsV2.load(Path(ROOT, 'src/params_v2.json'))
    features = legacy.features(vp)
    camera = legacy.camera(vp.err_scale, vp.pan_window)
    fused = inputs.ball_atd(params)['ball']
    centres = action_points(inputs.atd, features, inputs.atd['pitch'])
    known = np.isfinite(fused[:, :2]).all(axis=1)
    centres[known] = fused[known, :2]
    video = Video()
    crops = (128, 320)
    manifest = dict(version=1, backbone='dinov2_vits14', model_revision=revision,
        model_sha256=MODEL_HASH, model_source='https://github.com/facebookresearch/dinov2',
        raw_ball_sha256=hashlib.sha256(inputs.atd['ball'].tobytes()).hexdigest(),
        crop_ball_sha256=hashlib.sha256(fused.tobytes()).hexdigest(),
        frame_plan_sha256=hashlib.sha256(args.frames.read_bytes()).hexdigest(),
        camera_params=asdict(vp), ball_source=params.ball_source,
        crops=list(crops), input_size=224, embedding_size=384,
        colour='RGB', normalisation='ImageNet mean/std',
        video_frames=video.n, video_size=list(video.size), public_gt_used=False)
    if args.patch_pools:
        manifest.update(version=2, patch_pools=['ball_neighbourhood', 'above_ball'],
                        patch_pool_size=4, patch_grid=16)
    metadata = args.out.with_suffix('.json')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if metadata.exists():
        if json.loads(metadata.read_text()) != manifest:
            raise ValueError('existing extraction metadata differs')
    else:
        if args.out.exists():
            raise ValueError('existing embeddings have no provenance metadata')
        metadata.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.out.exists():
        old = np.load(args.out)
        done, embeddings = list(old['frames']), list(old['embeddings'])
        pools = list(old['patch_pools']) if args.patch_pools else []
    else:
        done, embeddings, pools = [], [], []
    todo = np.setdiff1d(frames, done)
    mean = np.array([.485, .456, .406], np.float32)
    std = np.array([.229, .224, .225], np.float32)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(todo), args.batch):
            part = todo[start:start + args.batch]
            images = video.frames(part)
            batch, available, anchors = [], [], []
            for t in part:
                if t not in images:
                    raise ValueError(f'video is missing requested frame {t}')
                centre, _ = homography.apply(camera.matrix[[t]], centres[t][None, None])
                for size in crops:
                    origin = crop_box(centre[0, 0], (size, size), video.size)
                    patch = crop(images[t], origin, (size, size))
                    patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_CUBIC)
                    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                    batch.append(((rgb - mean) / std).transpose(2, 0, 1))
                    relative = (centre[0, 0] - np.asarray(origin)) / size
                    if not np.isfinite(relative).all():
                        relative = np.array([.5, .5])
                    anchors.append(np.clip(np.floor(relative*16).astype(int), 0, 15))
                available.append(t)
            tensor = torch.from_numpy(np.stack(batch))
            if args.patch_pools:
                features = model.forward_features(tensor)
                output = model.head(features['x_norm_clstoken']).numpy()
                tokens = features['x_norm_patchtokens'].reshape(len(batch), 16, 16, 384).numpy()
                pooled = []
                for token, (x, y) in zip(tokens, anchors):
                    x0 = int(np.clip(x-2, 0, 12))
                    near_y = int(np.clip(y-2, 0, 12))
                    above_y = int(np.clip(y-4, 0, 12))
                    pooled.append(np.stack([token[near_y:near_y+4, x0:x0+4].mean(axis=(0, 1)),
                                            token[above_y:above_y+4, x0:x0+4].mean(axis=(0, 1))]))
                pools.extend(np.asarray(pooled).reshape(len(available), len(crops), 2, 384))
            else:
                output = model(tensor).numpy()
            output = output.reshape(len(available), len(crops), 384)
            if not np.isfinite(output).all():
                raise ValueError('DINOv2 produced invalid features')
            done.extend(available)
            embeddings.extend(output)
            if start // args.batch % 4 == 0 or start + args.batch >= len(todo):
                order = np.argsort(done)
                temporary = Path(str(args.out) + '.tmp.npz')
                arrays = dict(frames=np.asarray(done, np.int64)[order],
                              embeddings=np.asarray(embeddings, np.float32)[order])
                if args.patch_pools:
                    arrays['patch_pools'] = np.asarray(pools, np.float32)[order]
                np.savez(temporary, **arrays)
                temporary.replace(args.out)
                rate = (start + len(part)) / max(.001, time.monotonic() - started)
                print(f'DINO {start + len(part)}/{len(todo)} frames, {rate:.2f}/s, '
                      f'eta {(len(todo)-start-len(part))/max(.001,rate)/60:.1f} min', flush=True)
    video.close()
    print(f'Complete: {len(done)} frames, {time.monotonic()-started:.1f} s', flush=True)


if __name__ == '__main__':
    main()
