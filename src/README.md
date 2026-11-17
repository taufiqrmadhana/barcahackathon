# Pipeline setup

Run commands from the repository root. The inference environment uses Python 3.11
and `requirements-src.txt`. Detection and FILM extraction use separate environments
(`requirements-cv.txt` and `requirements-v3.txt`).

## Data and artifacts

Place the licensed bundle under `data/`, or set `MTAH_DATA_DIR` to its directory.
See [DATA.md](../DATA.md) for the layout. Match data, extracted features and fitted
models stay outside Git.

The default configuration is `src/params.json`. It requires:

- Fused ball tracking and whistle features in `cache/cv/`.
- Raw dense, raw 5 Hz and fused-crop FILM features in `cache/`.
- Ownership and shape models in `pretrain_data/ownership_v2/`.
- The final ensemble in `pretrain_data/macro_v11/ensemble.joblib`.

Model and feature provenance is checked at inference. Use the matching artifacts,
or regenerate the features and refit the models together.

## Prepare observations

With the detection environment installed:

```bash
.venv-cv/bin/python src/cv/export_models.py
.venv-cv/bin/python src/cv/detect_crops.py --start 11185 --end 54000
.venv-cv/bin/python src/cv/detect_grid.py --frames src/cv/frames/grid_frames_pub.txt --tag grid_pub
.venv-cv/bin/python src/cv/detect_grid.py --frames src/cv/frames/grid_frames_pub_rest.txt --tag grid_pub_rest
.venv-cv/bin/python src/cv/detect_grid.py --frames src/cv/frames/grid_frames.txt --tag grid
.venv/bin/python src/cv/build_fused.py --tag atd+grid_pub+grid_pub_rest+grid --name fused_full
.venv-cv/bin/python src/cv/whistle.py
```

FILM uses a TensorFlow environment with NumPy 1.x. Extract each view separately:

```bash
.venv-v3/bin/python src/run_v3.py extract --frames src/cv/frames/valdi_film_dense.txt --cache-kind film_dense
.venv-v3/bin/python src/run_v3.py extract --frames src/cv/frames/valdi_film_5hz.txt --cache-kind film_5hz
.venv-v3/bin/python src/extract_aligned_video.py \
  --frames src/cv/frames/valdi_film_dense.txt --cache-kind film_fused_dense \
  --ball-source fused_full
```

The committed sampling plans contain frame indices only. Training stays on the
native 1 Hz public grid, regardless of the image sampling rate.

## Fit the models

Download the external Metrica samples and fit ownership/shape models:

```bash
.venv/bin/python src/bench/metrica_open.py --download
.venv/bin/python src/bench/shape_bench.py --fit-all --out pretrain_data/ownership_v2
```

The ensemble has six members. Use `src/run_hybrid.py` to fit them from the prepared
features, with `--weights .5 --lock-dead 1 --objective macro`:

| Output directory | Feature cache (`--interp`) | Features (`--sets`) | Model (`--models`) | C |
|---|---|---|---|---:|
| `pretrain_data/macro_v1` | `film_dense` | `film_soft` | `logreg` | 1 |
| `pretrain_data/macro_view5hz` | `film_5hz` | `film_soft` | `logreg` | 3 |
| `pretrain_data/macro_motion` | `film_dense` | `film_motion` | `logreg` | 1 |
| `pretrain_data/macro_roi_fused` | `film_fused_dense` | `film_soft` | `logreg` | 3 |
| `pretrain_data/macro_dead_video_c10` | `film_dense` | `film_dead_motion` | `dead_logreg` | 10 |
| `pretrain_data/broadcast_dead_context` | `film_dense` | `broadcast_dead_context` | `dead_context_blend` | 0.3 |

Pass C through `--logreg-c`, the output directory through `--out`, and use
`--strict-fold-edges` for the broadcast model. All members use matching five-fold
intervals and a 125-frame purge. The broadcast member contains two binary heads;
the final output still has four labels.

Combine the fitted members:

```bash
.venv/bin/python src/run_ensemble.py \
  --models pretrain_data/macro_v1/film_soft_logreg.joblib \
           pretrain_data/macro_view5hz/film_soft_logreg.joblib \
           pretrain_data/macro_motion/film_motion_logreg.joblib \
           pretrain_data/macro_roi_fused/film_soft_logreg.joblib \
           pretrain_data/macro_dead_video_c10/film_dead_motion_dead_logreg.joblib \
           pretrain_data/broadcast_dead_context/broadcast_dead_context_dead_context_blend.joblib \
  --weights .75 .25 0 0 0 0 --team-expert 2 --team-expert 3 \
  --consensus-weights .5 0 .5 0 0 0 --consensus-dead-weight .5 \
  --oscillation-max-frames 100 --dead-expert 4 --dead-expert-weight .5 \
  --phase-expert 5 --phase-weight .75 --phase-dead-margin 50 --phase-keep-teams \
  --out pretrain_data/macro_v11/ensemble.joblib --name Borrow_Khan
```

Refitting may change artifact checksums; compare regenerated predictions with the
frozen aggregate results in `src/bench/results/`.
