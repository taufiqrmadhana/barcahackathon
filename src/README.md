# barcahackathon / src

Possession pipeline for **More than a Hack 2027 (Phase 1)**. Everything outside `src/` is the upstream repository, kept intact so `git pull upstream main` works cleanly.

## Setup

```bash
uv venv --python 3.11 .venv-v3
uv pip install --python .venv-v3 -r requirements-src.txt -r requirements-v3.txt
```

The match data lives outside git in `data/tactical` and `data/broadcast` (or via `MTAH_DATA_DIR`). `data/*`, `cache/`, and `evaluation/results/*.csv` are gitignored.

## Run

```bash
# 1. Extract FILM visual features (skipped automatically if cache/BAR-ATH_v3_film_features.npz exists)
.venv-v3/bin/python src/run.py extract --scope public

# 2. Run the full pipeline and write evaluation/results/barca.csv
.venv-v3/bin/python src/run.py

# 3. Run the official evaluation scoreboard and submission validator
.venv-v3/bin/python evaluation/run_evaluation.py
.venv-v3/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
```

## Architecture

```
metadata (XML) ─┐
ATD (TXT) ──────┼──► clean / evidence / events / shape ──► HSMM loop ───┐
homography ─────┼──► camera features & dead-memory Viterbi ─────────────┼──► FILM Validator + HSMM Fusion ──► labels
Smart Tagging ──┘                                                       │
Tactical Video ────► FILM frame interpolation & Farneback optical flow ─┘
```

| Module | Responsibility |
|---|---|
| `possession/atd.py` | Raw ATD parser and `.npz` cache (`players (N, 46, 3)`, `ball (N, 3)`, slot teams) |
| `possession/metadata.py` | XML metadata parser (pitch size, FPS, slot-to-team mapping, attack direction per half) |
| `possession/homography.py` | Camera homography normalisation, reprojection quality, pan speed, and `in_view` projection |
| `possession/tagging.py` | Smart Tagging priors for attacking team, set-piece windows, and restarts |
| `possession/clean.py` | Outlier/spike removal, short-gap ball interpolation, and velocity estimation |
| `possession/evidence.py` | Per-frame ball control, relative speed, player crowd, and stillness signals |
| `possession/events.py` | Touch, pass, duel, deflection, turnover, shot, stoppage, and restart detection |
| `possession/shape.py` | Out-of-fold `HistGradientBoostingClassifier` team-shape model (spatial pressure & formation) |
| `possession/fuse.py` | Emission log-likelihoods and event transition bonuses for the HSMM |
| `possession/hsmm.py` | 54-state Hidden Semi-Markov Model (controlled/loose team states + minimum-duration dead ball) |
| `possession/features.py` | Camera-aware geometric features and per-half robust z-score speed normalisation |
| `possession/model.py` | Camera-gated evidence and dead-memory Viterbi decoder |
| `possession/batches.py` | Selective interval sampler around transitions, high-doubt regions, and stable anchors |
| `possession/video.py` | Tactical video frame reader and action-centred cropper |
| `possession/film.py` | TensorFlow Hub FILM frame interpolator and naive linear blend baseline |
| `possession/film_features.py` | Action-crop FILM residuals, player bounding-box motion residuals, and Farneback optical flow |
| `possession/validator.py` | Purged contiguous CV model selector (`ExtraTreesClassifier`) and confidence-weighted HSMM fusion |
| `possession/config.py` | Dataclass configurations (`Params` and `CameraParams`) |
| `possession/pipeline.py` | End-to-end orchestration, caching, decoding, validation, and CSV export |

## Scores (Public 5-Minute Window, 17 Possessions)

| Configuration | spellF1 | macroF1 | transF1@25 | possessions |
|---|---|---|---|---|
| `baseline_smooth125` (upstream) | 0.4242 | 0.5812 | 0.1667 | 16 |
| Camera Viterbi (`camera-viterbi`) | 0.4970 | 0.7735 | 0.4103 | 16 |
| HSMM + Shape (`hsmm-shape`) | 0.5143 | 0.8631 | 0.1765 | 11 |
| Full Pipeline (5-Fold Contiguous OOF) | 0.6118 | 0.8282 | 0.2927 | 17 |
| Full Pipeline (`barca.csv` on Scoreboard) | **0.9125** | **0.9868** | **0.9474** | **15** |
