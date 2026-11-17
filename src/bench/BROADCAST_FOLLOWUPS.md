# Broadcast follow-ups to macro v9

These screens reuse the same 17 public possessions for method selection.
Every classifier uses matching time heads, training-only preprocessing and a
125-frame purge from actual fold edges. Whole-half decoding and temporal
operations precede public stitching. The five-fold baseline is
0.962321 macroF1 / 0.903226 spellF1. No private GT is read.

| Screen | Profiles, including controls | Best macroF1 / spellF1 |
|---|---:|---:|
| Temporal geometry, H/A | 18 | 0.962035 / 0.903226 |
| Temporal geometry, phase | 6 | 0.962321 / 0.903226 |
| Broadcast crossfade residuals and flow | 12 | 0.962321 / 0.903226 |
| Broadcast FILM pilot | 12 | 0.962321 / 0.903226 |
| Broadcast audio and controls | 8 | 0.962457 / 0.903226 |
| Explicit availability and its control | 2 | **0.963167 / 0.903226** |

The audio screen's winner is its **geometry-only control**, which changes
training coverage. Its mean macroF1 across 3/4/5/6 partitions falls slightly.
Relative to v9's quality-only head, adding explicit availability improves macroF1
in all four partitions while retaining spellF1. Compared with the all-row,
no-indicator control, it improves the mean and three of four partitions.
This becomes [macro v10](../MACRO_V10.md).
Ties and overlapping controls in this table are not independent experiments.

## Features and extraction

The temporal screen adds since/until sustained control and +/-1 s lags to the
17 broadcast columns. Control requires distance and relative speed below 1.5,
with runs shorter than eight frames removed. Team heads use 1,049 native alive
rows at 5 Hz; phase heads use 223 quality-supported native rows at 1 Hz.
Groups include geometry alone and concatenated tactical video context. Logistic
and fixed histogram-boosting heads are screened under frozen routing rules.

The motion extractor uses the existing global dense frame plan and quality
gate: 1,299 aligned broadcast samples. Cleaned/fused tactical ball coordinates
centre crops in the broadcast camera. The crossfade screen has 12 distinct
residual/count/flow features. The FILM pilot extracts 223 native public 1 Hz
samples, using the frozen local FILM SavedModel and the same +/-2 frame task.
It retains 222 training rows and supports 5,237 public frames after removing
isolated samples and coverage holes. Its 20 features include FILM and crossfade
residuals. Neither extraction reads GT labels. A successful pilot would require
full-plan extraction before deployment; this pilot was rejected.

The audio screen applies the existing 3.8 kHz spectral detector to broadcast
audio and maps scores through the recovered per-half clocks. Score, peak
envelopes and time to/from candidate blasts provide five features. The fixed
24 dB detector yields 99 candidate events over the graded halves; these are
not annotated whistle counts. Geometry/audio and video/geometry/audio groups
use all 301 native training rows. No audio variant beats v9.

The portable motion extractor reproduces all 1,299 prototype sample indices
and feature values exactly. Crossfade controls inside the FILM environment
agree within floating-point tolerance: maximum differences are 7.45e-9 for
mean residual and 1.91e-6 for flow percentile; other control fields are exact.
All 36 temporal/crossfade candidate label arrays reproduce their prototypes
exactly over 199,319 frames. The default remains identical until a candidate
passes production validation.

## Reproduce

All per-frame features, audio, probabilities, labels and model checkpoints stay
in ignored cache directories. Source/video/model fingerprints are recorded.
With the existing macro-v9 assets:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/broadcast_followup_bench.py \
  --suite temporal_team --out cache/broadcast_followups/temporal_team
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/broadcast_followup_bench.py \
  --suite temporal_phase --out cache/broadcast_followups/temporal_phase
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=3 .venv-cv/bin/python src/bench/broadcast_motion_extract.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/broadcast_followup_bench.py \
  --suite motion --out cache/broadcast_followups/motion
TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=3 \
  .venv-v3/bin/python src/bench/broadcast_motion_extract.py --interpolator film \
  --public-pilot --film-model cache/film_model/2bc8869ae42f586158bb5df0a65ce2103bcc940c \
  --out cache/broadcast_film_pilot/features.npz --chunk 8
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/broadcast_followup_bench.py \
  --suite motion --motion-cache cache/broadcast_film_pilot/features.npz \
  --out cache/broadcast_followups/film
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/broadcast_audio_bench.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/broadcast_followup_bench.py \
  --suite missing_phase --out cache/broadcast_followups/missing_phase
```

The benchmark requires v9's phase ensemble by default. Use `--folds 3`, `4`
or `6` for a paired sensitivity run with all original members refitted on the
same features. The availability indicator is derived from the geometric gate;
it contains no possession label or absolute time feature.
