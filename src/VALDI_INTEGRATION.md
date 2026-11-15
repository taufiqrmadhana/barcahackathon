# Valdi merge and FILM fusion

This records the November 12 merge at 3d641db; its exact configuration is now
`configs/pre_pruning.json`. The November 15 default adds the confidence rule documented in
[PRUNING_EXPERIMENTS.md](PRUNING_EXPERIMENTS.md).

`origin/valdi` at 6511c13 is merged into `taufiq`, including its metadata, homography,
video reader, FILM extractor, batch selection, validator and v2/v3 command-line tools.
Its original parameter classes live in `possession/config_valdi.py`, so their defaults
do not change the current tracking/shape decoder. The original v1 feature/model modules
were restored for Valdi v2 compatibility. Its three-state decoder now has an exactly
equivalent optional Numba implementation.

The default in `params.json` now combines the previous contact public-score candidate
with Valdi's video/context validator. On the public five minutes:

| Configuration | spellF1 | macroF1 |
|---|---:|---:|
| Previous contact-only default | 0.6703 | 0.9137 |
| Contact-only public candidate | 0.6947 | 0.9143 |
| **Combined default** | **0.7053** | **0.9231** |

Both prior contact configurations are preserved in `configs/contact_default.json` and
`configs/contact_public.json`. Valdi v2 alone reproduces 0.4970 / 0.7735; its v3 tool
is also runnable independently. The combined method uses the stronger current decoder
as its starting point.

## What is combined

- Valdi's XML team metadata, camera quality/pan, normalised player speeds, dead-ball memory,
  raw-ball geometry and v2 predictions supply context alongside the current decoder and
  external shape probabilities.
- Google Research [FILM](https://github.com/google-research/frame-interpolation) predicts
  each sampled frame from its neighbours at +/-2 frames. Residuals on native-resolution
  256x192 crops, team player boxes, a naive crossfade reference and optical flow supply
  video evidence. The public model is downloaded; competition video is processed locally.
- A regularised logistic classifier is trained at 1 Hz on public labels. The submitted
  public predictions use five contiguous **60-second** held-out folds, with a five-second
  purge. Each fold's classifier predicts and decodes the full halves independently before
  its held-out output is spliced into the public portion of the CSV. This prevents labels
  leaking back through neighbouring frames decoded by another fold's model.
- The all-public classifier is used only outside the public window. No corrected private
  labels or hand corrections are used. The classifier is applied after shape self-training;
  its predictions never become shape pseudo-labels.
- Weight-1 centred class log-probabilities augment the original emissions, followed by
  contact refinement. The previous D mask is locked. Frames without nearby video samples
  retain the contact-only baseline label exactly. Features interpolate only between samples
  at most one second apart, within a selected interval; they never span an unobserved long gap.
- The hybrid's peak windows use video-frame distances, so mixed extraction rates do not
  silently change a peak from 0.4 s to several seconds.

The selected model uses FILM features, not just a merge of files or the original v2 output.
Context-only, optical-flow/crossfade and FILM feature sets, logistic and boosted classifiers,
fusion weights and locked/unlocked D were compared. The boosted model on the former default
raised macroF1 but lowered spellF1 after final sampling and was not promoted.

## Reproduce the deployed file

The existing ATD, fused ball, whistles and ownership/shape checkpoints described in
`README.md` must already be available. Use a separate TensorFlow environment for extraction:
TensorFlow 2.16 requires NumPy 1.x, while the main pipeline uses NumPy 2.4.

```bash
uv venv --python .venv/bin/python .venv-v3
uv pip install --python .venv-v3/bin/python -r requirements-v3.txt
TFHUB_CACHE_DIR=cache/film_model TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 \
  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=3 .venv-v3/bin/python src/run_v3.py extract \
  --frames src/cv/frames/valdi_film.txt --chunk 8
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/contact_public.json --sets film --models logreg \
  --weights 1 --lock-dead 1 --out pretrain_data/valdi_hybrid_public
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
.venv/bin/python evaluation/run_evaluation.py --no-detail
```

The committed frame plan contains **1,257 frame indices**: 850 in the first half, 407 in
the second. It combines public samples and algorithmically selected chunks distributed over
the match. It contains no video, tracking values or possession labels. Extraction is resumable;
running the exact plan again uses the cache and performs no work. The model/cached video
features stay gitignored. Inference from cached features requires no TensorFlow or GPU.

Standalone Valdi commands remain available:

```bash
.venv/bin/python src/run_v2.py --name valdi_v2
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_v3.py run \
  --sets v2+flow+film --name valdi_v3_oof
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python -m unittest discover \
  -s src/bench -p 'test_*.py' -v
```

V3 CSV output now retains held-out predictions on public frames; the original branch wrote
an all-public refit there after reporting an OOF score. Decoder acceleration is checked against
the reference path. Other contracts cover independent parameter defaults, unsupported video
gaps, D/uncovered-frame preservation, and held-out-label perturbations that cannot change
their own fold's decoded output. Real-video seek checks agree at three frames in both halves.

Checkpoint validation rejects a changed baseline configuration or a changed ATD ball when the
cached crops still came from the original tracking. To run the old masked-ball robustness
experiment, pass `--params src/configs/contact_default.json`; cached crops centred on hidden
ATD coordinates must not be reused to claim a leak-free masked score.

## Limits

All choices were inspected on the same public five minutes (17 possessions). Holding out
classifier training does not remove bias from selecting features, models and fusion weights
on those results. The video head has not been validated against corrected private labels or
on unseen matches. The earlier external-game validation applies to its underlying ownership
and shape models, not to this match-specific video head. **Private performance is unknown.**
Aggregates, source/configuration fingerprints, model hashes and final validation results are
recorded in `bench/results/valdi_2026-11-12.json`.
