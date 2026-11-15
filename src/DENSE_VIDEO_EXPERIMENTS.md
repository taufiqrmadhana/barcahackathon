# Denser video evidence, November 15

Adding video observations improves public possession boundaries while keeping classifier
training at 1 Hz. The deployed five-fold classifier specification and decoder are unchanged.

| Public five minutes | spellF1 | macroF1 | possessions |
|---|---:|---:|---:|
| Start of this improvement round, `configs/pre_blend.json` | 0.7444 | 0.9292 | 19 |
| Sparse-video blend, `configs/pre_dense.json` | 0.7529 | **0.9353** | 17 |
| **Dense-video blend, new default** | **0.7882** | 0.9346 | 17 |

The new default favours spellF1, the [official ranking metric](../EVALUATION.md). Relative to
the intermediate checkpoint, spellF1 gains 0.0353 and macroF1 falls 0.0008. That checkpoint
remains reproducible with its own configuration and cache. Both metrics improve relative to
the configuration at the start of this round. All numbers use the same public 17 possessions;
private performance is unknown.

## Data and model

The plan `cv/frames/valdi_film_dense.txt` contains 1,739 frame indices. It preserves all 1,257
original samples and adds 482 observations to complete a uniform 2.5 Hz grid over the public
window. Existing observations at other cadences remain present. Samples outside that window
are unchanged. Every new crop produced finite mean residual features, and the values of every
original cached sample are exactly preserved in the new store.

`run_v3.py --cache-kind film_dense` writes the separate
`cache/BAR-ATH_v3_film_dense_features.npz`; the original `film` cache is unchanged. The frame
plan contains no image, tracking value or possession label. Extraction took about ten minutes
on the local CPU and is resumable. Re-running the complete plan finds zero missing samples.

The same 42 features and 1:3 complementary logistic blend are fitted on the same 301 native
1 Hz public training rows. The new observations supply more detailed interpolation and motion
peaks between those rows. Five 60-second folds, a five-second purge and independent whole-half
decoding per held-out head remain in use. The full public refit is used outside the public
window. D stays locked and frames without video retain their previous labels.

More training rows did not help: training at 2.5 Hz reached only 0.7059 / 0.9299 with sample-count
normalised regularisation, or 0.6889 / 0.9339 with the original regularisation. The deployed
training cadence and C=0.3 remain unchanged. Shortening or doubling the motion-peak window
also failed; it remains 21 frames.

## Paired time-fold audit

| Time folds | Sparse spell / macro | Dense spell / macro |
|---|---:|---:|
| 3 | 0.6947 / 0.9248 | 0.6737 / 0.9247 |
| 4 | 0.5895 / 0.9159 | 0.7250 / 0.9273 |
| 5, deployed | 0.7529 / 0.9353 | **0.7882 / 0.9346** |
| 6 | 0.6842 / 0.9292 | 0.7333 / 0.9390 |

SpellF1 improves in three partitions and falls in one. These are correlated sensitivity
checks on the same public window, not independent validation sets. Feature, model and
parameter choices were inspected on that window; held-out training does not remove selection
bias. The extra video has not been validated on corrected private labels or unseen matches.

Further rejected screens tested classifier-based boundary snapping, Smart Tagging team/context
features, training weights for short spells, normalised/log video features, nonlinear SVM
heads, stronger/weaker/sparse logistic regularisation and final contact weights. None improved
the chosen spell/macro tradeoff. For example, stronger logistic fitting reached 0.9435 macroF1
but only 0.7111 spellF1. These results do not show those methods cannot help on other data.

## Reproduce

Use the existing ATD, fused ball, whistles and external ownership/shape checkpoints:

```bash
TFHUB_CACHE_DIR=cache/film_model TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 \
  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=3 .venv-v3/bin/python src/run_v3.py extract \
  --frames src/cv/frames/valdi_film_dense.txt --cache-kind film_dense --chunk 8
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/pre_dense.json --sets film --models logreg_blend --interp film_dense \
  --weights 1 --lock-dead 1 --out pretrain_data/valdi_hybrid_dense --name hybrid_dense
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
.venv/bin/python evaluation/run_evaluation.py --no-detail
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/video_density_bench.py
# Restore the intermediate checkpoint's output:
.venv/bin/python src/run.py --params src/configs/pre_dense.json --name before_dense
```

Inference uses cached features and requires no TensorFlow. Match features, model checkpoints
and submissions stay gitignored. Aggregate screens, paired results and fingerprints are in
`bench/results/video_density_2026-11-15.json`.

Production, the fitting CLI and the frozen-shape tuning path produce identical labels across
all 199,319 frames. The official format validator and scorer confirm the submission and public
metrics. The intermediate configuration reproduces its exact previous CSV. Relative to that
checkpoint, 148 frames change (138 public, ten elsewhere); D and uncovered frames are unchanged,
as are the full-match count of 404 possessions and dead share of 46.13%.
