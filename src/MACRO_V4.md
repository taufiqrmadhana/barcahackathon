# Aligned crop expert: public macroF1 0.950003

The new default keeps macro v3's primary video mixture and motion expert, then adds
a second conditional ownership expert whose FILM crops follow the fused ball track.

| Frozen configuration | Public macroF1 | Public spellF1 |
|---|---:|---:|
| Macro v3, `configs/macro_v3.json` | 0.949550 | 0.752941 |
| **Macro v4, current default** | **0.950003** | 0.752941 |
| Retained spell alternative, `configs/balanced_v1.json` | 0.942279 | **0.800000** |

The five-fold gain over v3 is small: four additional correct public H/A frames.
The paired time-partition audit improves at all four tested partitions:

| Time folds | v3 macroF1 | v4 macroF1 | v4 spellF1 |
|---|---:|---:|---:|
| 3 | 0.931539 | 0.938995 | 0.729412 |
| 4 | 0.930173 | 0.935521 | 0.688889 |
| 5, deployed | 0.949550 | 0.950003 | 0.752941 |
| 6 | 0.936910 | 0.943613 | 0.729412 |

These are repartitions of the same 17 public possessions, not independent test sets.
All feature/model choices reuse those labels. The checkpoint has no measured private
or unseen-match score; the fifth decimal is not evidence of statistical significance.

## Approach

The physical baseline combines cleaned/fused tracking, external ownership and player
shape models, whistle evidence, contact ownership and the HSMM decoder. The primary
video probabilities remain 0.75 dense/raw C=1 plus 0.25 raw/5 Hz C=3. Both use the
34-feature continuous/video set.

The first conditional H/A expert is unchanged: dense/raw C=1 with six additional
player-motion features. The second is a 34-feature C=3 head with `film_fused_dense`
crops. Both experts have zero ordinary averaging weight. In order, an expert can
supply conditional H/A only when current confidence is below 0.8 and its own is above
0.9. A confident earlier expert keeps priority. Every expert preserves the primary
P(D), and missing evidence abstains without shrinking primary video coverage.

The new crop store uses exactly the original dense plan's 1,739 indices, including
1,019 public observations. Existing 1,021 fused-crop rows were preserved. The crop
source changes from raw ATD to `fused_full`; training remains the same 301 native
1 Hz rows, five contiguous folds and 125-frame purge. The new head alone reaches
only 0.945989 / 0.637500. Its contribution is selective correction, not a stronger
standalone classifier. Raw 10 Hz sampling alone did not beat the previous checkpoint.

Schema 4 stores an ordered expert list; schemas 1/2/3 retain their existing paths.
All members use aligned held-out heads, with independent whole-half decoding before
public splicing. The all-public refits apply outside the public window. Matching
tracking/crop hashes, feature parameters and cache fingerprints prevent an aligned
head from silently using different video observations. See [VIDEO_ALIGNMENT.md](VIDEO_ALIGNMENT.md).

## Scores and predicted match statistics

The public slice has 17 predicted possessions versus 17 in GT, mean duration 14.04 s
versus 14.19 s. Class F1 values are H 0.947601, A 0.960368, D 0.942039. Transition F1
within one second is 0.7000.

Full-match figures below describe predictions over the two graded half windows:

| Window | Possessions | Mean duration | Home share of alive play | Dead share |
|---|---:|---:|---:|---:|
| First half | 206 | 8.62 s | 61.54% | 44.11% |
| Second half | 198 | 8.33 s | 58.82% | 48.05% |
| Both halves | **404** | **8.48 s** | **60.23%** | **46.08%** |

Home has 199 spells averaging 10.37 s; away has 205 averaging 6.65 s. Away's share of
alive play is 39.77%. There are 130 D intervals when each half is counted separately.
The diagnostic counter now respects half boundaries rather than joining D at the
end of one half to D at the start of the next.

Compared with v3, 68 match labels change, including four public labels and one
non-public D label. P(D) preservation does not guarantee an identical coupled Viterbi
D path. The public D labels, total possession count and unsupported-video labels
remain unchanged. These full-match figures have no full-match GT for comparison.

## Validation and reproduction

All 41 tests pass. The official fitter reproduces the prototype's coefficients.
Prototype, builder, production and frozen-shape tuner ensemble predictions match
over all 199,319 frames. Macro v3 still reproduces its exact
previous CSV. The official validator and public scorer confirm the new default.

With the existing raw dense/5 Hz and motion members:

```bash
TFHUB_CACHE_DIR=cache/film_model TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 \
  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=3 .venv-v3/bin/python src/extract_aligned_video.py \
  --frames src/cv/frames/valdi_film_dense.txt --cache-kind film_fused_dense \
  --ball-source fused_full --chunk 8
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/macro_v3.json --sets film_soft --models logreg \
  --interp film_fused_dense --logreg-c 3 --objective macro --weights .5 --lock-dead 1 \
  --out pretrain_data/macro_roi_fused --name macro_roi_fused
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_ensemble.py \
  --models pretrain_data/macro_v1/film_soft_logreg.joblib \
           pretrain_data/macro_view5hz/film_soft_logreg.joblib \
           pretrain_data/macro_motion/film_motion_logreg.joblib \
           pretrain_data/macro_roi_fused/film_soft_logreg.joblib \
  --weights .75 .25 0 0 --team-expert 2 --team-expert 3 \
  --params src/configs/macro_v3.json --out pretrain_data/macro_v4/ensemble.joblib --name macro_v4
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py \
  --baseline src/configs/macro_v3.json --params src/configs/macro_v4.json
```

Aggregate experiments, sensitivity checks and fingerprints are in
`bench/results/macro_v4_2026-11-15.json`. Features, model checkpoints and prediction
CSVs remain gitignored. Macro-focused experimentation continues.
