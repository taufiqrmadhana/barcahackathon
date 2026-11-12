# Contact refinement, November 12, 2026

This records the contact-only stage at f09da85. Its default is now preserved as
`configs/contact_default.json`. The current default includes the Valdi merge described in
[VALDI_INTEGRATION.md](VALDI_INTEGRATION.md).

The default now adds Neo's contact-ownership idea to Taufiq's video, audio, external ownership
and shape pipeline. No competition labels are loaded at inference and no frame is hand-corrected.

| Full production pipeline, public five minutes | spellF1 | macroF1 | possessions |
|---|---:|---:|---:|
| Before this change, `configs/ownership_v2.json` | 0.5333 | 0.9060 | 19 |
| Contact-only default, `configs/contact_default.json` | **0.6703** | **0.9137** | 20 |
| Public-score candidate, `configs/contact_public.json` | **0.6947** | **0.9143** | 21 |

There are 17 GT possessions. The default's improvement is +0.1369 spellF1 and +0.0078 macroF1.
The default is preferred over the public-score candidate because its external masked/corrupted
regressions are smaller. Both configurations retain every D frame of the previous pipeline.
Over the full match the default produces 369 possessions, the public-score candidate 413,
and the previous pipeline 352. All have a graded dead-ball share of 0.4613.

## Branches inspected

- [`neo/loop-pipeline`, d61c78f](https://github.com/taufiqrmadhana/barcahackathon/tree/d61c78f/src_s2):
  contact runs, owner persistence through flights, a midpoint ownership handover, and short-spell
  cleanup were adapted into `possession/contact.py`. Its independent D detector was not imported:
  the existing whistle and match-structure signals are stronger on our public check.
- [`valdi`, 6511c13](https://github.com/taufiqrmadhana/barcahackathon/tree/6511c13/src):
  inspected camera quality, dead-ball memory, FILM features and grouped public-label validation.
  Existing Taufiq code already implements the main ball-out/restart memory. FILM was not integrated
  in this experiment; no measured improvement from it is claimed.
- `kevinza` is identical to `main` at ad81d6d. `neo/report` contains report scaffolding.

All writes and the push stay on `taufiq`.

## Method

1. Keep the original ATD as an independent ownership input. Giving the contact detector the
   video-fused track instead introduced too many contacts in the experiment.
2. Separately inside each half, remove positions more than 3 m from a five-observation median,
   and fill gaps of at most 10 frames. Detect a nearby kick or a player carrying the ball with
   a clear nearest-opponent margin. Merge contact holes of at most 3 frames; retain runs of
   at least 8 frames or a kick. These constants come from Neo's S4 implementation.
3. Hold ownership between consecutive contacts; when teams differ, switch halfway between the
   previous release and the next contact. Abstain when no ball observation is within 125 frames.
   Median padding, velocity differences and contact runs never cross half boundaries.
4. Add a weight-1 contact vote to both control/loose states for that team. Recenter the alive
   emissions, then decode again with the first decode's D mask locked. The default reduces the
   switch cost from 16 to 2 **only toward the team supported by the contact vote**. Unsupported
   frames retain the original transition costs. The public-score candidate instead uses a global
   second-decode switch cost of 4.
5. Merge H/A islands shorter than 12 frames only when surrounded by the other team inside one
   half. Never merge into or across D. This retains possession spells at restart boundaries.

Refinement happens after the final decode and never changes shape pseudo-labels or training.
The new sparse Numba Viterbi traverses legal transitions only. It matches the original dense
decoder on 80 random/tied path cases, several D durations, and the complete previous CSV.
The NumPy reference remains available as a fallback when Numba is absent.

## Validation and selection limits

This was an exploratory search on the public window: contact source, flight fraction, vote
weights, transition costs, missing-ball support, emission recentering and short-island durations
were compared. Failed directions included contact votes that unintentionally boost alive evidence,
replacing ownership outright, and locking every long missing-ball gap to the baseline.
These scores have selection bias. They are not estimates of the private full-match score.

The paired missing-ball check recomputes **round-one** shape from the baseline configuration.
The public fold is trained on other folds' ball-only pseudo-labels. Both original ATD and fused
tracks receive the same masks. Hidden frames cannot use ATD-centred video crops. This differs from
the older bench's filename-only cached shape, so its absolute scores should not be compared with
the old section 9 table. Mask seeds 6–11 were historical holdouts, but were inspected in this
experiment and are no longer untouched validation data.

| Six masked public runs, mean | spellF1 | macroF1 |
|---|---:|---:|
| Baseline, round-one shape | 0.5806 | 0.9098 |
| Default contact refinement | **0.6417** | 0.9069 |
| Public-score candidate | **0.6764** | 0.9075 |

The clean round-one scores are 0.5333 / 0.9031, 0.6703 / 0.9108 and 0.6947 / 0.9114,
respectively. Production uses two shape rounds, hence the different macroF1 in the first table.

Three external Metrica games use existing leave-one-game-out ownership/shape checkpoints;
the held-out game's labels never train those models. Ball gaps and tracking corruption match
the previous external benchmark. These event-derived labels use a different convention from
the competition's corrected tracking, and there is one corruption seed.

| External mean spellF1 | Baseline | Default | Public-score candidate |
|---|---:|---:|---:|
| Clean | 0.6432 | 0.6466 | 0.6394 |
| ATD ball masking | 0.5623 | 0.5544 | 0.5451 |
| Mask + tracking corruption | 0.5188 | 0.5141 | 0.4998 |

The gains are not universal. MacroF1 changes on these external conditions are within 0.002.
For public spellF1, 1,000 segment-bootstrap draws give 5–95% intervals of 0.378–0.678 (baseline),
0.530–0.789 (default), and 0.568–0.811 (public-score candidate). These overlap and do not correct
for parameter selection. All six masks still reuse the same 17 GT possessions.

## Reproduce

Use the data, video preprocessing and local checkpoints described in `README.md` and
`bench/README.md`. No match assets, predictions or checkpoints are committed.

```bash
uv pip install --python .venv/bin/python -r requirements-src.txt
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py \
  --params src/configs/contact_public.json --name contact_public
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
.venv/bin/python evaluation/run_evaluation.py --no-detail
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python -m unittest discover \
  -s src/bench -p 'test_*.py' -v
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/contact_bench.py \
  --suite public --variants baseline contact_default contact_public \
  --out cache/contact_public_check.json
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/contact_bench.py \
  --suite external --variants baseline contact_default contact_public \
  --out cache/contact_external_check.json
```

Missing public mask files are generated with `lt.masked.Bench` from local detections; model and
video dependencies must exist first. External evaluation needs the fold checkpoints produced by
`shape_bench.py`. Its new `--params` default is `configs/external_training.json`, frozen from the
original training manifest, with competition-only goal/kickoff rules disabled. Deployment params
can now change without breaking training. Use a fresh output directory if a manifest already
exists. `ownership_bench.py` uses the same explicit training configuration.

The official scorer validates all 199,319 rows. Whole-match production, frozen-shape tuning and
repeated CSV generation were compared exactly; contact refinement preserves D throughout the
match. Aggregates and checkpoint fingerprints are in `bench/results/contact_2026-11-12.json`.
