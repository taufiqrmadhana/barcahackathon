# src_s2: contact-ownership pipeline

An independent alternative to `src/`. It emulates what the ground truth is built from: the
corrected tracking's ball *owner* and *dead-ball* flag (CHALLENGE.md, "What is already settled").

```
ATD ──► clean ball ──► contact owner per frame ──► dead ball ──► cleanup ──► labels
        (noise-matched   (kick / carry near a player;  (ball out until restart kick, dead
         kick span)       owner flips mid-flight;       memory, slow-player stoppages;
                          regain after a ball-less gap)  shape model when ball missing)
                                     ▲
           team-shape model (players only, cross-fitted, pseudo-labels from ball-seen frames)
```

## Run

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements-src.txt
# match data in data/tactical/ (or MTAH_DATA_DIR)
.venv/bin/python src_s2/run.py --params src_s2/params_s8.json --name s8   # -> evaluation/results/s8.csv
.venv/bin/python src_s2/hidden.py --params src_s2/params_s8.json         # leak-free hidden-ball test
.venv/bin/python src_s2/stress.py ...                                     # ball-noise stress test
```

First run parses the ATD and fits the shape model (a few minutes); later runs take ~15 s.
Set `BARCA_CACHE_DIR` to choose the cache folder. Cache keys include the source, so a code change
always refits. From an empty cache, two runs give the same file: sha256 `9a019a5d…a59aadc`
(Windows ARM64, Python 3.12, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1).

Every component is switchable from the params file (`use_*` keys); `params_s4.json` and
`params_s7b.json` reproduce the earlier versions.

## Scores

Public = the 5-minute public GT (17 possessions). Video = two windows we labelled from the video
(validation only, never training; labellers agree with each other at spellF1 ~0.72, so these are
noisy). Lockbox = a third video-labelled window kept unseen until the final pick.

| | public spellF1 | public macroF1 | video (40 poss) | lockbox (28 poss) | all 4 sets pooled (88) |
|---|---|---|---|---|---|
| `src/` (Taufiq, reference) | 0.519 | 0.870 | ~0.24 | 0.263 | 0.310 |
| S4 (`params_s4.json`) | 0.611 | 0.842 | 0.50 | 0.305 | 0.475 |
| S7b (`params_s7b.json`) | 0.611 | 0.832 | 0.53 | 0.312 | 0.493 |
| **S8 (`params_s8.json`)** | **0.649** | **0.853** | **0.55** | 0.263 | **0.501** |

S8 vs S7b, paired bootstrap over all 88 possessions: +0.008 [90%: −0.021, +0.037]. On the lockbox
alone S8 hits one possession fewer than S7b (6 vs 7 of 28). Every pipeline scores much lower on the
lockbox than on public: expect the hidden full-match score to be well below the public 0.65.

What changed from S4 to S8:
- **Dead ball** (S7b): dead memory (ball leaves play → dead until a fast restart), slow-player
  stoppages, shape model trained on graded frames. Full-match dead share 0.363 → 0.455.
- **Robust kicks** (S8): the kick detector's differencing span is matched to the ball's noise
  level. Public spellF1 under 0.3 m ball noise: S7b 0.27–0.31, S8 0.57.
- **Gap kick** (S8): the ball reappearing after a gap next to the other team, moving, is a regain,
  not a restart (fixes public possession #15, checked on video).
- **Pruned** 9 rules that the ablation showed had no effect (one, shape_restart, was harmful).
  Ablation tables in `experiments/`.

## Known limits

- Parameters were chosen on the public window plus the two video windows (~60 possessions).
- Still missed on public: brief touches near the touchline (#2, #10, #12, #13).
- 17 stoppages shorter than 5.5 s on the full match, which the public GT never shows.
- Hidden-ball test (real missing-ball masks): 0.51.
