# src_s2: contact-ownership pipeline (work in progress)

An independent alternative to `src/`. It emulates what the ground truth is built from: the
corrected tracking's ball *owner* and *dead-ball* flag (CHALLENGE.md, "What is already settled").

```
ATD ──► clean ball ──► contact owner per frame ──► dead ball ──► cleanup ──► labels
                       (kick / carry near a player;   (ball out until restart kick,
                        owner flips mid-flight         shape model when ball missing)
                        between teams)
                                     ▲
           team-shape model (players only, cross-fitted, pseudo-labels from ball-seen frames)
```

## Run

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements-src.txt
# match data in data/tactical/ (or MTAH_DATA_DIR)
.venv/bin/python src_s2/run.py --params src_s2/params_s4.json --name s4   # -> evaluation/results/s4.csv
.venv/bin/python src_s2/hidden.py --params src_s2/params_s4.json         # leak-free hidden-ball test
```

First run parses the ATD and fits the shape model (a few minutes); later runs take ~15 s.
Set `BARCA_CACHE_DIR` to choose the cache folder. Cache keys include the source of `s2.py` and
`shape.py`, so a code change always refits.

## Scores (public 5 minutes, 17 possessions)

| | spellF1 [bootstrap 5–95%] | macroF1 | possessions |
|---|---|---|---|
| `src/` (Taufiq, reference) | 0.5185 [0.31, 0.70] | 0.870 | 10 |
| `params.json` (S2) | 0.5946 [0.44, 0.72] | 0.841 | 20 |
| `params_s4.json` (S4, shape model) | **0.6111** [0.46, 0.74] | 0.842 | 19 |

Also: hidden-ball test ≈ 0.51 (reference ≈ 0.48); on a video-checked 2.6-minute window of the
second half it scores 0.66 vs 0.29 for the reference, which finds 4 of ~17 possessions there.

## Not finished: read before submitting

- **Not certified reproducible across machines.** Byte-identical here (Windows ARM64, Python
  3.12, sklearn 1.9.1) from an empty cache; not yet checked on x86/Linux.
- **Dead share is low on the full match: 0.36** (reference 0.46, typical football ~0.40–0.46).
  Likely missed stoppages where the ball is not tracked; being investigated.
- **Parameters were chosen on the 17 public possessions.** The gain over the reference is not
  statistically established (paired bootstrap +0.10, 90% interval −0.06…+0.27).
- Some 10-minute blocks switch possession more often than the public ground truth does, and there
  are 22 stoppages shorter than 5.5 s that the public ground truth never shows.
