# <TEAM NAME>: ball possession from tracking

More than a Hack 2027, Phase 1 — technical document.

<!--
Scaffold. Every "TODO" is a decision or a number the team still owes.
Submit as <team_name>.pdf or <team_name>.md (SUBMISSION.md). A few pages is plenty.
Required sections (SUBMISSION.md): approach · assets used and dropped · possession rules ·
limitations · reproduction. Keep numbers in sync with what `report/README.md` says to run.
-->

**Repository:** TODO public URL · **Submitted file:** `<team_name>.csv` · **Pipeline:** TODO (`src_s2/` S4, `src/`, or a merge)

## 1. Approach

TODO: one paragraph — what the pipeline does end to end, and why this design rather than the
alternatives we tried.

```
TODO: pipeline diagram (inputs → evidence → decoder → labels)
```

Key idea to explain: the ground truth is derived mechanically from *corrected tracking*
(owner team / dead flag, CHALLENGE.md), so we model a provider-style **ownership** signal rather
than a human notion of "controlled possession". TODO: confirm this is the framing we submit.

### Components

| stage | what it does | where |
|---|---|---|
| cleaning | TODO | TODO |
| ownership evidence | TODO | TODO |
| dead ball | TODO | TODO |
| missing ball (41% of graded frames) | TODO | TODO |
| temporal decoding / cleanup | TODO | TODO |

## 2. Assets used, and dropped

| asset | use | verdict |
|---|---|---|
| Tactical ATD (tracking) | TODO | core |
| Smart Tagging | TODO | TODO |
| Homography | TODO | TODO |
| Tactical video | TODO (e.g. kit-colour team correction, video-labelled validation windows) | TODO |
| Broadcast set | not used (own timeline, not synchronised) | dropped |
| External data | TODO (e.g. DFL/Sportec open tracking, CC BY 4.0 — licence must be stated) | TODO |

Tried and dropped (one line each, with the reason): TODO
- e.g. global switch-cost tuning — trades correct short possessions for spurious ones ~1:1
- e.g. "control then kick" bonus — no net gain
- e.g. FILM frame-interpolation features (Valdi v3) — out-of-fold public spellF1 0.50 vs 0.497 without

## 3. Possession rules implemented

TODO: state each rule as implemented, with the parameter that controls it.

- **Change of possession:** TODO (when does the new team get the ball — first contact, carry, kick-away?)
- **Loose balls / passes in flight:** TODO
- **Challenges, deflections, aerial duels:** TODO
- **Ball out of play:** TODO (from line crossing until the restart kick)
- **Fouls / whistle stoppages:** TODO
- **Restarts:** TODO (restarting team owns from the kick)
- **Ball not tracked:** TODO
- **Outside the halves:** TODO (`X` or `D`)

## 4. Results

Public window only (5 minutes, 17 possessions). Scored with `evaluation/run_evaluation.py`.

| | spellF1 [bootstrap 5–95%] | macroF1 | possessions (GT 17) |
|---|---|---|---|
| baseline_smooth50 (organisers) | 0.3256 | 0.6207 | 26 |
| reference (`src/`, Taufiq) | 0.5185 [0.31, 0.70] | 0.8701 | 10 |
| **submitted** | TODO | TODO | TODO |

Robustness and full-match checks (no private GT used): TODO
- hidden-ball test (real missing-ball masks from other windows): TODO
- full match: possessions, mean duration, dead share, possessions per alive minute inside vs outside the public window: TODO
- video-checked validation windows (labels used for validation only, never training): TODO

State plainly: the public window has 17 possessions; differences below ~0.04 spellF1 are noise.

## 5. How parameters were chosen

TODO: which numbers are physical (radii, speeds, durations), which were tuned, on what, and the
response curves. Say how many parameters touched the public window.

## 6. Limitations

TODO
- e.g. ground-truth switch timing is inconsistent (first control vs kick-away)
- e.g. stoppages invisible in the tracking (ball still rolling when the whistle goes)
- e.g. slot identity swaps (keepers) in the raw ATD
- e.g. cross-machine float differences
- With more time: TODO

## 7. Reproduction

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements-src.txt
# unpack the data bundle into data/ (or set MTAH_DATA_DIR)
TODO: exact command that writes the submitted CSV
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

TODO: runtime, determinism statement (byte-identical from an empty cache), platform tested.
All processing is local; no match data is sent to an external service, and the labels come out of
the algorithm only.
