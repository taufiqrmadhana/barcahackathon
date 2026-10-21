# Ball possession from tracking evidence, events and a semi-Markov decoder

More than a Hack 2027, Phase 1 — technical document. Code: `src/` in this repository.

## 1. Approach

The task is not "which team is closest to the ball" but "which team is in control, over the whole
match, and when is the ball dead". So the pipeline looks for **evidence of ball control**,
**candidate events** (touches, passes, turnovers, stoppages, restarts), and joins them into
possessions with a **temporal model** that looks before and after each frame.

```
ATD ──► 1 clean ──► 2 evidence ──► 3 events ─────────────────────► transition bonuses ─┐
                        │                                                               │
                        └──► emissions (ball) ──► 4 HSMM decode #1 ──► shape model ──► emissions ──► decode #2, #3
Smart Tagging ──► set-piece starts ─────┘                               (players only, self-trained)
```

1. **Clean.** Positions in metres. Player detections more than 5 m off the pitch are dropped;
   slot velocities over ±2 frames, discarded above 12 m/s (a slot jumping to another person).
   Ball points that jump away and back faster than 45 m/s, and observed runs of under 3 frames
   isolated by 2 s gaps, are removed. Gaps are interpolated only up to 12 frames and only if the
   implied speed is possible. (Short gaps are rare: almost every missing-ball gap in this ATD is
   3 s or longer.) Slots are treated as team members with short-range motion, never as people.
   The attacking direction per half is read from where each team stands.
2. **Evidence per frame.** For each team, the nearest slot to the ball and its speed relative to
   the ball. Control needs both proximity and the ball moving with the player:
   `p = sigmoid((1.3 m − d) / 0.35) · exp(−max(0, Δv − 2.5) / 3)`. Exclusive control
   `p_H(1−p_A)`, `p_A(1−p_H)`, and the rest is "loose". A dead-ball log-odds collects: ball beyond
   a line, last seen beyond a line, collective player slowness (2 s mean speed against 1.3 m/s),
   a resting ball, a fast ball inside the pitch (against), and the shape model's P(D).
3. **Events.** Touches are peaks of ball velocity change next to a player, labelled by their
   neighbours as pass (same team, ball travelled ≥ 5 m), turnover, deflection (other team for one
   touch inside a sequence), or duel (both teams within 1.5 m). Shots are hard kicks towards the
   attacked goal. Stoppages: ball over the goal line or touchline, ball vanishing on a course out
   of the pitch, a collective slowdown (whistle), a Smart Tagging set-piece start. Restarts: a
   resting ball that is struck, a ball coming back from beyond a line (throw-in), a ball found
   again after a long gap. A reliable stoppage is paired with the first restart after a minimum
   hold (8 s after a goal-line out, 3 s after a touchline out) into a dead interval.
4. **Temporal model.** Hidden states `H_ctrl, H_loose, A_ctrl, A_loose, D_1..D_50`, Viterbi per
   half. The loose states keep the last team while the ball travels or is contested, so a pass
   costs ~nothing. Possession can change only by entering the other team's *control* state
   (16 frames of evidence; +5 if straight from the opponent's control, i.e. a tackle). Dead ball
   is a chain of 50 states, an explicit 2 s minimum duration, so a short "dead" blip cannot split
   a possession. Leaving D goes straight into a control state, because every restart has a taker.
   Events lower these costs on the frames where they happen (touch → switch to the touching
   team; stoppage → into D; restart → out of D towards the restarting team); dead intervals add
   strong D evidence, the 2 s after a restart add alive evidence.
5. **Missing ball: the shape model.** 41% of graded frames have no ball, mostly in long gaps, so
   the decision there must come from the players. A gradient-boosted classifier (scikit-learn,
   175 features) reads both teams' shape and motion in their attacking direction — how advanced
   and how spread each team is, its forward velocity, players in the opponent half, marking
   distances, player speed — plus 1 s and 3 s smoothed versions and their changes. It is trained
   on frames where the ball is tracked, with labels from the ball-based decode, **never from the
   ground truth**, and cross-fitted over 8 time blocks with a 10 s guard, so no frame is scored
   by a model that saw it. It is then refitted once on its own improved decode (two rounds; worth
   only +0.018 spellF1 in leak-free tests, see §6). On the public window, where its labels never
   came from, it gets the team right on 87.6% of alive frames from the players alone, and gives
   P(D) 0.80 on dead frames vs 0.09 on alive ones.

Output: `H`/`A`/`D` on every frame of both halves, `X` outside them.

## 2. Assets used, and dropped

| asset | use | verdict |
|---|---|---|
| Tactical ATD | everything above | core |
| Smart Tagging | set-piece starts as stoppage candidates and local D evidence | kept; 56 of 58 tagged set pieces get a D within 6 s |
| Smart Tagging phase team | per-frame team prior | **dropped**: hurt both the public score and the hidden-ball score at every weight tried (0.1, 0.3) |
| Homography | projecting the ATD into the video for the CV check | used in the experiment only |
| Homography camera aim / pan / zoom | as shape-model features (the camera follows play; its aim is ~12 m from the ball in x) | **dropped**: team accuracy against the cross-fitted labels 0.851 with vs 0.862 without |
| Tactical video, CV ball recovery | detector trained on the ATD ball projected to pixels | **dropped**, see below |
| Broadcast video and ATD | — | not used: not synchronised, homography usable on 66% |
| Public ground truth | correctness checks, error analysis, choosing a handful of params | see §5 |

**Why the CV step was dropped** (`src/experiments/cv_probe.py`, aggregates only, no frame is
written or viewed). The cheap route to a ball detector without hand labels is to use the ATD ball,
projected through the homography, as the training label. The probe confirms the axis convention
and the frame alignment (bright blob at the projected ball: 0.45 at offset 0, 0.15–0.35 at ±2–8
frames, 0.03–0.05 with flipped axes), but more than half of projected ATD balls sit on no visible
bright blob, a fast ball shows *less* camera-compensated motion at its projected point than a
slow one (median 4 vs 36 — in flight, the ATD ball is an estimate, not the pixel detection),
blob proposals cover the ball in 23% of frames, and a patch classifier reaches AUC 0.65–0.72
with the true patch first in ~30% of frames. A detector built on these labels would inject wrong
ball positions exactly where the decoder has nothing to check them against, so it is not in the
pipeline; the shape model covers the missing-ball frames instead. What this shows is that *the
ATD ball is a poor free label*, not that the video carries no usable information: a pretrained
detector (GPU) or a few verified frames could change the picture. `src/experiments/overlay.py`
draws the projected tracking and the proposals onto frames, locally, for a person to inspect.

## 3. Possession rules implemented

- **Control** = a player within ~1.3 m of the ball *and* moving with it. A ball passing a player
  at speed is not controlled by him.
- **Loose ball, aerial duel, deflection**: possession stays with the last team (loose states).
  A deflection is one touch of the other team inside a sequence of the same team.
- **Change of possession** only through the other team's control, with enough accumulated
  evidence (≈16 frames), cheaper at that team's touches. A contested ball does not flip it.
- **Dead ball** from the moment the ball crosses a line (or play freezes) to the restart:
  struck placed ball, throw-in, or the ball reappearing in play. At least 2 s; goal-line outs at
  least 8 s before a restart is accepted. While dead, a player carrying or placing the ball does
  not make it alive. After the restart, possession belongs to the team that takes it.
- **Dead-ball ownership** is not inferred (the answer is `D`).
- **Abstention**: `X` only outside the two halves. Inside play we always guess, because the
  scorer would otherwise carry our previous label across a hand-over.

## 4. Results

Public window (5 min, 17 possessions; `python evaluation/run_evaluation.py`):

| | spellF1 | macroF1 | possessions |
|---|---|---|---|
| baseline_smooth50 | 0.3256 | 0.6207 | 26 |
| baseline_smooth125 | 0.4242 | 0.5812 | 16 |
| previous version of this repo (switch-penalised Viterbi) | 0.4800 | 0.7112 | 18 |
| **this pipeline** | **0.5143** | **0.8631** | 11 |
| this pipeline, ball hidden, leak-free (6 masks, §6) | 0.5080 (sd 0.056) | 0.8373 | 12–14 |

All rows: same scorer (upstream `run_evaluation.py`, unchanged), same window (frames
11185–18684), spellF1 = mean F1 over IoU 0.3–0.7 on H/A spells. The baselines were regenerated
from the data with `starter/baseline.py` and are byte-identical to the shipped files. **The
published floor of 0.2433 is a different number**: baseline_smooth50 on the *full* match
(private ground truth). The same baselines lose a lot from public to full (smooth50 0.326 → 0.243,
smooth125 0.424 → 0.224), so 0.514 on the public window says nothing direct about the full match.

Bootstrap 5–95% for spellF1 on 17 possessions is 0.31–0.70: the public window cannot rank
configs finely. Per class F1 H 0.880, A 0.864, D 0.846; alive/dead F1 0.962; team accuracy on
alive frames 0.872. At IoU 0.5: precision 0.73, recall 0.47 — the dominant error is
**under-segmentation** (short possessions missed), not spurious ones.

Full match, no ground truth (`python src/analyze.py`): 213 possessions, mean 16.1 s; dead share
0.459 in 152 stretches; 23 possessions decided with no tracked ball at all; agreement with the
Smart Tagging phase team 0.839 of tagged alive frames.

## 5. How parameters were chosen

Every number is in `src/possession/config.py`, the submitted values in `src/params.json`. Most
are physical (radii, speeds, durations) and were set from the football meaning and checked on
aggregates. A small number of decoder weights were chosen on the public window, screened with
`src/tune.py` on the mean of the clean score and a hidden-ball score, with full-match plausibility
(possession count, dead share, dead stretches) printed next to every row. The landscape around the
chosen point is flat: switch costs 12–25 and dead costs 20–30 all land within 0.51–0.53.

Two caveats, found in the audit below: that hidden-ball screen reused shape probabilities from
the unmasked run, which leaks (§6), and every public-window number here was also used to choose
something, so all of them carry selection bias. None of them is an estimate of the private score.

## 6. Validation audit

**Leakage.** The first hidden-ball test hid the ball on the public window but reused the shape
model's probabilities from the run where the ball was visible. In round 1 that is harmless: the
model scoring fold 0 (which holds the whole public window) is trained only on the other folds,
whose labels do not change when the window is masked. In round 2 it is not: the fold-0 model
learns from labels of other folds that were produced by models trained partly on the public
window *with* its ball. `src/robustness.py` now hides the ball before everything — cleaning,
evidence, events, pseudo-labels, every shape fit and every self-training round. On identical
masks the old shortcut inflated round-2 spellF1 by +0.024 and macroF1 by +0.013; for round 1 the
two agree exactly, which confirms where the leak was.

**Cross-fit.** Folds are 8 contiguous 13-minute blocks with a guard on both sides, never random
frames. Widening the guard from 10 s to 60 s and 120 s changes the shape model's public-window
team accuracy from 0.858 to 0.864 and 0.867: adjacent frames are not inflating it.

**Do the simulated gaps look like the real ones?** Real gaps outside the window: 129, median
11.7 s, p90 39.7 s, max 221 s. Inside the window, 56% of the frames where the ball is already
missing are dead ball. A hazard model fitted outside the window on the 8 frames before each real
gap finds them more likely near the lines (odds ratio 0.57 per sd of distance), in crowds (1.20)
and with slow players (0.77 per sd of speed); ball speed and duels do not predict them. Three gap
models are reported, all leak-free: *empirical* (real lengths, anywhere), *context* (real lengths,
placed by that hazard) and *transplant* (the real missing-ball mask of another five minutes, laid
over the window: 24–83% missing). Hidden frames are 0–38% dead ball against 56% for the natural
gaps, so even *context* under-represents stoppages. This measures robustness to simulated gaps;
it is not an estimate of the private score.

**Self-training.** Same masks for every variant (clean window + 3 gap models × 2 seeds):

| shape model | spellF1 clean | spellF1 hidden | paired Δ spellF1 vs submitted | full-match possessions |
|---|---|---|---|---|
| 1 round, all pseudo-labels | 0.486 | 0.492 | −0.018 (−0.065 … +0.015) | 217 |
| **2 rounds, all pseudo-labels (submitted)** | **0.514** | **0.508** | — | 213 |
| 1 round, confident only (margin ≥ 20) | 0.497 | 0.484 | −0.023 (−0.117 … +0.046) | 223 |
| 2 rounds, confident only | 0.497 | 0.443 | −0.059 (−0.157 … +0.031) | 219 |

"Confident" is the decoder's max-marginal margin (best path score minus the best score with a
different label on that frame); frame accuracy rises from 0.87 to 0.95 above 20, but training
only there makes the shape model worse, presumably because it loses exactly the hard moments
near transitions. The second round is a small gain, within the noise, and does not reduce the
possession count — so no sign here that self-training amplifies the under-segmentation, but no
strong case for it either. (The earlier claim of a much larger gain came from the leaky test.)

**Why each ground-truth possession fails** (`src/spells.py`). Each public spell is put in one of
five categories; misses are then decoded again with the same emissions and only a lower switch
cost, to separate "no evidence" from "evidence cancelled by the cost":

| | hit | boundary | merged | missed: no evidence | missed: in D | missed: cost |
|---|---|---|---|---|---|---|
| clean window (17 spells) | 8 | 3 | 2 | 2 | 1 | 1 |
| 6 masked runs (102 spells) | 47 | 16 | 15 | 13 | 8 | 3 |

- All six spells shorter than 4 s are lost in the clean window (0–1 per masked run).
- **The switch cost is rarely the cause** (1 of 17, 3 of 102). Lowering it globally finds more
  spells but loses more elsewhere: at cost 4 the window has 17 predicted possessions like the
  ground truth, yet spellF1 drops from 0.514 to 0.494; at 0.5 it is 0.421.
- **"No evidence"** spells really have none in the tracking: no frame of control for that team,
  no touch by it, and the other team is the nearer one on most frames. Decoder changes cannot
  recover these; better data near the ball can (CV, team checks of the nearest slot).
- **"In D"**: a short possession just before a stoppage swallowed by an early dead ball.
- **Boundary** and **merged** spells are found but cut wrong: starts/ends off by 0.8–5 s, or two
  same-team possessions glued across a missed opponent spell or a missed stoppage.
- One clean-window miss (13704, A, 3.2 s) is the opposite: its own team is the nearest on every
  frame and makes the only two touches, yet no frame passes the control definition (proximity
  *and* moving with the ball). This is the kind of case a touch-based control signal would target.

So the changes worth testing next are local, not global: control evidence from touches and
receptions, boundary timing at the touch that starts a possession, and the start of dead
intervals. Each should be judged on the hidden-ball runs per category, not on the clean score.

## 7. Limitations, and what we would do with more time

- **Short possessions** (< 4 s) are almost all missed. The audit (§6) splits why: mostly no
  support in the tracking at all, sometimes an early dead ball, rarely the switch cost.
- **Whistle stoppages** without the ball leaving the pitch (fouls, offsides) are found from the
  collective slowdown, which comes late and also happens in calm open play.
- **Restarts the tracker does not see** (ball lost during the stoppage) are timed by the ball
  reappearing, which can be late.
- **Ball in flight**: there is no height, so a high ball passing over a player can still look
  close for a frame.
- With more time: a pretrained ball detector on a GPU, fine-tuned on a small set of verified
  frames; jersey-colour team checks for the slot nearest to the ball; duration distributions per
  restart type learned from the decode; per-player identity stitching to recover pass chains.

## 8. Reproduction

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements-src.txt
# unpack the data bundle into data/ (or set MTAH_DATA_DIR)
.venv/bin/python src/run.py                     # -> evaluation/results/barca.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
```

Deterministic: two runs from an empty cache produce byte-identical files (gradient boosting with
a fixed seed and no early stopping). About three minutes on 12 CPU cores. All
processing is local; nothing is sent to an external service.

The audit: `python src/robustness.py --sim transplant context empirical --seeds 2` (about 30
minutes) and `python src/spells.py`, optionally with `--hide context --seeds 2`.
