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
   by a model that saw it. It is then refitted once on its own improved decode (two rounds). On
   the public window, where its labels never came from, it gets the team right on 87.6% of alive
   frames from the players alone, and gives P(D) 0.80 on dead frames vs 0.09 on alive ones.

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
ball positions exactly where the decoder has nothing to check them against. Without a GPU, and
without looking at the video ourselves (match data stays local), a pretrained detector was out
of reach for this submission; the shape model covers the missing-ball frames instead.

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
| baseline_smooth125 | 0.4242 | 0.5812 | 16 |
| previous version of this repo (switch-penalised Viterbi) | 0.4800 | 0.7112 | 18 |
| **this pipeline** | **0.5143** | **0.8631** | 11 |
| this pipeline, ball hidden on 41% of the window (5 seeds) | 0.5076 | 0.8512 | 10–13 |

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
aggregates. A small number of decoder weights were chosen on the public window, with two guards
against the trap EVALUATION.md describes: (1) each config is also scored with the ball hidden on
random 3–40 s gaps covering 41% of the window (`src/robustness.py`), which is what the rest of the
match looks like, and ranked on the mean of both; (2) full-match plausibility (possession count,
dead share, dead stretches) is printed next to every row (`src/tune.py`). The landscape around the
chosen point is flat: switch costs 12–25 and dead costs 20–30 all land within 0.51–0.53.

## 6. Limitations, and what we would do with more time

- **Short possessions** (< 3 s) are often missed: the tracking near the ball sometimes names the
  wrong team (nearest player within 1 m agrees with the ground-truth team only 91% of the time),
  and the switch cost that protects long possessions from noise also swallows brief ones.
- **Whistle stoppages** without the ball leaving the pitch (fouls, offsides) are found from the
  collective slowdown, which comes late and also happens in calm open play.
- **Restarts the tracker does not see** (ball lost during the stoppage) are timed by the ball
  reappearing, which can be late.
- **Ball in flight**: there is no height, so a high ball passing over a player can still look
  close for a frame.
- With more time: a pretrained ball detector on a GPU, fine-tuned on a small set of verified
  frames; jersey-colour team checks for the slot nearest to the ball; duration distributions per
  restart type learned from the decode; per-player identity stitching to recover pass chains.

## 7. Reproduction

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
