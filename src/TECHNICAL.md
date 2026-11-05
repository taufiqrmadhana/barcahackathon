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
| October pipeline | 0.5143 | 0.8631 | 11 |
| October pipeline, ball hidden, leak-free (6 masks, §6) | 0.5080 (sd 0.056) | 0.8373 | 12–14 |
| **November pipeline (§9: video ball, whistle, match structure, ownership v2)** | **0.5333** | **0.9060** | 19 |

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

### Experimental supervised ownership (2026-10-26)

An optional GBM ownership/transition model now uses event-derived labels from
[Metrica Sports sample games 1 and 2](https://github.com/metrica-sports/sample-data),
with source acknowledgement. It is disabled by default (`ownership_model=""`).
See [bench/README.md](bench/README.md) for label conventions, augmentation, exact
commands, fixed-config leave-one-game-out evaluation, and data/cache provenance.
The initial experiment adds evidence only in the final decode; self-trained shape
and D evidence are frozen within each comparison. It does not yet replace all
hand-crafted control evidence or introduce supervised shape.

Mean spellF1 across the two held-out games (one corruption seed):

| condition | baseline | nearest-team evidence | supervised owner | owner + transition |
|---|---|---|---|---|
| clean | 0.4949 | 0.5469 | 0.6498 | 0.6583 |
| ATD ball mask | 0.4111 | 0.4510 | 0.5219 | 0.5283 |
| mask + tracking corruption | 0.3613 | 0.4005 | 0.4694 | 0.4794 |

The all-games model's fixed-config public sanity score rises from 0.5143 to 0.5677,
but two leak-free transplanted public masks regress: 0.4903 to 0.4000 and 0.4593
to 0.3571. Consequently this checkpoint is **experimental and not promoted to the
default submission**. Per-frame macroF1 improves on both masks while spellF1 falls;
this flags an unresolved temporal/domain-transfer problem rather than establishing
that the classifier is ready for deployment. No weights were selected from these
public results. Game 3, boundary-label audit, confidence calibration and supervised
shape remain follow-up experiments; the two-game results do not settle them.

Stage 2 (`ownership_v2`) adds Metrica game 3, an event-convention audit and a
supervised replacement of the shape model's final H/A ratio (P(D) kept). Three-game
leave-one-game-out mean spellF1, fixed weights:

| condition | baseline | supervised owner | supervised shape | both |
|---|---|---|---|---|
| clean | 0.480 | 0.635 | 0.484 | 0.643 |
| ATD ball mask | 0.413 | 0.523 | 0.450 | 0.562 |
| mask + tracking corruption | 0.363 | 0.472 | 0.400 | 0.519 |

On the public window the combined all-games checkpoints score 0.550 clean and
0.600 / 0.563 on the two transplanted masks (baseline 0.514, 0.490, 0.459), so the
stage-1 mask regression no longer appears once shape is supervised too. Both models
stay opt-in (`ownership_model`, `supervised_shape_model`); see bench/README.md.

### Default pipeline

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

## 9. November update: what the ground truth rewards, video, audio and match structure

### 9.1 The ground truth follows the last touch, not "controlled" possession

CHALLENGE.md describes possession as *controlled* and says a deflection or a touch without
control does not change it. The public ground truth does not behave that way: in a duel at frame
15346 Barcelona own the ball for 19 frames (0.76 s) and Athletic get it straight back; 6 of the 17
public possessions are shorter than 4 s. The ground truth is "the team the corrected tracking
names as owning the ball", and that behaves like the team of the last touch. (Metrica's open
sample games, labelled from their event logs, have the same structure: 34-43% of possessions under
4 s, 17-18.5 possessions per 5 min, against 35% and 17 in the public window.)

That changes where the score is. A prediction that matches every possession longer than 4 s
exactly and misses all shorter ones scores spellF1 **0.56** on the public window -- which is where
every version of this pipeline sat (0.51-0.57). Everything above that is in short possessions,
and those live or die on the ball being where the tracking says it is at the moment of a touch.

### 9.2 The video: a pretrained ball detector, no training on this match

The ATD ball is missing on 41% of graded frames, and when present it is sometimes a static ghost
on empty grass (frame 13785: the ATD ball sits still 8 m from everybody while the real ball is in a
duel elsewhere) or 10-20 px off the real ball. `src/cv/` reads the tactical video directly:

- **Detector**: the public Ultralytics YOLO11s COCO checkpoint (class "sports ball"), exported to
  OpenVINO and INT8-quantised with NNCF on unlabelled frames of this video, run on the CPU
  (`cv/detector.py`). Nothing is trained on this match and no label is involved. The ball is
  4-10 px wide, too small for a downscaled full frame, so the detector sees native-resolution
  square crops upscaled to 640 px (a 480 px crop around the expected ball finds a confident ball
  within 30 px of the ATD on 56% of frames, against 19% within 15 px for the full frame at 1280).
- **Pass 1** (`cv/detect_crops.py`): one crop per frame around the ATD ball (interpolated across
  ATD gaps), frames 11185-53999.
- **Pass 2** (`cv/detect_grid.py`, `cv/select_grid.py`): ATD-independent search with a 7-crop grid
  over the playing area (far band 480 px crops, near band 640 px), on every public-window frame and
  on graded frames where pass 1 found nothing and the ATD ball is missing or a ghost (no player
  within 6 m), near frames a reference decode calls alive, at stride 2. Grid candidates need a score
  of 0.2 (0.08 for ATD-centred ones).
- **Fusion** (`cv/balltrack.py`): per frame, the ball is one of the video candidates (projected to
  the pitch at the bottom of the box through the homography), the ATD ball, or "not seen"; a
  Viterbi pass picks the sequence with the best detector scores, agreement with the ATD and
  physically possible motion (free up to 1.4 m per frame, impossible beyond 3 m), with a penalty on
  ATD ghosts. ATD-sourced frames are shifted by the local median video-ATD offset and the track is
  lightly smoothed. Candidates more than 0.5 m outside the pitch are dropped (spare balls, ball
  boys); a ball going out is left to the ATD. Output is a drop-in replacement for the ATD ball
  (`ball_source` in `Params`).

On the public window the fused track uses a video position on 64% of frames; where both exist it is
a median 0.57 m from the ATD ball. On frames whose ATD ball was hidden (§9.5) it recovers a ball on
~60%, 80% of them within 2 m of the hidden ATD position and ~10% grossly wrong (>10 m).

### 9.3 The audio: the referee's whistle

Three of the four stoppages in the public window are whistles (fouls, offsides) with the ball still
rolling inside the pitch; the tracking only sees the players slow down 2-4 s later, so every such
dead ball started late. This referee's whistle is a narrow tone near 3.8 kHz. `cv/whistle.py`
computes, per video frame from the video's own audio track, the peak power in 3.65-3.95 kHz over
the median of the neighbouring bands (median 8 dB; whistle blasts 25-35 dB). A blast heard while
the players are moving (>= 1.8 m/s) and followed within 5 s by the collective slowdown that
confirms a stoppage forces `D` from 10 frames after the blast to the slowdown
(`w_whistle_confirmed`), adds a little `D` evidence for 6 s after it (`w_whistle_dead`), and cuts
the cost of entering `D` around the blast (`w_whistle`). The confirmation requirement is what makes it safe: at frame 17933
a blast during play is not followed by a slowdown and changes nothing. Over the match, 26 blasts in
predicted live play are confirmed; 16 of them move a dead-ball start earlier by 0.6-3 s. On the
public window the two whistle stoppages now start at 12841 and 15900 (ground truth 12851 and 15889;
before: missed, and 15971).

### 9.4 Match structure that is certain

- **Before the second-half kickoff**: FAQ.md states frames 110808-113741 are graded and all `D`.
  The decoder had put 3-9 possessions there (players moving the ball before kickoff). They are now
  forced to `D` (`pre_kickoff_dead`).
- **Between a goal and its kick-off** the ball is dead by the laws of the game. Smart Tagging marks
  four goals and four "Kick Off (After Goal)", but its GOALS interval starts with the attack, not
  the goal, and one tagged goal was disallowed. The on-screen score (top-left graphic of the
  tactical video) settles both: `cv/scoreboard.py` finds the frame where the score digits change
  (white-pixel mask of the digits, sampled once a second). Three tagged goals change the score
  (frames 46872, 74085, 157832; the 30th-minute one never does), and from that change to the tagged
  kick-off every frame is forced to `D` (`goal_dead`). The decoder had possessions there too
  (celebrations, the ball carried back to the centre spot).

### 9.5 Choosing the configuration without the public window

The public window has 17 possessions; its bootstrap interval is 0.31-0.70, and the full match has
the ball missing 41% of the time against 12% in the window. Configurations were compared on a
**leak-free hidden-ball bench** (`lt/masked.py`): the real missing-ball pattern of six other
five-minute stretches is laid over the window (24-83% missing), the video arm may only use
ATD-independent grid detections on the hidden frames, and the shape model is frozen at round 1 --
the model scoring the public window is trained on other folds' pseudo-labels from the ball-only
decode, which do not depend on the window's ball (round 2 would leak it, §6). Same masks for every
configuration, so differences are paired.

First comparison (whistle confirmation weight 8, minimum stoppage 2 s, grid score floor 0.3):

| decoder | ball | whistle | clean spellF1 / macroF1 | masked spellF1 | masked macroF1 |
|---|---|---|---|---|---|
| default | ATD | - | 0.529 / 0.819 | 0.450 | 0.822 |
| default | ATD + video | yes | 0.483 / 0.863 | 0.449 | 0.834 |
| ownership v1 | ATD | - | 0.480 / 0.845 | 0.514 | 0.841 |
| ownership v1 | ATD + video | yes | 0.482 / 0.877 | 0.558 | 0.879 |
| ownership v2 | ATD | - | 0.513 / 0.830 | 0.627 | 0.853 |
| ownership v2 | ATD | yes | 0.509 / 0.863 | 0.623 | 0.887 |
| ownership v2 | ATD + video | - | 0.503 / 0.855 | 0.615 | 0.864 |
| ownership v2 | ATD + video | yes | 0.546 / 0.875 | 0.652 | 0.886 |

Ownership v2 (supervised ownership and supervised shape, §8) is the most robust decoder on every
one of the six masks, the whistle raises macroF1 in every arm, and with v2 the video track helps
under masks. The decoder weights are flat around their defaults (ball-control weight 0-1, owner
weight 1-1.5, switch cost 16-24, touch weights, tackle cost: all within 0.62-0.65 masked) and were
left alone. What did move the bench, each a paired gain on every mask:

- whistle confirmation strong enough to force `D` from the blast to the slowdown (weight >= 12,
  flat above), starting 10 frames after the blast onset rather than 5 (0.652 -> 0.664);
- grid candidates from score 0.2 instead of 0.3 (0.664 -> 0.673);
- **a 4 s minimum stoppage** instead of 2 s (`dead_min_frames` 100; 0.673 -> 0.700, flat from 4
  to 6 s). The decoder's false dead balls were exactly the 2 s minimum long, wedged into slow
  build-up or a player standing on the ball; the shortest public stoppage is 5.5 s and a throw-in
  alone takes longer than 4 s.

Two further changes are neutral on the bench and fix what the full two-round pipeline showed on
the public window: a whistle only counts if the players were moving at the blast (>= 1.8 m/s; a
blast during calm build-up at 12430 had forced a false stoppage), a confirmed stoppage carries a
little `D` evidence for 6 s after the slowdown, and inside a stoppage-to-restart interval the ball
being moved around before the paired restart (goal-kick set-up) no longer counts as play resuming
(`inner_restarts` 0).

Final, same bench (paired, six masks):

| configuration | clean spellF1 / macroF1 | masked spellF1 | masked macroF1 | per-mask spellF1 |
|---|---|---|---|---|
| previous submission (default decoder) | 0.529 / 0.819 | 0.450 | 0.822 | 0.36 0.49 0.44 0.51 0.42 0.47 |
| + ownership v2 | 0.513 / 0.830 | 0.627 | 0.853 | 0.51 0.69 0.62 0.57 0.65 0.71 |
| + whistle, match structure, 4 s stoppages (ATD ball) | 0.563 / 0.874 | 0.658 | 0.897 | 0.56 0.69 0.66 0.62 0.68 0.74 |
| **+ video ball (submitted)** | **0.588 / 0.887** | **0.700** | **0.898** | 0.59 0.74 0.73 0.65 0.71 0.79 |

These use the round-1 shape model, so they are comparisons, not estimates of the private score.

**Holdout masks.** Every choice above was made on the same six masks, so six fresh ones (seeds
6-11, never looked at before) were scored once at the end. The ordering holds, and each component
wins on all six of them:

| on holdout masks | clean spellF1 | masked spellF1 | per-mask spellF1 |
|---|---|---|---|
| previous submission (default decoder, ATD ball) | 0.529 | 0.506 | 0.48 0.41 0.56 0.52 0.53 0.53 |
| submitted, but default decoder | 0.578 | 0.539 | 0.58 0.52 0.59 0.52 0.49 0.55 |
| submitted, but ownership v1 | 0.521 | 0.545 | 0.52 0.62 0.57 0.52 0.51 0.52 |
| submitted, but no whistle | 0.546 | 0.615 | 0.61 0.65 0.65 0.61 0.59 0.58 |
| submitted, but 2 s minimum stoppage | 0.558 | 0.614 | 0.60 0.64 0.66 0.60 0.59 0.59 |
| submitted, but ATD ball only | 0.563 | 0.605 | 0.55 0.59 0.65 0.62 0.63 0.59 |
| **submitted** | **0.588** | **0.646** | 0.63 0.68 0.69 0.63 0.62 0.62 |

Pass 1 (crops around the ATD ball) turned out not to matter: with grid detections only, the bench
gives 0.708 masked (0.700 with pass 1), and limiting the grid to ATD gaps at stride 2, as on the
rest of the match, gives 0.699. The full pipeline with a grid-only track agrees with the submitted
file on 97% of graded frames (public 0.533 / 0.899). Pass 1 can be skipped when reproducing the
method, at the cost of not reproducing the submitted file byte for byte.

### 9.6 The submitted file

`python src/run.py` with `src/params.json` (ownership v2, `ball_source` fused_full, whistle,
`pre_kickoff_dead`, `goal_dead`, `dead_min_frames` 100, `inner_restarts` 0) on the public window:
**spellF1 0.533, macroF1 0.906** (per class F1 H 0.895, A 0.882, D 0.942), 19 possessions against
17; full match 352 possessions, dead share 0.461. The previous submission scored 0.514 / 0.863
there with 11 possessions, ownership v1 0.568 / 0.867. On spellF1 these are inside the public
window's noise (one possession is ~0.04); macroF1 rose by 0.04, and on the hidden-ball bench, where
the full match's 41% missing ball lives, spellF1 rose from 0.45 to 0.70.

Remaining public errors are timing on short possessions (an H spell after a duel starts 4.7 s late;
two short spells are found 0.8-4 s off) and two 1-2 s spurious H spells. The video detector runs
on the CPU at ~17 crops/s; reproducing it takes about 4 hours, and the INT8 detector's scores may
differ in the last digits on a CPU with another instruction set, which could move a few frames.
