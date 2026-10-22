# Pretrained shape model (LSTM)

Replaces, or mixes with, the self-trained gradient-boosting shape model for frames where the ball is
not tracked. It is trained on **other matches** with real possession labels, so it needs no
pseudo-labels and no cross-fit on the competition match.

```
external match ─► data.py ─► augment.py ─► features.py ─► train.py ─► shape_lstm.pt
 (kloppy)         25 fps      look like      same 175       BiLSTM,        │
 H/A/D labels     metres      the ATD        shape feats    5 Hz           ▼
                                                            run.py --set pretrained_shape=...
```

* **Features are the pipeline's own** (`possession.shape.features`, ball-free), computed after the
  same cleaning, so the network reads the ATD features unchanged.
* **Labels** are the provider's frame flags (`ball_state`, `ball_owning_team`): DEAD -> D, ALIVE
  with an owner -> H or A. Note this is the provider's notion of possession, not the competition's
  "controlled possession": expect some label noise at loose balls.
* **Augmentation** (`augment.py`): jitter, dropouts of 1-25 s, a person tracked twice, home/away
  swap, pitch mirroring. The rates are assumptions; compare `tracked/frame` printed by
  `features.py` with the ATD's 20.5.

## Run (on the machine that will train)

```bash
uv venv --python 3.11 .venv-pretrain
uv pip install --python .venv-pretrain -r requirements-pretrain.txt   # pick the torch build for your GPU

.venv-pretrain/bin/python src/pretrain/data.py sportec                # 7 open DFL/IDSSE matches, CC BY 4.0
                                                                      # ~3.5 min and ~4 GB RAM per match
.venv-pretrain/bin/python src/pretrain/train.py --val sportec_J03WPY  # hold one match out, see if it learns
.venv-pretrain/bin/python src/pretrain/train.py --val none --out pretrain_data/shape_lstm.pt   # final fit

.venv-pretrain/bin/python src/run.py --name lstm --set pretrained_shape=pretrain_data/shape_lstm.pt
.venv-pretrain/bin/python src/run.py --name mix  --set pretrained_shape=pretrain_data/shape_lstm.pt pretrained_mix=0.5
```

Other providers: any kloppy `TrackingDataset` that fills `ball_state` and `ball_owning_team` goes
through `data.from_kloppy` + `data.resample` + `data.save(match, name)`; the rest is unchanged.
Providers with events only (Metrica sample) need a labeller that is not written yet.

## Reading the result

`train.py` reports per-frame macroF1 / team accuracy / D-F1 on the held-out match, clean and
corrupted. That tells you the network learned shape -> possession on external data. The question
that matters is whether `run.py` improves on the public window (spellF1 0.5143 with the default),
and, as in `TECHNICAL.md` §6, that is only a noisy guide: differences below ~0.04 are noise, and
the hidden-ball runs of `src/robustness.py` are the fairer test.

`pretrained_shape` is empty by default, so nothing changes unless you set it.

## Rules

Training data and checkpoints stay in `pretrain_data/` (gitignored). The competition data is never
read by anything in this folder; only the pipeline reads it, locally, when you run `run.py`.
The technical document must list the external datasets, their licences and this model.
