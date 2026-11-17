# Training and evaluation

Run from the repository root with the inference environment active.

## Tests

```bash
python -m unittest discover -s src/bench -p 'test_*.py' -v
```

The tests cover duration constraints, contact ownership, probability fusion,
video provenance, temporal corrections and held-out-label isolation.

## Evaluation exports

```bash
python src/bench/phase_learning_curve.py --params src/configs/macro_v11.json
.venv-cv/bin/python src/bench/export_evaluation.py
```

The learning curve requires the match assets and fitted ensemble. The exporter
only reads the frozen aggregate reports. It creates tables, CSVs and plots under
`src/submission/build/evaluation/`. See [EVALUATION_BUNDLE.md](EVALUATION_BUNDLE.md).

## Training tools

- `metrica_open.py`: fetch external tracking and event samples.
- `shape_bench.py`: fit ownership and shape models.
- `macro_bench.py`: refit the ensemble under different time partitions.
- `phase_learning_curve.py`: vary phase training-set size with other members fixed.

The remaining probe scripts support feature and calibration comparisons. Outputs
belong in `cache/` or `pretrain_data/`.

Three aggregate reports are retained: the final results, its availability-only
comparison, and its learning curve. Public validation uses the same 17 possessions
throughout; repartitioning them is a sensitivity check, not an independent test.
