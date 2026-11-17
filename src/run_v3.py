"""v3: v2 predictions validated by a FILM frame-interpolation layer.

    python src/run_v3.py batches                        # show which video batches v3 looks at
    python src/run_v3.py extract --scope public         # FILM pass on the public window (resumable)
    python src/run_v3.py extract --scope all --budget-min 120   # then the rest of the batches
    python src/run_v3.py run                            # model zoo + ablation + write barca_v3.csv
    python src/run_v3.py run --weight 0.5 --bootstrap 1000

Needs the v3 environment (TensorFlow + tensorflow-hub for FILM, OpenCV, scikit-learn, LightGBM):
    python3.11 -m venv .venv-v3 && .venv-v3/bin/pip install -r requirements-v3.txt

FILM is only needed by `extract`; `run` reads the cached features (cache/*_v3_film_features.npz),
so a submission is regenerated exactly from the cache, or re-extracted from the video.
Everything scored on the public window is out of fold (see possession/validator.py).
"""

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from possession.paths import RESULTS_DIR, SRC  # noqa: E402
from possession import batches as batch_sel  # noqa: E402
from possession import film_features, validator  # noqa: E402
from possession import pipeline_v3 as pipeline  # noqa: E402
from run_v2 import load_params, V2_PARAMS  # noqa: E402
import possession_metrics as pm  # noqa: E402


def headline(result):
    f1_d = result.get("f1_D", result.get("per_class", {}).get("D", {}).get("f1", 0.0))
    trans_f1 = result.get("trans_f1_25", result.get("transitions", {}).get(25, {}).get("f1", 0.0))
    return "spellF1 {:.4f}  macroF1 {:.4f}  D-F1 {:.3f}  transF1@25 {:.3f}  poss {:>3}".format(
        result["spell_f1_mean"], result["macro_f1"], f1_d, trans_f1,
        result["n_pred_spells"])


def cmd_batches(inputs, params, gt, args):
    mask, reasons, labels = pipeline.select_batches(inputs, params, gt)
    print(batch_sel.describe(mask, reasons, labels))
    for scope in ("public", "all"):
        s = scope_samples(mask, reasons, args.stride, scope)
        print("  {:<6} {:>6} FILM samples at stride {}".format(scope, len(s), args.stride))


def scope_samples(mask, reasons, stride, scope):
    if scope == "public":
        return batch_sel.samples(reasons["public"] & mask, stride)
    return batch_sel.samples(mask, stride)


def cmd_extract(inputs, params, gt, args):
    mask, reasons, _ = pipeline.select_batches(inputs, params, gt)
    frames = (np.unique(np.loadtxt(args.frames,dtype=np.int64,ndmin=1)) if args.frames
              else scope_samples(mask, reasons, args.stride, args.scope))
    if (frames < 2).any() or (frames >= inputs.meta.n_frames - 2).any():
        raise ValueError('sample frame and its +/-2 neighbours must be inside the video')
    if args.max_samples and len(frames) > args.max_samples:
        # Keep short sequential chunks, spread over the match. Selecting isolated
        # frames would leave no supported interval for video-feature interpolation.
        chunks = [frames[a:a + args.chunk] for a in range(0,len(frames),args.chunk)]
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(chunks), min(len(chunks),
                            max(1,args.max_samples // args.chunk)), replace=False)
        frames = np.sort(np.concatenate([chunks[j] for j in chosen]))
    store = film_features.Store(args.cache_kind or args.interp)
    todo = store.missing(frames)
    print("{} samples in scope '{}', {} cached, {} to extract".format(
        len(frames), 'frame plan' if args.frames else args.scope,
        len(frames) - len(todo), len(todo)), flush=True)
    if len(todo) == 0:
        return
    extractor = pipeline.make_extractor(inputs, params, args.interp)
    budget = args.budget_min * 60 if args.budget_min else None
    film_features.extract(extractor, store, frames, chunk=args.chunk, budget_s=budget)


def cmd_run(inputs, params, gt, args):
    store = film_features.Store(args.cache_kind or args.interp)
    layer = pipeline.Layer(inputs, params, gt, store,
                           fusion=validator.FusionConfig(weight=args.weight),
                           train_stride=args.train_stride)
    print("FILM samples: {} cached, {} in batches, {} with public labels".format(
        len(store.frames), len(layer.sample_frames), len(layer.train_frames)))
    print("validated frames: {} ({:.1%} of graded)".format(
        int(layer.mask.sum()), layer.mask.sum() / sum(l - f + 1 for _, f, l in inputs.blocks)))
    if len(layer.train_frames) < 100:
        raise SystemExit("not enough FILM samples in the public window: run `extract --scope public`")

    base = layer.evaluate(layer.labels, args.bootstrap)
    print("\nv2 (no validation)          " + headline(base))

    zoo = validator.model_zoo()
    folds = validator.contiguous_folds(layer.train_frames, args.folds)
    results = {}
    print("\nmodel selection (contiguous {}-fold CV on public samples, lower log-loss is better)"
          .format(args.folds))
    for set_name in args.sets:
        names = validator.FEATURE_SETS[set_name]
        x = validator.matrix(layer.columns, names, layer.train_frames)
        best, report = validator.select_model(x, layer.y, folds, zoo, layer.train_frames)
        print("  [{}]".format(set_name))
        for name, r in sorted(report.items(), key=lambda kv: kv[1]["log_loss"]):
            print("    {:<14} log-loss {:.3f}  frame macroF1 {:.3f}{}".format(
                name, r["log_loss"], r["macro_f1"], "   <- best" if name == best else ""))
        proba = layer.proba(names, zoo[best], args.folds, oof=True)
        labels = layer.decode_oof(names, zoo[best], args.folds)
        results[set_name] = (best, proba, labels, layer.evaluate(labels))

    print("\nout-of-fold result on the public window, fusion weight {}".format(args.weight))
    print("  {:<26} {}".format("v2", headline(base)))
    for set_name, (best, _, _, r) in results.items():
        print("  {:<26} {}".format("v3 " + set_name + " / " + best, headline(r)))

    final = args.final if args.final in results else list(results)[-1]
    best, proba, _, _ = results[final]
    print("\nfusion weight sensitivity ({} / {}):".format(final, best))
    for w in (0.25, 0.5, 1.0, 2.0):
        print("  w={:<5} {}".format(w, headline(layer.evaluate(
            layer.decode_oof(validator.FEATURE_SETS[final], zoo[best], args.folds, w)))))

    # Public CSV rows are the independently decoded held-out predictions. The
    # all-public fit is used only outside the public training window.
    labels = results[final][2]
    path = pipeline.write(os.path.join(args.out, args.name + ".csv"), labels)
    changed = (labels != layer.labels) & (layer.labels != "X")
    print("\nwrote {}  ({} / {}; {} frames changed vs v2, {:.2%} of graded)".format(
        os.path.relpath(path), final, best, int(changed.sum()),
        changed.sum() / max(1, (layer.labels != "X").sum())))
    if args.bootstrap:
        r = layer.evaluate(results[final][2], args.bootstrap)
        for name,lo,hi in [('spellF1','spell_bootstrap_low','spell_bootstrap_high'),
                           ('macroF1','bootstrap_low','bootstrap_high')]:
            print('  {}  v2 [{:.3f}, {:.3f}]  v3 OOF [{:.3f}, {:.3f}]'.format(
                name,base[lo],base[hi],r[lo],r[hi]))
    print("\nwhole match (no ground truth needed):")
    for tag, lab in (("v2", layer.labels), ("v3", labels)):
        for row in pipeline.match_summary(lab, inputs.blocks):
            print("  {} half {half}: {possessions} possessions, mean {mean_s:.1f}s, "
                  "H {share_H:.1%}  A {share_A:.1%}  D {share_D:.1%}".format(tag, **row))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("batches", "extract", "run"))
    parser.add_argument("--params", default=V2_PARAMS)
    parser.add_argument("--interp", default="film", choices=("film",))
    parser.add_argument("--cache-kind", help="separate feature cache name; keeps earlier extraction reproducible")
    parser.add_argument("--scope", default="public", choices=("public", "all"))
    parser.add_argument("--stride", type=int, default=batch_sel.BatchConfig.stride)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--budget-min", type=float, default=0)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="deterministic chunk sample spread across the selected scope")
    parser.add_argument("--frames", help="exact frame-index plan; extraction is resumable and idempotent")
    parser.add_argument("--weight", type=float, default=validator.FusionConfig.weight)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-stride", type=int, default=25,
                        help="one training sample per this many frames; avoids cache-density bias")
    parser.add_argument("--sets", nargs="+", default=list(validator.FEATURE_SETS),
                        choices=list(validator.FEATURE_SETS))
    parser.add_argument("--final", default="v2+flow+film", choices=list(validator.FEATURE_SETS))
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--name", default="barca_v3")
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args()

    started = time.time()
    inputs = pipeline.Inputs()
    params = load_params(args.params, {})
    gt = pipeline.public_gt()
    {"batches": cmd_batches, "extract": cmd_extract, "run": cmd_run}[args.command](
        inputs, params, gt, args)
    print("\ndone in {:.0f}s".format(time.time() - started))


if __name__ == "__main__":
    main()
