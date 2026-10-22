"""Train the shape LSTM on external matches and save a checkpoint the pipeline can load.

    python src/pretrain/train.py --val sportec_J03WPY            # hold one match out, report on it
    python src/pretrain/train.py --val none --out pretrain_data/shape_lstm.pt   # final: all matches

Validation reports per-frame macroF1 and team accuracy on the held-out match, clean and corrupted.
These numbers say whether the network learned possession from player shape on the external data.
They are not a competition score: run the pipeline on the public window for that.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain import data, features, net  # noqa: E402


def macro_f1(y, pred):
    keep = y >= 0
    y, pred = y[keep], pred[keep]
    f1 = []
    for c in range(3):
        tp = ((y == c) & (pred == c)).sum()
        fp = ((y != c) & (pred == c)).sum()
        fn = ((y == c) & (pred != c)).sum()
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1)), f1


def team_accuracy(y, proba):
    keep = (y == 0) | (y == 1)
    return float((proba[keep, :2].argmax(1) == y[keep]).mean())


def evaluate(predictor, halves):
    ys, ps = [], []
    for x, y in halves:
        ps.append(predictor.proba_steps(x))
        ys.append(y)
    y, p = np.concatenate(ys), np.concatenate(ps)
    f1, per = macro_f1(y, p.argmax(1))
    return {"macroF1": f1, "H": per[0], "A": per[1], "D": per[2], "team_acc": team_accuracy(y, p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matches", nargs="+")
    ap.add_argument("--val", default=None, help="match name to hold out, or 'none'")
    ap.add_argument("--versions", type=int, default=3, help="corrupted copies per match (1 = clean only)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(data.EXT_DIR, "shape_lstm.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    names = args.matches or data.stored_matches()
    val = None if args.val == "none" else args.val or (names[-1] if len(names) > 1 else None)
    train_names = [n for n in names if n != val]
    print("train on", train_names, "| validate on", val, flush=True)

    train = []
    for name in train_names:
        for path in features.build(name, args.versions, args.seed):
            train += [h for h in features.load(path) if len(h[0]) >= net.WINDOW]
    val_sets = {}
    if val:
        paths = features.build(val, min(2, args.versions) if args.versions > 1 else 1, args.seed)
        val_sets = {"clean": features.load(paths[0])}
        if len(paths) > 1:
            val_sets["corrupted"] = features.load(paths[1])
    if not train:
        raise SystemExit("no training sequences (halves must be at least {} steps)".format(net.WINDOW))

    flat = np.concatenate([x for x, _ in train])
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    std = np.maximum(np.nanstd(flat, axis=0), 1e-3).astype(np.float32)
    del flat
    config = {"n_features": train[0][0].shape[1], "hidden": args.hidden, "layers": args.layers,
              "dropout": args.dropout}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = net.SeqNet(**config).to(device)
    predictor = net.Predictor(model, mean, std, config, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * args.steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    weights = np.array([len(x) for x, _ in train], dtype=np.float64)
    weights /= weights.sum()
    normed = [(predictor.normalise(x), y.astype(np.int64)) for x, y in train]
    print("{} sequences, {} features, device {}".format(len(train), config["n_features"], device),
          flush=True)

    best, started = -1.0, time.time()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for _ in range(args.steps):
            xs, ys = [], []
            for k in rng.choice(len(normed), size=args.batch, p=weights):
                x, y = normed[k]
                lo = rng.integers(0, len(x) - net.WINDOW + 1)
                xs.append(x[lo:lo + net.WINDOW])
                ys.append(y[lo:lo + net.WINDOW])
            xb = torch.from_numpy(np.stack(xs)).to(device)
            yb = torch.from_numpy(np.stack(ys)).to(device)
            xb = xb + 0.05 * torch.randn_like(xb)
            loss = loss_fn(model(xb).reshape(-1, 3), yb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach())
        line = "epoch {:3d}  loss {:.4f}  {:.0f}s".format(epoch + 1, total / args.steps,
                                                           time.time() - started)
        score = None
        for tag, halves in val_sets.items():
            r = evaluate(predictor, halves)
            line += "  | {} macroF1 {:.3f} team {:.3f} D {:.3f}".format(
                tag, r["macroF1"], r["team_acc"], r["D"])
            score = r["macroF1"] if tag == "corrupted" or score is None else score
        print(line, flush=True)
        if val_sets and score > best or not val_sets:
            best = score if score is not None else best
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            net.save(args.out, model, mean, std, config, {"trained_on": train_names, "val": val})
    print("saved", args.out)


if __name__ == "__main__":
    main()
