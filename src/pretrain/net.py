"""The sequence model and the wrapper that applies it to a whole match."""

import numpy as np
import torch
from torch import nn

from .features import STRIDE

WINDOW = 300          # steps of STRIDE frames: 60 s at 25 fps / 5
HOP = 150


class SeqNet(nn.Module):
    """features (B, T, F) -> logits (B, T, 3) for H / A / D. Bidirectional: the decoder that
    consumes the probabilities also looks both ways."""

    def __init__(self, n_features, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                           bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 3))

    def forward(self, x):
        return self.out(self.rnn(self.inp(x))[0])


def save(path, model, mean, std, config, extra=None):
    torch.save({"state": model.state_dict(), "mean": mean, "std": std, "config": config,
                "extra": extra or {}}, path)


class Predictor:
    """Per-frame (N, 3) probabilities of H / A / D from the (N, F) shape features."""

    def __init__(self, model, mean, std, config, device):
        self.model, self.mean, self.std, self.config, self.device = model, mean, std, config, device

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = SeqNet(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["state"])
        return cls(model.eval(), ckpt["mean"], ckpt["std"], ckpt["config"], device)

    def normalise(self, x):
        return np.nan_to_num((x - self.mean) / self.std, nan=0.0).astype(np.float32)

    @torch.no_grad()
    def proba_steps(self, x):
        """(T, 3) probabilities for one sequence of already sub-sampled, raw features."""
        was_training = self.model.training
        self.model.eval()
        x = self.normalise(x)
        acc, hits = np.zeros((len(x), 3), np.float32), np.zeros(len(x), np.float32)
        for lo in range(0, max(1, len(x) - WINDOW + HOP), HOP):
            chunk = torch.from_numpy(x[lo:lo + WINDOW])[None].to(self.device)
            p = torch.softmax(self.model(chunk), dim=-1)[0].cpu().numpy()
            acc[lo:lo + len(p)] += p
            hits[lo:lo + len(p)] += 1
        self.model.train(was_training)
        return acc / np.maximum(hits, 1)[:, None]

    def proba(self, X, blocks):
        """Probabilities for every frame. `blocks` = (first, last) frame pairs inclusive (halves);
        frames outside them get 1/3 each."""
        out = np.full((len(X), 3), 1.0 / 3.0, dtype=np.float32)
        for first, last in blocks:
            idx = np.arange(first, last + 1, STRIDE)
            p = self.proba_steps(X[idx])
            frames = np.arange(first, last + 1)
            for c in range(3):
                out[first:last + 1, c] = np.interp(frames, idx, p[:, c])
            out[first:last + 1] /= out[first:last + 1].sum(axis=1, keepdims=True)
        return out
