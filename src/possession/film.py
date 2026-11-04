"""Frame interpolation: Google Research FILM (Reda et al., ECCV 2022), plus a naive reference.

    https://github.com/google-research/frame-interpolation
    SavedModel: https://tfhub.dev/google/film/1  (the `film_net` style checkpoint, Apache-2.0)

FILM synthesises the frame half way between two frames. In v3 it is not used to make slow
motion: it is used as a *motion predictor*. Given frames t-k and t+k, FILM predicts frame t as if
everything moved smoothly; the real frame t is then compared with that prediction. Where the play
is smooth (a carried ball, players jogging, a panning camera) the prediction is near perfect;
where something happens that smooth motion cannot explain -- a kick, a touch that turns the ball,
a tackle, a sudden stop -- the interpolation residual spikes, and it spikes *where* it happened.

`NaiveInterpolator` (a 50/50 blend) is the no-motion-model reference: residual(naive) -
residual(FILM) measures how much of the change FILM could explain as motion.

Everything works on float32 RGB in [0, 1], shape (B, H, W, 3).
"""

import os

import numpy as np

FILM_HUB_URL = os.environ.get("FILM_HUB_URL", "https://tfhub.dev/google/film/1")
ALIGN = 64  # FILM's pyramid needs sides divisible by 64 (same as interpolator.py in the repo)


def _pad(images, align=ALIGN):
    _, h, w, _ = images.shape
    ph, pw = (-h) % align, (-w) % align
    if ph == 0 and pw == 0:
        return images, (h, w)
    return np.pad(images, ((0, 0), (0, ph), (0, pw), (0, 0)), mode="edge"), (h, w)


class FilmInterpolator:
    """Thin wrapper around the FILM SavedModel: mid-frames for batches of frame pairs."""

    name = "film"

    def __init__(self, url=FILM_HUB_URL, batch_size=4):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        import tensorflow_hub as hub

        self.tf = tf
        self.model = hub.load(url)
        self.batch_size = batch_size

    def __call__(self, x0, x1, t=0.5):
        """x0, x1: (B, H, W, 3) float32 in [0, 1] -> (B, H, W, 3) frame at time t in (0, 1)."""
        x0 = np.asarray(x0, dtype=np.float32)
        x1 = np.asarray(x1, dtype=np.float32)
        out = np.empty_like(x0)
        for start in range(0, len(x0), self.batch_size):
            stop = start + self.batch_size
            a, (h, w) = _pad(x0[start:stop])
            b, _ = _pad(x1[start:stop])
            time = np.full((len(a), 1), t, dtype=np.float32)
            result = self.model({"x0": self.tf.constant(a), "x1": self.tf.constant(b),
                                 "time": self.tf.constant(time)}, training=False)
            out[start:stop] = np.clip(result["image"].numpy()[:, :h, :w], 0.0, 1.0)
        return out


class NaiveInterpolator:
    """Linear cross-fade: what interpolation looks like with no motion model at all."""

    name = "naive"

    def __call__(self, x0, x1, t=0.5):
        return (1.0 - t) * np.asarray(x0, np.float32) + t * np.asarray(x1, np.float32)


def load(kind="film", **kwargs):
    if kind == "film":
        return FilmInterpolator(**kwargs)
    if kind == "naive":
        return NaiveInterpolator()
    raise ValueError("unknown interpolator: {}".format(kind))
