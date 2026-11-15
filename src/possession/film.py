import os
import time as _time

import numpy as np

FILM_HUB_URL = os.environ.get("FILM_HUB_URL", "https://tfhub.dev/google/film/1")
ALIGN = 64


def _pad(images, align=ALIGN):
    _, h, w, _ = images.shape
    ph, pw = (-h) % align, (-w) % align
    if ph == 0 and pw == 0:
        return images, (h, w)
    return np.pad(images, ((0, 0), (0, ph), (0, pw), (0, 0)), mode="edge"), (h, w)


class FilmInterpolator:
    name = "film"

    def __init__(self, url=FILM_HUB_URL, batch_size=1, gpu_sleep_s=0.08):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        import tensorflow_hub as hub

        self.tf = tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        self.device = "/GPU:0" if gpus else "/CPU:0"
        self.gpu_sleep_s = gpu_sleep_s
        print("FILM interpolator device: {} ({}, throttled batch_size={}, sleep={}s)".format(
            self.device, gpus or "CPU", batch_size, gpu_sleep_s), flush=True)
        with tf.device(self.device):
            self.model = hub.load(url)
        self.batch_size = batch_size

    def __call__(self, x0, x1, t=0.5):
        x0 = np.asarray(x0, dtype=np.float32)
        x1 = np.asarray(x1, dtype=np.float32)
        out = np.empty_like(x0)
        with self.tf.device(self.device):
            for start in range(0, len(x0), self.batch_size):
                stop = start + self.batch_size
                a, (h, w) = _pad(x0[start:stop])
                b, _ = _pad(x1[start:stop])
                time = np.full((len(a), 1), t, dtype=np.float32)
                result = self.model({"x0": self.tf.constant(a), "x1": self.tf.constant(b),
                                     "time": self.tf.constant(time)}, training=False)
                out[start:stop] = np.clip(result["image"].numpy()[:, :h, :w], 0.0, 1.0)
                if self.device == "/GPU:0" and self.gpu_sleep_s > 0:
                    _time.sleep(self.gpu_sleep_s)
        return out


class NaiveInterpolator:
    name = "naive"

    def __call__(self, x0, x1, t=0.5):
        return (1.0 - t) * np.asarray(x0, np.float32) + t * np.asarray(x1, np.float32)


def load(kind="film", **kwargs):
    if kind == "film":
        return FilmInterpolator(**kwargs)
    if kind == "naive":
        return NaiveInterpolator()
    raise ValueError("unknown interpolator: {}".format(kind))
