"""Referee whistle candidates from the tactical video's audio track.

A whistle is a loud narrow-band tone around 2-4.5 kHz held for a few hundred milliseconds, unlike
crowd noise (broadband) or chants (low). Per video frame (40 ms hop): peak-to-median ratio of the
power spectrum in that band, the peak frequency, and how stable it is over neighbouring frames.

    ffmpeg -i data/tactical/BAR-ATH_tactical.mp4 -vn -ac 1 -ar 16000 -f s16le cache/cv/audio16k.raw
    python src/cv/audio.py          # -> cache/cv/audio_feats.npz
"""

import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "cache", "cv", "audio16k.raw")
OUT = os.path.join(ROOT, "cache", "cv", "audio_feats.npz")
SR, HOP, WIN = 16000, 640, 1024
LO, HI = 2000.0, 5000.0


def features(n_frames, chunk=20000):
    a = np.memmap(RAW, dtype=np.int16, mode="r")
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    band = (freqs > LO) & (freqs < HI)
    win = np.hanning(WIN).astype(np.float32)
    tonal = np.zeros(n_frames, np.float32)
    peak_f = np.zeros(n_frames, np.float32)
    level = np.zeros(n_frames, np.float32)
    for f0 in range(0, n_frames, chunk):
        f1 = min(n_frames, f0 + chunk)
        s = np.asarray(a[f0 * HOP:f1 * HOP + WIN], dtype=np.float32)
        if len(s) < WIN:
            break
        fr = sliding_window_view(s, WIN)[::HOP][:f1 - f0] * win
        S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        B = S[:, band]
        pk = B.max(axis=1)
        tonal[f0:f0 + len(B)] = 10 * np.log10(pk / (np.median(B, axis=1) + 1e-6) + 1e-9)
        peak_f[f0:f0 + len(B)] = freqs[band][B.argmax(axis=1)]
        level[f0:f0 + len(B)] = 10 * np.log10(S.sum(axis=1) + 1e-6)
    return tonal, peak_f, level


def whistle_score(tonal, peak_f, run=4, df=120.0):
    """Mean tonality over `run` consecutive frames whose peak frequency stays within df Hz."""
    n = len(tonal)
    score = np.zeros(n, np.float32)
    for k in range(n - run):
        f = peak_f[k:k + run]
        if f.max() - f.min() <= df:
            score[k:k + run] = np.maximum(score[k:k + run], tonal[k:k + run].min())
    return score


def main():
    sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ROOT]
    from games import source_frames
    n = source_frames("tactical")
    tonal, peak_f, level = features(n)
    np.savez(OUT, tonal=tonal, peak_f=peak_f, level=level)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
