"""Referee whistle from the tactical video's audio track.

This referee's whistle is a narrow tone near 3.8 kHz. Per video frame (40 ms hop, 64 ms Hann
window on 16 kHz mono audio): the peak power in 3.65-3.95 kHz over the median power of the
neighbouring bands (3.0-3.55 and 4.05-4.6 kHz), in dB. Crowd noise is broadband and chants are
low, so a blast stands out by 25-35 dB against a median of ~8.

    ffmpeg -i data/tactical/BAR-ATH_tactical.mp4 -vn -ac 1 -ar 16000 -f s16le cache/cv/audio16k.raw
    python src/cv/whistle.py                       # -> cache/cv/whistle38.npy
"""

import os
import subprocess
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "cache", "cv", "audio16k.raw")
OUT = os.path.join(ROOT, "cache", "cv", "whistle38.npy")
SR, HOP, WIN = 16000, 640, 1024
BAND, LO, HI = (3650, 3950), (3000, 3550), (4050, 4600)


def extract_audio(video):
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(SR),
                    "-f", "s16le", RAW], check=True)


def compute(n_frames, chunk=20000):
    a = np.memmap(RAW, dtype=np.int16, mode="r")
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    win = np.hanning(WIN).astype(np.float32)
    mw = (freqs >= BAND[0]) & (freqs < BAND[1])
    mr = ((freqs >= LO[0]) & (freqs < LO[1])) | ((freqs >= HI[0]) & (freqs < HI[1]))
    out = np.zeros(n_frames, np.float32)
    for f0 in range(0, n_frames, chunk):
        f1 = min(n_frames, f0 + chunk)
        s = np.asarray(a[f0 * HOP:f1 * HOP + WIN], dtype=np.float32)
        if len(s) < WIN:
            break
        fr = sliding_window_view(s, WIN)[::HOP][:f1 - f0] * win
        S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        out[f0:f0 + len(S)] = 10 * np.log10(S[:, mw].max(axis=1) / (np.median(S[:, mr], axis=1) + 1e-6))
    return out


def load(n_frames):
    if not os.path.isfile(OUT):
        sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ROOT]
        if not os.path.isfile(RAW):
            from games import video_path
            extract_audio(video_path())
        np.save(OUT, compute(n_frames))
    x = np.load(OUT)
    return x[:, 0] if x.ndim == 2 else x


def events(score, graded, thresh=26.0, run=3, merge=50):
    """Frames where a blast starts: `run` consecutive frames >= thresh dB, merged within `merge`.
    Returns (frames, peak dB of the blast)."""
    m = (score >= thresh) & graded
    full = np.convolve(m.astype(np.int8), np.ones(run, np.int8), mode="valid") >= run
    on = np.flatnonzero(full[1:] & ~full[:-1]) + 1
    frames = []
    for o in on:
        if not frames or o - frames[-1] > merge:
            frames.append(int(o))
    peaks = [float(score[f:f + 25].max()) for f in frames]
    return np.array(frames, np.int64), np.array(peaks, np.float32)


if __name__ == "__main__":
    sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ROOT]
    from games import source_frames
    if os.path.isfile(OUT):
        os.remove(OUT)
    x = load(source_frames("tactical"))
    print("wrote", OUT, x.shape)
