"""Tactical video access: frame-accurate reads of contiguous runs, crops around the action.

Row i of a submission is frame i of the tactical video (DATA.md), so a frame index here is the
same index as everywhere else in the pipeline. Seeking in h264 is only keyframe-accurate in some
builds, so `Video.run` seeks once to a little before the run and then decodes forward, counting
frames itself; `check_seek` verifies that against a pure sequential read.
"""

import cv2
import numpy as np

from games import GAME, asset_paths


class Video:
    def __init__(self, game=GAME, data_dir=None, source="tactical", path=None):
        self.path = path or asset_paths(game, data_dir, source)["video"]
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError("cannot open video: {}".format(self.path))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._next = None  # index of the frame the decoder will return next

    def _seek(self, frame):
        if self._next == frame:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self._next = frame

    def run(self, first, last):
        """Yield (index, BGR uint8 frame) for every frame in [first, last]."""
        first, last = max(0, first), min(self.n - 1, last)
        self._seek(first)
        for index in range(first, last + 1):
            ok, image = self.cap.read()
            if not ok:
                self._next = None
                return
            self._next = index + 1
            yield index, image

    def frames(self, indices):
        """dict index -> BGR frame for a sorted collection of indices, decoded run by run."""
        indices = np.unique(np.asarray(indices, dtype=np.int64))
        out = {}
        if len(indices) == 0:
            return out
        breaks = np.flatnonzero(np.diff(indices) > 50) + 1
        for chunk in np.split(indices, breaks):
            wanted = set(chunk.tolist())
            for index, image in self.run(int(chunk[0]), int(chunk[-1])):
                if index in wanted:
                    out[index] = image
        return out

    def check_seek(self, frame, lead=60):
        """True if a seek to `frame` decodes the same image as reading forward from frame-lead."""
        sequential = None
        for index, image in self.run(frame - lead, frame):
            sequential = image
        self._next = None
        seeked = next(self.run(frame, frame))[1]
        return sequential is not None and np.array_equal(sequential, seeked)

    def close(self):
        self.cap.release()


def to_rgb01(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def crop_box(center, size, image_size):
    """(x0, y0) of a size=(w, h) box centred on `center`, clamped inside the image."""
    w, h = size
    width, height = image_size
    cx, cy = center
    if not np.all(np.isfinite(center)):
        cx, cy = width / 2.0, height / 2.0
    x0 = int(np.clip(round(cx - w / 2.0), 0, width - w))
    y0 = int(np.clip(round(cy - h / 2.0), 0, height - h))
    return x0, y0


def crop(image, origin, size):
    x0, y0 = origin
    w, h = size
    return image[y0:y0 + h, x0:x0 + w]
