"""Ball candidates from the tactical video with a pretrained COCO detector (class 32, sports ball).

The model is the public Ultralytics YOLO11s COCO checkpoint exported to OpenVINO (optionally
INT8-quantised with NNCF on unlabelled frames of this video) and run on the CPU. Nothing is trained
on this match, so the detector cannot have seen any possession label.

The ball is 4-10 px wide in the 1920x1080 tactical frame, too small for a full frame downscaled to
the network. So the detector looks at a native-resolution square crop around where the ball is
expected (2*half px wide, resized to the network input), and keeps up to MAX_CANDIDATES boxes above
a low score; choosing among them is the tracker's job (balltrack.py).
"""

import glob
import os

import cv2
import numpy as np
import openvino as ov

BALL_CLASS = 32
PERSON_CLASS = 0
MAX_CANDIDATES = 6
MIN_SCORE = 0.03
NMS_IOU = 0.5
PERSON_MIN_SCORE = 0.3
MAX_PEOPLE = 40


def model_path(name):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "cache", "cv", name)
    return glob.glob(os.path.join(root, "*.xml"))[0]


class Detector:
    """Asynchronous OpenVINO detector. submit() a uint8 BGR input of the network size together
    with the affine that maps network pixels back to frame pixels; drain() returns
    {key: (balls (k,5), people (m,8))} in frame pixels."""

    def __init__(self, model_dir, size=640, device="CPU", nreq=0, people=True, hint="THROUGHPUT"):
        core = ov.Core()
        model = core.read_model(model_path(model_dir))
        ppp = ov.preprocess.PrePostProcessor(model)
        ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")) \
            .set_color_format(ov.preprocess.ColorFormat.BGR)
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        ppp.input().preprocess().convert_element_type(ov.Type.f32) \
            .convert_color(ov.preprocess.ColorFormat.RGB).scale(255.0)
        model = ppp.build()
        self.compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": hint})
        self.nreq = nreq or self.compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        self.size = size
        self.people = people
        self.results = {}
        self.queue = ov.AsyncInferQueue(self.compiled, self.nreq)
        self.queue.set_callback(self._done)

    def _done(self, request, userdata):
        key, scale, ox, oy, image = userdata
        raw = request.get_output_tensor(0).data[0]
        balls = parse(raw, scale, ox, oy)
        ppl = people(raw, scale, ox, oy, image) if self.people else np.zeros((0, 8), np.float32)
        self.results[key] = (balls, ppl)

    def submit(self, key, net_input, scale, ox, oy, image=None):
        self.queue.start_async({0: net_input[None]}, (key, scale, ox, oy, image))

    def submit_crop(self, key, image, cx, cy, half):
        """Square native crop of side 2*half centred on (cx, cy), zero-padded at the borders."""
        crop, x0, y0 = square_crop(image, cx, cy, half)
        net = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        self.submit(key, net, 2.0 * half / self.size, x0, y0, image if self.people else None)

    def drain(self):
        self.queue.wait_all()
        out, self.results = self.results, {}
        return out


def square_crop(image, cx, cy, half):
    h, w = image.shape[:2]
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    crop = np.zeros((2 * half, 2 * half, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + 2 * half), min(h, y0 + 2 * half)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return crop, x0, y0


def _boxes(out, cls, min_score, scale, ox, oy):
    score = out[4 + cls]
    keep = np.flatnonzero(score >= min_score)
    if len(keep) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    box = out[:4, keep].T * scale
    box[:, 0] += ox
    box[:, 1] += oy
    return box, score[keep]


def _nms(box, sc, min_score, iou, limit):
    rects = [[float(x - w / 2), float(y - h / 2), float(w), float(h)] for x, y, w, h in box]
    idx = cv2.dnn.NMSBoxes(rects, sc.tolist(), min_score, iou)
    return np.array(idx, dtype=np.int64).reshape(-1)[:limit]


def parse(out, scale, ox=0.0, oy=0.0):
    """(84, N) raw head -> (k, 5): x, y, w, h, score in frame pixels."""
    box, sc = _boxes(out, BALL_CLASS, MIN_SCORE, scale, ox, oy)
    if len(sc) == 0:
        return np.zeros((0, 5), np.float32)
    idx = _nms(box, sc, MIN_SCORE, NMS_IOU, MAX_CANDIDATES)
    return np.column_stack([box[idx], sc[idx]]).astype(np.float32)


def people(out, scale, ox, oy, image):
    """Person boxes with the mean BGR colour of the torso: (k, 8) x, y, w, h, score, b, g, r."""
    box, sc = _boxes(out, PERSON_CLASS, PERSON_MIN_SCORE, scale, ox, oy)
    if len(sc) == 0 or image is None:
        return np.zeros((0, 8), np.float32)
    idx = _nms(box, sc, PERSON_MIN_SCORE, 0.5, MAX_PEOPLE)
    rows = []
    H, W = image.shape[:2]
    for i in idx:
        x, y, w, h = box[i]
        x0, x1 = int(max(0, x - w * 0.25)), int(min(W, x + w * 0.25))
        y0, y1 = int(max(0, y - h * 0.35)), int(min(H, y - h * 0.05))
        patch = image[y0:y1, x0:x1]
        col = patch.reshape(-1, 3).mean(axis=0) if patch.size else np.full(3, np.nan)
        rows.append([x, y, w, h, sc[i], *col])
    return np.array(rows, np.float32).reshape(-1, 8)
