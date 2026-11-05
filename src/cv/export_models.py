"""Export the public YOLO11s COCO checkpoint to OpenVINO at 640x640 and quantise it to INT8.

Calibration uses 200 unlabelled crops of the tactical video around the ATD ball (with jitter);
no label of any kind is involved. Output: cache/cv/s_640/ (FP32) and cache/cv/s_640_int8/.

    .venv-cv/bin/python src/cv/export_models.py
"""

import glob
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path[:0] = [SRC, ROOT]
OUT = os.path.join(ROOT, "cache", "cv")


def export_fp32():
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(OUT)
    try:
        path = YOLO("yolo11s.pt").export(format="openvino", imgsz=640)
    finally:
        os.chdir(cwd)
    target = os.path.join(OUT, "s_640")
    if os.path.isdir(target):
        shutil.rmtree(target)
    shutil.move(os.path.join(OUT, os.path.basename(path.rstrip("/"))), target)
    return target


def quantise(fp32_dir, n=200, seed=0):
    import nncf
    import openvino as ov
    from cv.detector import square_crop
    from games import GAME, half_video_blocks, video_path
    from possession import atd as atd_io
    from starter.load_homography import load as load_homography, project
    ball = atd_io.load()["ball"]
    H = load_homography()[0]
    graded = np.zeros(len(ball), bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    rng = np.random.default_rng(seed)
    frames = np.sort(rng.choice(np.flatnonzero(graded & ~np.isnan(ball[:, 0])), n, replace=False))
    cap = cv2.VideoCapture(video_path())
    data = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        u, v = project(H[f], ball[f, 0], ball[f, 1])
        crop, _, _ = square_crop(img, u + rng.normal(0, 60), v + rng.normal(0, 60), 240)
        crop = cv2.resize(crop, (640, 640))
        data.append(crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255)
    core = ov.Core()
    model = core.read_model(glob.glob(os.path.join(fp32_dir, "*.xml"))[0])
    q = nncf.quantize(model, nncf.Dataset(data), subset_size=len(data),
                      preset=nncf.QuantizationPreset.MIXED,
                      ignored_scope=nncf.IgnoredScope(types=["Multiply", "Subtract", "Sigmoid"]))
    target = os.path.join(OUT, "s_640_int8")
    os.makedirs(target, exist_ok=True)
    ov.save_model(q, os.path.join(target, "yolo11s.xml"))
    return target


if __name__ == "__main__":
    print(quantise(export_fp32()))
