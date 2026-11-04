"""FILM features: interpolation residuals around the action, per sampled frame.

For a sample frame t (inside a v3 batch) the frames t-k and t+k are interpolated by FILM to a
predicted frame t, which is compared to the real one. The comparison is done on a full-resolution
crop centred on the ball (tracked, interpolated or last seen, projected with the homography), so
the ball keeps its ~7 px instead of vanishing in a downscale.

Per sample:
    film_res_mean / film_res_p99     residual over the crop: how much un-smooth motion there is
    naive_res_mean                   same with a 50/50 cross-fade (no motion model)
    film_explained                   1 - film / naive: share of the change FILM explains as motion
    film_res_ball                    residual in a disk around the ball pixel (a touch, a kick)
    film_res_home / film_res_away    max residual over each team's player boxes in the crop
    film_near_home / film_near_away  residual of each team's player nearest the ball
    film_cnt_home / film_cnt_away    players of each team inside the crop
    naive_res_ball/home/away/near_*  the same localised residuals for the cross-fade, so an ablation
                                     can separate FILM's motion model from plain frame differencing
    flow_cam                         median optical flow of the whole (downscaled) frame: camera pan
    flow_act                         90th pct of flow minus the camera: how much the players move
    diff_global                      mean absolute frame difference t-k -> t+k

Player boxes come from the ATD positions projected with the homography (feet point, body above it,
height scaled with the camera zoom), which is what makes the residual *team-specific*.
"""

import os
import time

import cv2
import numpy as np

from .homography import apply
from .paths import CACHE_DIR
from .video import Video, crop, crop_box, to_rgb01

FEATURES = ("film_res_mean", "film_res_p99", "naive_res_mean", "film_explained", "film_res_ball",
            "film_res_home", "film_res_away", "film_near_home", "film_near_away",
            "film_cnt_home", "film_cnt_away",
            "naive_res_p99", "naive_res_ball", "naive_res_home", "naive_res_away",
            "naive_near_home", "naive_near_away",
            "flow_cam", "flow_act", "diff_global")

CROP = (256, 192)       # pixels at native 1920x1080 (~8 x 6 m around the ball), multiples of 64
                        # 512x384 costs ~4 s per sample on an M2 CPU, this ~0.9 s
K = 2                   # neighbour offset in frames: interpolate t from t-2 and t+2
GLOBAL = (480, 270)     # downscale for the whole-frame flow
PLAYER_H_M = 1.9        # body box height (m) above the projected feet point
PLAYER_W_M = 0.9
BALL_R_M = 0.6


def action_points(atd, feat, pitch):
    """Normalised pitch point to centre the crop on, per frame: ball, else last ball, else players."""
    scale = np.asarray(pitch, dtype=np.float64)
    ball = feat["ball_xy"] / scale
    last = feat["last_ball"]
    centre = np.where(np.isnan(ball[:, :1]), last, ball)
    players = atd["players"][:, :, :2].astype(np.float64)
    with np.errstate(all="ignore"):
        mean_players = np.nanmean(players, axis=1)
    return np.where(np.isnan(centre[:, :1]), mean_players, centre)


class Extractor:
    def __init__(self, atd, feat, camera, interpolator, naive, video=None):
        self.atd, self.feat, self.cam = atd, feat, camera
        self.interp, self.naive = interpolator, naive
        self.video = video or Video()
        self.centre = action_points(atd, feat, atd["pitch"])
        self.teams = atd["teams"]
        self.width_px = float(camera.image_size[0])

    def _ppm(self, t):
        vw = self.cam.view_width[t]
        return self.width_px / vw if np.isfinite(vw) and vw > 1 else 30.0

    def _project(self, t, xy_norm):
        px, w = apply(self.cam.matrix[[t]], xy_norm[None].astype(np.float64))
        px = px[0]
        px[w[0] <= 0] = np.nan
        return px

    def _team_residual(self, maps, t, origin, ball_px):
        """maps: {prefix: residual map}. Max / nearest-to-ball residual over each team's boxes."""
        ppm = self._ppm(t)
        bw, bh = PLAYER_W_M * ppm / 2.0, PLAYER_H_M * ppm
        xy = self.atd["players"][t, :, :2]
        px = self._project(t, np.nan_to_num(xy, nan=-10.0))
        px[np.isnan(xy[:, 0])] = np.nan
        x0, y0 = origin
        w, h = CROP
        out = {}
        for side, name in ((0, "home"), (1, "away")):
            boxes, dists = [], []
            for u, v in px[self.teams == side]:
                if not (np.isfinite(u) and x0 <= u < x0 + w and y0 <= v < y0 + h):
                    continue
                a = int(max(0, u - bw - x0)), int(min(w, u + bw - x0))
                b = int(max(0, v - bh - y0)), int(min(h, v + 0.1 * bh - y0))
                if a[1] <= a[0] or b[1] <= b[0]:
                    continue
                boxes.append((a, b))
                dists.append(np.hypot(u - ball_px[0], v - ball_px[1])
                             if np.all(np.isfinite(ball_px)) else np.inf)
            out["film_cnt_" + name] = float(len(boxes))
            for prefix, res in maps.items():
                vals = [float(res[b[0]:b[1], a[0]:a[1]].mean()) for a, b in boxes]
                out[prefix + "_res_" + name] = max(vals) if vals else 0.0
                out[prefix + "_near_" + name] = vals[int(np.argmin(dists))] if vals else 0.0
        return out

    def _global(self, a, b):
        ga = cv2.cvtColor(cv2.resize(a, GLOBAL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(b, GLOBAL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        cam = np.median(flow.reshape(-1, 2), axis=0)
        rel = np.hypot(*(flow - cam).reshape(-1, 2).T)
        scale = self.width_px / GLOBAL[0]
        return {"flow_cam": float(np.hypot(*cam) * scale),
                "flow_act": float(np.percentile(rel, 90) * scale),
                "diff_global": float(np.abs(ga.astype(np.float32) - gb).mean() / 255.0)}

    def run(self, frames):
        """Feature rows (dict of arrays) for sample frames; frames must be sorted."""
        frames = np.asarray(frames, dtype=np.int64)
        need = np.unique(np.concatenate([frames - K, frames, frames + K]))
        images = self.video.frames(need)
        rows = {k: np.full(len(frames), np.nan, dtype=np.float32) for k in FEATURES}
        x0s, x1s, mids, meta = [], [], [], []
        for i, t in enumerate(frames):
            if not all(j in images for j in (t - K, t, t + K)):
                continue
            ball_px = self._project(t, self.centre[t][None])[0]
            origin = crop_box(ball_px, CROP, self.video.size)
            x0s.append(to_rgb01(crop(images[t - K], origin, CROP)))
            x1s.append(to_rgb01(crop(images[t + K], origin, CROP)))
            mids.append(to_rgb01(crop(images[t], origin, CROP)))
            meta.append((i, t, origin, ball_px))
            for k, v in self._global(images[t - K], images[t + K]).items():
                rows[k][i] = v
        if not meta:
            return rows
        x0s, x1s, mids = np.stack(x0s), np.stack(x1s), np.stack(mids)
        pred = self.interp(x0s, x1s, 0.5)
        blend = self.naive(x0s, x1s, 0.5)
        for j, (i, t, origin, ball_px) in enumerate(meta):
            res = np.abs(pred[j] - mids[j]).mean(axis=-1)
            nres = np.abs(blend[j] - mids[j]).mean(axis=-1)
            rows["film_res_mean"][i] = res.mean()
            rows["film_res_p99"][i] = np.percentile(res, 99)
            rows["naive_res_mean"][i] = nres.mean()
            rows["naive_res_p99"][i] = np.percentile(nres, 99)
            rows["film_explained"][i] = 1.0 - res.mean() / max(nres.mean(), 1e-6)
            bx, by = ball_px[0] - origin[0], ball_px[1] - origin[1]
            r = max(4, int(BALL_R_M * self._ppm(t)))
            if np.isfinite(bx) and 0 <= bx < CROP[0] and 0 <= by < CROP[1]:
                ys = slice(max(0, int(by) - r), int(by) + r)
                xs = slice(max(0, int(bx) - r), int(bx) + r)
                if res[ys, xs].size:
                    rows["film_res_ball"][i] = res[ys, xs].mean()
                    rows["naive_res_ball"][i] = nres[ys, xs].mean()
            maps = {"film": res, "naive": nres}
            for k, v in self._team_residual(maps, t, origin, ball_px).items():
                rows[k][i] = v
        return rows


class Store:
    """Per-sample feature cache, appended in chunks so a long extraction can stop and resume."""

    def __init__(self, kind="film", game="BAR-ATH"):
        self.path = os.path.join(CACHE_DIR, "{}_v3_{}_features.npz".format(game, kind))
        self.frames = np.zeros(0, dtype=np.int64)
        self.values = {k: np.zeros(0, dtype=np.float32) for k in FEATURES}
        if os.path.isfile(self.path):
            with np.load(self.path) as stored:
                self.frames = stored["frame"]
                for k in FEATURES:
                    if k in stored:
                        self.values[k] = stored[k]

    def missing(self, frames):
        return np.setdiff1d(np.asarray(frames, dtype=np.int64), self.frames)

    def add(self, frames, rows):
        self.frames = np.concatenate([self.frames, frames])
        for k in FEATURES:
            self.values[k] = np.concatenate([self.values[k], rows[k]])
        order = np.argsort(self.frames, kind="stable")
        self.frames = self.frames[order]
        for k in FEATURES:
            self.values[k] = self.values[k][order]

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, frame=self.frames, **self.values)
        os.replace(tmp, self.path)

    def table(self):
        return self.frames, self.values


def extract(extractor, store, frames, chunk=32, log_every=20, budget_s=None):
    """Fill the store for `frames` (skipping cached ones). Returns number of new samples."""
    todo = store.missing(frames)
    started, done = time.time(), 0
    for c, start in enumerate(range(0, len(todo), chunk)):
        part = todo[start:start + chunk]
        store.add(part, extractor.run(part))
        done += len(part)
        if c % log_every == 0 or start + chunk >= len(todo):
            store.save()
            rate = done / max(time.time() - started, 1e-6)
            print("  film {}/{} samples  {:.2f}/s  eta {:.0f} min".format(
                done, len(todo), rate, (len(todo) - done) / max(rate, 1e-6) / 60), flush=True)
        if budget_s and time.time() - started > budget_s:
            break
    store.save()
    return done
