from dataclasses import dataclass

import numpy as np

from .film_features import FEATURES as RAW_FILM
from .model import H, A, D

LABEL_CODE = {"H": H, "A": A, "D": D}

V2_COLS = ("v2_team", "v2_dead", "v2_is_H", "v2_is_A", "v2_is_D", "v2_since_change",
           "v2_until_change", "dist_home", "dist_away", "ball_seen", "ball_filled", "since_seen",
           "last_edge_m", "z_player_speed", "pan_speed", "dead_memory", "quality",
           "players_in_view")
LOOP_COLS = ("ctrl_home", "ctrl_away", "rel_home", "rel_away", "second_home", "second_away",
             "crowd_home", "crowd_away", "shape_H", "shape_A", "shape_D",
             "ball_speed_smooth", "player_speed_smooth", "still_run", "missing_run",
             "bonus_to_H", "bonus_to_A", "bonus_dead_in", "bonus_H_out", "bonus_A_out")
FLOW_COLS = ("flow_cam", "flow_act", "diff_global")
NAIVE_COLS = ("naive_res_mean", "naive_res_p99", "naive_res_ball", "naive_res_home",
              "naive_res_away", "naive_near_home", "naive_near_away", "naive_team_diff",
              "naive_near_diff", "naive_ball_peak")
FILM_COLS = ("film_res_mean", "film_res_p99", "film_explained", "film_res_ball", "film_res_home",
             "film_res_away", "film_near_home", "film_near_away", "film_cnt_home",
             "film_cnt_away", "film_team_diff", "film_near_diff", "film_ball_peak",
             "film_team_peak_diff")

FEATURE_SETS = {
    "v2": V2_COLS,
    "loop": V2_COLS + LOOP_COLS,
    "v2+flow+naive": V2_COLS + LOOP_COLS + FLOW_COLS + NAIVE_COLS,
    "v2+flow+film": V2_COLS + LOOP_COLS + FLOW_COLS + FILM_COLS,
    "v2+all": V2_COLS + LOOP_COLS + FLOW_COLS + NAIVE_COLS + FILM_COLS,
}


@dataclass
class FusionConfig:
    weight: float = 1.0
    p_floor: float = 0.02
    peak_window: int = 2


def _rolling_max(x, w):
    out = x.copy()
    for s in range(1, w + 1):
        out[s:] = np.fmax(out[s:], x[:-s])
        out[:-s] = np.fmax(out[:-s], x[s:])
    return out


def sample_table(frames, values, intervals, cfg=FusionConfig()):
    cols = {k: np.asarray(values[k], dtype=np.float64) for k in RAW_FILM}
    cols["film_team_diff"] = cols["film_res_home"] - cols["film_res_away"]
    cols["film_near_diff"] = cols["film_near_home"] - cols["film_near_away"]
    cols["naive_team_diff"] = cols["naive_res_home"] - cols["naive_res_away"]
    cols["naive_near_diff"] = cols["naive_near_home"] - cols["naive_near_away"]
    peak = {k: np.full(len(frames), np.nan) for k in
            ("film_ball_peak", "naive_ball_peak", "film_team_peak_diff")}
    for first, last in intervals:
        sel = np.flatnonzero((frames >= first) & (frames <= last))
        if len(sel) == 0:
            continue
        w = cfg.peak_window
        peak["film_ball_peak"][sel] = _rolling_max(np.nan_to_num(cols["film_res_ball"][sel]), w)
        peak["naive_ball_peak"][sel] = _rolling_max(np.nan_to_num(cols["naive_res_ball"][sel]), w)
        peak["film_team_peak_diff"][sel] = (
            _rolling_max(cols["film_near_home"][sel], w) - _rolling_max(cols["film_near_away"][sel], w))
    cols.update(peak)
    return cols


def frame_video_features(frames, cols, intervals, n):
    out = {k: np.full(n, np.nan) for k in cols}
    for first, last in intervals:
        sel = np.flatnonzero((frames >= first) & (frames <= last))
        if len(sel) == 0:
            continue
        t = np.arange(first, last + 1)
        for k, v in cols.items():
            vals = v[sel]
            ok = np.isfinite(vals)
            if ok.sum() == 0:
                continue
            out[k][first:last + 1] = np.interp(t, frames[sel][ok], vals[ok])
    return out


def v2_context(feat, score, labels, blocks, dead_mem):
    n = len(labels)
    codes = np.array([LABEL_CODE.get(l, -1) for l in labels])
    since = np.zeros(n)
    until = np.zeros(n)
    for _, first, last in blocks:
        seg = codes[first:last + 1]
        change = np.concatenate(([0], np.flatnonzero(np.diff(seg)) + 1, [len(seg)]))
        for a, b in zip(change[:-1], change[1:]):
            since[first + a:first + b] = np.arange(b - a)
            until[first + a:first + b] = np.arange(b - a)[::-1]
    dist = np.clip(np.nan_to_num(feat["dist"], posinf=40.0), 0, 40.0)
    return {
        "v2_team": score[:, H] - score[:, A],
        "v2_dead": score[:, D] - np.maximum(score[:, H], score[:, A], dtype=np.float64),
        "v2_is_H": (codes == H).astype(float), "v2_is_A": (codes == A).astype(float),
        "v2_is_D": (codes == D).astype(float),
        "v2_since_change": np.minimum(since, 500), "v2_until_change": np.minimum(until, 500),
        "dist_home": dist[:, 0], "dist_away": dist[:, 1],
        "ball_seen": feat["ball_seen"].astype(float), "ball_filled": feat["ball_filled"].astype(float),
        "since_seen": np.minimum(feat["since_seen"], 500).astype(float),
        "last_edge_m": np.clip(np.nan_to_num(feat["last_edge_m"], nan=0.0), -10, 40),
        "z_player_speed": np.nan_to_num(feat["z_player_speed"]),
        "pan_speed": np.nan_to_num(feat["pan_speed"]),
        "dead_memory": dead_mem, "quality": feat["quality"],
        "players_in_view": feat["players_in_view"].astype(float),
    }


def loop_context(loop_state):
    ev = loop_state["ev"]
    emit = loop_state["emit"]
    bonus = loop_state["bonus"]
    shape_q = loop_state["shape"]
    labels = loop_state["labels"]
    margin = loop_state["margin"]
    codes = np.array([LABEL_CODE.get(l, -1) for l in labels])

    emit_h = np.maximum(emit[:, 0], emit[:, 1])
    emit_a = np.maximum(emit[:, 2], emit[:, 3])
    emit_d = emit[:, 4]
    ctrl = np.nan_to_num(ev["ctrl"], nan=0.0)
    rel = np.clip(np.nan_to_num(ev["rel"], nan=15.0, posinf=15.0), 0.0, 15.0)
    second = np.clip(np.nan_to_num(ev["second"], nan=40.0, posinf=40.0), 0.0, 40.0)
    crowd = ev["crowd"].astype(float)

    return {
        "loop_team": emit_h - emit_a,
        "loop_dead": emit_d - np.maximum(emit_h, emit_a),
        "loop_is_H": (codes == H).astype(float),
        "loop_is_A": (codes == A).astype(float),
        "loop_is_D": (codes == D).astype(float),
        "loop_margin": np.clip(np.nan_to_num(margin, nan=0.0), 0.0, 50.0),
        "ctrl_home": ctrl[:, 0],
        "ctrl_away": ctrl[:, 1],
        "rel_home": rel[:, 0],
        "rel_away": rel[:, 1],
        "second_home": second[:, 0],
        "second_away": second[:, 1],
        "crowd_home": crowd[:, 0],
        "crowd_away": crowd[:, 1],
        "shape_H": np.nan_to_num(shape_q[:, 0], nan=1.0 / 3),
        "shape_A": np.nan_to_num(shape_q[:, 1], nan=1.0 / 3),
        "shape_D": np.nan_to_num(shape_q[:, 2], nan=1.0 / 3),
        "loop_p_dead": np.nan_to_num(loop_state["p_dead"], nan=0.5),
        "ball_speed_smooth": np.nan_to_num(ev["ball_speed_smooth"], nan=0.0),
        "player_speed_smooth": np.nan_to_num(ev["player_speed_smooth"], nan=0.0),
        "still_run": np.minimum(ev["still_run"], 500).astype(float),
        "missing_run": np.minimum(ev["missing_run"], 500).astype(float),
        "bonus_to_H": bonus[:, 0],
        "bonus_to_A": bonus[:, 1],
        "bonus_dead_in": bonus[:, 2],
        "bonus_H_out": bonus[:, 3],
        "bonus_A_out": bonus[:, 4],
    }


def matrix(columns, names, rows):
    return np.column_stack([columns[k][rows] for k in names]).astype(np.float64)


def model_zoo(seed=0):
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    zoo = {
        "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(C=0.3, max_iter=3000)),
        "random_forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=8, max_features="sqrt",
                                   n_jobs=-1, random_state=seed)),
        "extra_trees": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=400, min_samples_leaf=8, max_features="sqrt",
                                 n_jobs=-1, random_state=seed)),
        "hist_gbm": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=1.0,
            min_samples_leaf=20, random_state=seed),
    }
    try:
        from lightgbm import LGBMClassifier
        zoo["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=250, num_leaves=8, learning_rate=0.05, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, verbose=-1)
    except ImportError:
        pass
    return zoo


def _proba3(model, x):
    p = np.zeros((len(x), 3))
    p[:, model.classes_.astype(int)] = model.predict_proba(x)
    return p


def contiguous_folds(frames, n_folds):
    edges = np.quantile(frames, np.linspace(0, 1, n_folds + 1))
    return np.clip(np.searchsorted(edges, frames, side="right") - 1, 0, n_folds - 1)


def purged_train(frames, folds, f, gap=50):
    test = frames[folds == f]
    near = (frames >= test.min() - gap) & (frames <= test.max() + gap)
    return (folds != f) & ~near


def cross_validate(x, y, folds, factory, frames=None, gap=50):
    oof = np.zeros((len(y), 3))
    for f in np.unique(folds):
        test = folds == f
        train = purged_train(frames, folds, f, gap) if frames is not None else ~test
        if len(np.unique(y[train])) < 2:
            oof[test] = 1.0 / 3
            continue
        m = factory().fit(x[train], y[train])
        oof[test] = _proba3(m, x[test])
    return oof


def log_loss(y, p, floor=1e-3):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], floor, 1.0))))


def macro_f1(y, p):
    pred = p.argmax(axis=1)
    f1 = []
    for k in range(3):
        tp = np.sum((pred == k) & (y == k))
        fp, fn = np.sum((pred == k) & (y != k)), np.sum((pred != k) & (y == k))
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1))


def select_model(x, y, folds, zoo, frames=None):
    report = {}
    for name, factory in zoo.items():
        oof = cross_validate(x, y, folds, factory, frames)
        report[name] = {"log_loss": log_loss(y, oof), "macro_f1": macro_f1(y, oof), "oof": oof}
    best = min(report, key=lambda k: report[k]["log_loss"])
    return best, report


def fuse(score, proba, mask, cfg=FusionConfig()):
    p = np.clip(proba, cfg.p_floor, 1.0 - cfg.p_floor)
    logp = np.log(p)
    logp -= logp.mean(axis=1, keepdims=True)
    out = score.copy()
    ok = mask & np.isfinite(logp).all(axis=1)
    out[ok] += cfg.weight * logp[ok]
    return out


def fuse_hsmm(emit, proba, mask, cfg=FusionConfig()):
    p = np.clip(proba, cfg.p_floor, 1.0 - cfg.p_floor)
    logp = np.log(p)
    logp -= logp.mean(axis=1, keepdims=True)
    out = emit.copy()
    ok = mask & np.isfinite(logp).all(axis=1)
    out[ok, 0] += cfg.weight * logp[ok, 0]
    out[ok, 1] += cfg.weight * logp[ok, 0]
    out[ok, 2] += cfg.weight * logp[ok, 1]
    out[ok, 3] += cfg.weight * logp[ok, 1]
    out[ok, 4] += cfg.weight * logp[ok, 2]
    return out
