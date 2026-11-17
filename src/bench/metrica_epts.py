"""Convert public game 3 once with kloppy 3.19.1, streaming to bound memory use.

Run with .venv-pretrain/bin/python. The ordinary pipeline then reads the prepared npz
without importing kloppy. Raw event frame numbers are preserved, not timestamps rounded
relative to a half. No competition data is used here.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "pretrain_data/metrica_open/Sample_Game_3"
FILES = ["Sample_Game_3_" + s for s in ("metadata.xml", "tracking.txt", "events.json")]
VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    return {name: digest(DATA / name) for name in FILES}


def load_prepared():
    manifest = json.loads((DATA / "prepared.json").read_text())
    if manifest["version"] != VERSION or manifest["sources"] != source_hashes():
        raise ValueError("game 3 prepared data is stale; rerun metrica_epts.py")
    if digest(DATA / "prepared.npz") != manifest["prepared_sha256"]:
        raise ValueError("game 3 prepared archive checksum mismatch")
    with np.load(DATA / "prepared.npz") as z:
        result = tuple(z[k] for k in ("players", "ball", "teams", "blocks"))
    return result


def event_table():
    import pandas as pd
    info = json.loads((DATA / "prepared.json").read_text())
    teams = {t: label for t, label in zip(info["team_ids"], ["Home", "Away"])}
    records = json.loads((DATA / FILES[2]).read_text())["data"]
    rows = []
    for event in records:
        sub = event["subtypes"] or []
        sub = sub if isinstance(sub, list) else [sub]
        rows.append({"Team": teams[event["team"]["id"]], "Type": event["type"]["name"],
                     "Subtype": "-".join(s["name"] for s in sub), "Period": event["period"],
                     "Start Frame": event["start"]["frame"], "End Frame": event["end"]["frame"]})
    return pd.DataFrame(rows)


def main():
    import kloppy
    from kloppy import metrica
    if kloppy.__version__ != "3.19.1":
        raise RuntimeError("converter uses kloppy 3.19.1 streaming internals; install that version")
    DATA.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DATA / name
        if not target.exists():
            tmp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/" + name, tmp)
            tmp.replace(target)
    # These are the same metadata and row readers used by load_tracking_epts. Avoid
    # constructing 143k Frame objects, each with every PlayerData object, in memory.
    cls = metrica.load_tracking_epts.__globals__["MetricaEPTSTrackingDataDeserializer"]
    reader = cls.deserialize.__globals__
    with open(DATA / FILES[0], "rb") as f:
        meta = reader["load_metadata"](f)
    if meta.frame_rate != 25 or str(meta.pitch_dimensions.unit.value) != "normed":
        raise ValueError("unsupported FPS/coordinate units")
    if [t.ground.value for t in meta.teams] != ["home", "away"]:
        raise ValueError("unexpected team ordering")
    slots = [(side, player.player_id) for side, team in enumerate(meta.teams) for player in team.players]
    n = max(spec.end_frame for spec in meta.data_format_specifications)
    players = np.full((n, len(slots), 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    period = np.zeros(n, dtype=np.int8)
    seen = np.zeros(n, dtype=bool)
    with open(DATA / FILES[1], "rb") as f:
        for row in reader["read_raw_data"](f, meta):
            i = row["frame_id"] - 1
            if not 0 <= i < n or seen[i]:
                raise ValueError("duplicate or invalid frame id")
            seen[i] = True
            period[i] = row["period_id"] or 0
            for j, (_, pid) in enumerate(slots):
                players[i, j, :2] = [row.get(f"player_{pid}_x", np.nan), row.get(f"player_{pid}_y", np.nan)]
            ball[i, :2] = row["ball_x"], row["ball_y"]
    if not seen.all():
        raise ValueError("tracking frame gaps require explicit handling")
    blocks = []
    for p in meta.periods:
        idx = np.flatnonzero(period == p.id)
        if len(idx) != idx[-1] - idx[0] + 1:
            raise ValueError("noncontiguous period")
        blocks.append((p.id, int(idx[0]), int(idx[-1])))
    # Check coordinates/ids against the public loader on a small prefix.
    prefix = metrica.load_tracking_epts(meta_data=DATA / FILES[0], raw_data=DATA / FILES[1],
                                       coordinates="metrica", limit=100)
    slot_ids = {pid: j for j, (_, pid) in enumerate(slots)}
    for frame in prefix.frames:
        i = frame.frame_id - 1
        np.testing.assert_allclose(ball[i, :2], [frame.ball_coordinates.x, frame.ball_coordinates.y],
                                   atol=1e-6, equal_nan=True)
        for player, pd in frame.players_data.items():
            if pd.coordinates is not None:
                np.testing.assert_allclose(players[i, slot_ids[player.player_id], :2],
                    [pd.coordinates.x, pd.coordinates.y], atol=1e-6, equal_nan=True)
    np.savez_compressed(DATA / "prepared.npz", players=players, ball=ball,
                        teams=np.array([s for s, _ in slots], dtype=np.int8), blocks=np.array(blocks))
    info = {"version": VERSION, "kloppy": kloppy.__version__, "sources": source_hashes(),
            "prepared_sha256": digest(DATA / "prepared.npz"),
            "frame_alignment_ok": True, "prefix_loader_parity": 100,
            "team_ids": [t.team_id for t in meta.teams], "frames": n, "blocks": blocks,
            "pitch": [meta.pitch_dimensions.pitch_length, meta.pitch_dimensions.pitch_width]}
    if info["pitch"] != [105, 68]:
        raise ValueError("game 3 pitch differs from benchmark adapter")
    (DATA / "prepared.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("frames", "blocks", "frame_alignment_ok", "prefix_loader_parity")}))


if __name__ == "__main__":
    main()
