"""Automated event-convention sensitivity audit; no competition data or manual labels."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import metrica_open as metrica
from bench.ownership_bench import audit
from bench.shape_bench import training_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="cache/ownership_v2/boundary_audit.json")
    args = ap.parse_args()
    reports = []
    for game in (1, 2, 3):
        report = audit(game)
        _, ball, _, blocks = metrica.tracking(game)
        ev = metrica.events_table(game)
        ev["Subtype"] = ev["Subtype"].fillna("")
        labels = metrica.labels(game, len(ball))
        alternate = ev.copy()
        alternate.loc[alternate.Type == "CARRY", "Type"] = "PASS"
        report["carry_as_live_play_changed_frames"] = int(np.sum(
            metrica.labels_from_events(alternate, len(ball)) != labels))
        advantage = (ev.Type == "FAULT RECEIVED") & ev.Subtype.str.contains("ADVANTAGE")
        report["advantage_fouls"] = int(advantage.sum())
        report["ignore_advantage_fouls_changed_frames"] = int(np.sum(
            metrica.labels_from_events(ev[~advantage], len(ball)) != labels))
        report["training_excluded_frames"] = int(np.sum(training_labels(game, len(ball)) != labels))
        transition = np.flatnonzero((labels[1:] != labels[:-1]) & np.isin(labels[1:], ["H", "A"])
                                  & np.isin(labels[:-1], ["H", "A"])) + 1
        missing = ~np.isfinite(ball[:, 0])
        report["direct_team_transitions"] = len(transition)
        report["transition_ball_missing_share"] = float(missing[transition].mean())
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("notes", "source_sha256", "label_sha256")}))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
