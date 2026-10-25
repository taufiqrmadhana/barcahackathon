import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.metrica_open import labels_from_events


class EventLabels(unittest.TestCase):
    def test_one_based_frames_and_challenge_does_not_steal_possession(self):
        rows = [(1, 2, 2, "Home", "PASS", ""),
                (1, 4, 4, "Away", "CHALLENGE", "LOST"),
                (1, 6, 6, "Away", "RECOVERY", ""),
                (2, 10, 12, "Home", "PASS", "")]
        events = pd.DataFrame(rows, columns=["Period", "Start Frame", "End Frame", "Team", "Type", "Subtype"])
        labels = labels_from_events(events, 14)
        np.testing.assert_array_equal(labels, list("XHHHHAXXXHHHXX"))

    def test_dead_interval_ends_at_restart_and_live_play_cancels_it(self):
        rows = [(1, 1, 2, "Home", "PASS", ""),
                (1, 3, 4, "Home", "BALL OUT", ""),
                (1, 8, 8, "Away", "SET PIECE", "THROW IN"),
                (2, 12, 12, "Home", "PASS", "")]
        columns = ["Period", "Start Frame", "End Frame", "Team", "Type", "Subtype"]
        events = pd.DataFrame(rows, columns=columns)
        lab = labels_from_events(events, 14)
        np.testing.assert_array_equal(lab[3:8], list("DDDDA"))
        extra = pd.DataFrame([(1, 6, 6, "Home", "PASS", "")], columns=columns)
        lab = labels_from_events(pd.concat([events, extra]), 14)
        self.assertFalse((lab[:8] == "D").any())


if __name__ == "__main__":
    unittest.main()
