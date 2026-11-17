"""Match statistics must not merge separate halves through an excluded video gap."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tune import full_stats


class MatchStatisticsContracts(unittest.TestCase):
    def test_half_boundaries_separate_dead_spells_and_exclude_video_gaps(self):
        class Inputs:
            blocks = [(1, 1, 6), (2, 8, 13)]
            graded = np.array([False]+[True]*6+[False]+[True]*6)
        labels = np.array(list('XDDHHDDXDDAADD'))
        possessions, dead_share, dead_spells = full_stats(labels, Inputs())
        self.assertEqual(possessions, 2)
        self.assertAlmostEqual(dead_share, 2/3)
        self.assertEqual(dead_spells, 4)


if __name__ == '__main__':
    unittest.main()
