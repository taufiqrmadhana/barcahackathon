"""The accelerated sparse decoder must preserve the reference Viterbi path exactly."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession.config import Params
from possession.hsmm import Model


class DecoderParity(unittest.TestCase):
    def test_paths_with_bonuses_ties_and_dead_duration(self):
        rng = np.random.default_rng(7)
        for duration in (1, 2, 4, 50, 100):
            for tied in (False, True):
                model = Model(Params(dead_min_frames=duration))
                for _ in range(8):
                    emit = (rng.integers(-1, 2, (300, 5)).astype(float) if tied
                            else rng.normal(size=(300, 5)))
                    bonus = (np.zeros((300, 5)) if tied
                             else rng.uniform(-2, 35, (300, 5)))
                    np.testing.assert_array_equal(model.viterbi(emit, bonus),
                                                  model.viterbi_reference(emit, bonus))

    def test_single_frame(self):
        for duration in (1, 100):
            model = Model(Params(dead_min_frames=duration))
            emit, bonus = np.ones((1, 5)), np.zeros((1, 5))
            np.testing.assert_array_equal(model.viterbi(emit, bonus),
                                          model.viterbi_reference(emit, bonus))


if __name__ == '__main__':
    unittest.main()
