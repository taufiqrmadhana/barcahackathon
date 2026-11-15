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


    def test_video_gated_loose_arcs_match_the_reference(self):
        rng = np.random.default_rng(91)
        for duration in (1, 4, 100):
            for tied in (False, True):
                model = Model(Params(dead_min_frames=duration), loose_switch_cost=4.)
                emit = (rng.integers(-1, 2, (300, 5)).astype(float) if tied
                        else rng.normal(size=(300, 5)))
                standard = np.zeros((300, 5)) if tied else rng.uniform(-2, 20, (300, 5))
                gate = np.where(rng.random((300, 2)) > .5, standard[:, :2], -np.inf)
                bonus = np.column_stack([standard, gate])
                np.testing.assert_array_equal(model.viterbi(emit, bonus),
                                              model.viterbi_reference(emit, bonus))

    def test_confident_video_can_change_last_touch_during_flight(self):
        p = Params(dead_min_frames=4)
        emit = np.full((60, 5), -1000.)
        emit[:30, 1] = 0.; emit[:30, 3] = -4.
        emit[30:, 1] = -4.; emit[30:, 3] = 0.
        standard = np.zeros((60, 5))
        old = Model(p).viterbi(emit, standard)
        model = Model(p, loose_switch_cost=4.)
        gates = np.full((60, 2), -np.inf)
        np.testing.assert_array_equal(model.viterbi(emit, np.column_stack([standard, gates])), old)
        gates[30:, 1] = 0.
        path = model.viterbi(emit, np.column_stack([standard, gates]))
        np.testing.assert_array_equal(path[:30], 1)
        np.testing.assert_array_equal(path[30:], 3)
        with self.assertRaises(ValueError):
            model.viterbi(emit, standard)


if __name__ == '__main__':
    unittest.main()
