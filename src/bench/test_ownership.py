"""Contract tests for leakage boundaries, team symmetry and decoder integration."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import clean, ownership
from possession.config import Params
from bench.ownership_bench import corrupt


class OwnershipContracts(unittest.TestCase):
    def fixture(self):
        n = 200
        rng = np.random.default_rng(41)
        xy = rng.uniform([20, 10], [80, 50], size=(n, 6, 2)).astype(np.float32)
        bxy = rng.uniform([20, 10], [80, 50], size=(n, 2)).astype(np.float32)
        trk = {"players_xy": xy, "players_vel": rng.normal(size=xy.shape),
               "teams": np.array([0, 0, 0, 1, 1, 1]), "pitch": (105, 68),
               "ball": {"xy": bxy, "vel": rng.normal(size=bxy.shape),
                        "seen": np.ones(n, dtype=bool), "known": np.ones(n, dtype=bool)}}
        return trk, [(1, 0, 99), (2, 100, 199)]

    def test_team_swap_and_pitch_reflection_equivariance(self):
        trk, blocks = self.fixture()
        views, _, _ = ownership.features(trk, blocks)
        swapped = copy.deepcopy(trk)
        swapped["teams"] = 1 - trk["teams"]
        changed, _, _ = ownership.features(swapped, blocks)
        np.testing.assert_allclose(views[0], changed[1], equal_nan=True)
        np.testing.assert_allclose(views[1], changed[0], equal_nan=True)
        reflected = copy.deepcopy(trk)
        reflected["players_xy"] = np.asarray(trk["pitch"]) - trk["players_xy"]
        reflected["players_vel"] *= -1
        reflected["ball"]["xy"] = np.asarray(trk["pitch"]) - trk["ball"]["xy"]
        reflected["ball"]["vel"] *= -1
        changed, _, _ = ownership.features(reflected, blocks)
        for a, b in zip(views, changed):
            np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5, equal_nan=True)

    def test_future_half_cannot_change_previous_half_features(self):
        trk, blocks = self.fixture()
        old, _, _ = ownership.features(trk, blocks)
        other = copy.deepcopy(trk)
        other["players_xy"][100:] = np.nan
        other["ball"]["xy"][100:] = np.nan
        other["ball"]["seen"][100:] = False
        other["ball"]["known"][100:] = False
        new, _, _ = ownership.features(other, blocks)
        for a, b in zip(old, new):
            np.testing.assert_array_equal(a[:100], b[:100])

    def test_direct_team_boundaries_only_and_no_cross_half_halo(self):
        labels = np.array(list("HHAAAADDAAHHHH"))
        _, boundary, alive = ownership.targets(labels, [(1, 0, 9), (2, 10, 13)], radius=1)
        np.testing.assert_array_equal(np.flatnonzero(boundary[:, 1]), [1, 2, 3])
        self.assertEqual(boundary[:, 0].sum(), 0)  # A->H at halftime is not a turnover
        self.assertFalse(alive[6])
        self.assertFalse(alive[7])

    def test_long_gap_abstains_and_d_path_inputs_stay_identical(self):
        trk, blocks = self.fixture()
        trk["ball"]["seen"][:] = False
        trk["ball"]["known"][:] = False
        trk["ball"]["xy"][:] = np.nan
        _, support, _ = ownership.features(trk, blocks)
        self.assertFalse(support.any())
        rng = np.random.default_rng(2)
        emit, bonus = rng.normal(size=(200, 5)), rng.uniform(size=(200, 5))
        q = np.tile([0.9, 0.1], (200, 1))
        transitions = np.zeros((200, 2))
        new_e, new_b = ownership.apply(emit, bonus, (q, transitions, support), Params())
        np.testing.assert_array_equal(new_e, emit)
        np.testing.assert_array_equal(new_b, bonus)
        new_e, new_b = ownership.apply(emit, bonus, (q, q, ~support), Params())
        np.testing.assert_array_equal(new_e[:, 4], emit[:, 4])
        np.testing.assert_array_equal(new_b[:, 2:], bonus[:, 2:])
        np.testing.assert_allclose(new_e[:, 0] - emit[:, 0], new_e[:, 1] - emit[:, 1])
        np.testing.assert_allclose(new_e[:, 2] - emit[:, 2], new_e[:, 3] - emit[:, 3])

    def test_corruption_does_not_mutate_input_or_team_membership(self):
        trk, blocks = self.fixture()
        class Inputs:
            pass
        inputs = Inputs()
        inputs.game, inputs.blocks, inputs._cache = "synthetic", blocks, {}
        inputs.atd = {"players": np.dstack([trk["players_xy"] / [105, 68],
                                            np.zeros((200, 6))]),
                      "ball": np.column_stack([trk["ball"]["xy"] / [105, 68], np.zeros(200)]),
                      "pitch": (105, 68), "teams": trk["teams"]}
        before = inputs.atd["players"].copy()
        changed = corrupt(inputs, 7)
        np.testing.assert_array_equal(inputs.atd["players"], before)
        np.testing.assert_array_equal(changed.atd["teams"], inputs.atd["teams"])
        self.assertNotEqual(changed.game, inputs.game)
        self.assertFalse(np.array_equal(changed.atd["players"], before))


if __name__ == "__main__":
    unittest.main()
