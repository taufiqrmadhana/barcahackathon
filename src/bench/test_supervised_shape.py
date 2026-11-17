import copy
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import supervised_shape, pipeline
from possession.config import Params


class ShapeContracts(unittest.TestCase):
    def fixture(self):
        obj = object.__new__(pipeline.Inputs)
        obj.n = 220
        obj.blocks = [(1, 0, 109), (2, 110, 219)]
        obj._cache = {}
        rng = np.random.default_rng(7)
        players = np.full((obj.n, 8, 3), np.nan, dtype=np.float32)
        players[:, :, :2] = rng.uniform(0.1, 0.9, size=(obj.n, 8, 2))
        obj.atd = {"players": players, "ball": np.zeros((obj.n, 3)),
                   "teams": np.array([0] * 4 + [1] * 4), "pitch": (105, 68)}
        return obj

    def test_no_ball_dependence_or_cross_half_context(self):
        obj = self.fixture()
        x, _ = supervised_shape.features(obj, Params())
        other = copy.deepcopy(obj)
        other._cache = {}
        other.atd["ball"][:] = np.nan
        other.atd["players"][110:, :, :2] = 0.5
        changed, _ = supervised_shape.features(other, Params())
        np.testing.assert_array_equal(x[:110], changed[:110])
        other = copy.deepcopy(obj)
        other._cache = {}
        other.atd["ball"][:] = np.nan
        changed, _ = supervised_shape.features(other, Params())
        np.testing.assert_array_equal(x, changed)

    def test_team_view_matches_actual_team_swap(self):
        obj = self.fixture()
        x, names = supervised_shape.features(obj, Params())
        other = copy.deepcopy(obj)
        other._cache = {}
        other.atd["teams"] = 1 - obj.atd["teams"]
        changed, _ = supervised_shape.features(other, Params())
        np.testing.assert_allclose(supervised_shape.swap_view(x, names), changed,
                                   atol=1e-5, equal_nan=True)

    def test_team_replacement_preserves_dead_probability(self):
        q = np.array([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]], dtype=np.float32)
        new = supervised_shape.replace_team(q, np.array([0.8, 0.2]))
        np.testing.assert_array_equal(q[:, 2], new[:, 2])
        np.testing.assert_allclose(q[:, :2].sum(1), new[:, :2].sum(1))
        self.assertAlmostEqual(new[0, 0] / new[0, :2].sum(), 0.8)


if __name__ == "__main__":
    unittest.main()
