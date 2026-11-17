"""Contact votes must respect team symmetry, half isolation and the D mask."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import contact, hsmm, pipeline
from possession.config import Params


class ContactContracts(unittest.TestCase):
    def inputs(self):
        inputs = object.__new__(pipeline.Inputs)
        inputs.n = 600
        inputs.blocks = [(1, 0, 299), (2, 300, 599)]
        inputs._cache = {}
        players = np.zeros((600, 6, 3), dtype=np.float32)
        players[:, :, :2] = np.array([[20,20], [40,20], [60,20],
                                      [20,40], [40,40], [60,40]]) / [105,68]
        ball = np.zeros((600, 3), dtype=np.float32)
        ball[:300, :2] = players[:300, 0, :2]
        ball[300:, :2] = players[300:, 3, :2]
        inputs.atd = dict(players=players, ball=ball, pitch=(105,68),
                          teams=np.array([0,0,0,1,1,1]))
        return inputs

    def test_half_isolation_and_abstention_when_ball_is_hidden(self):
        inputs = self.inputs()
        p = Params(contact_window=25)
        old = contact.infer(inputs, p)
        np.testing.assert_array_equal(old[20:280], 0)
        changed = copy.deepcopy(inputs)
        changed._cache = {}
        changed.atd['players'][300:] = np.nan
        changed.atd['ball'][300:] = np.nan
        new = contact.infer(changed, p)
        np.testing.assert_array_equal(old[:300], new[:300])
        np.testing.assert_array_equal(new[300:], -1)
        changed.atd['ball'][80:220] = np.nan
        changed._cache = {}
        np.testing.assert_array_equal(contact.infer(changed, p)[110:190], -1)

    def test_team_swap_and_pitch_reflection(self):
        inputs = self.inputs()
        p = Params()
        old = contact.infer(inputs, p)
        other = copy.deepcopy(inputs)
        other._cache = {}
        other.atd['teams'] = 1 - other.atd['teams']
        other.atd['players'][:,:,:2] = 1 - other.atd['players'][:,:,:2]
        other.atd['ball'][:,:2] = 1 - other.atd['ball'][:,:2]
        new = contact.infer(other, p)
        np.testing.assert_array_equal(new[old >= 0], 1 - old[old >= 0])
        np.testing.assert_array_equal(new[old < 0], -1)

    def test_contact_refinement_preserves_dead_and_outside_frames(self):
        inputs = self.inputs()
        inputs.blocks = [(1, 10, 289), (2, 310, 589)]
        p = Params(dead_min_frames=25, w_contact=2., contact_switch_cost=4.)
        rng = np.random.default_rng(5)
        emit = rng.normal(size=(600, 5))
        emit[100:200, 4] += 6
        bonus = rng.uniform(0, 20, (600, 5))
        labels, _ = hsmm.Model(p).decode(emit, bonus, inputs.blocks, inputs.n)
        before_e, before_b = emit.copy(), bonus.copy()
        new, _ = contact.refine(inputs, p, labels, emit, bonus,
                                loose_allowed=np.ones((inputs.n, 2), dtype=bool))
        np.testing.assert_array_equal(new == 'D', labels == 'D')
        np.testing.assert_array_equal(new == 'X', labels == 'X')
        np.testing.assert_array_equal(emit, before_e)
        np.testing.assert_array_equal(bonus, before_b)

    def test_tidy_retains_dead_boundaries_and_short_spells_at_half_edges(self):
        labels = np.array(list('HHHAHHHDDHAAHDDHAAH'))
        out = contact.tidy(labels, [(1, 0, 15), (2, 16, 18)], 3)
        np.testing.assert_array_equal(out == 'D', labels == 'D')
        np.testing.assert_array_equal(out[:7], np.array(list('HHHHHHH')))
        np.testing.assert_array_equal(out[9:13], np.array(list('HHHH')))
        np.testing.assert_array_equal(out[16:], labels[16:])

    def test_local_switch_has_no_effect_without_contact_support(self):
        inputs = self.inputs()
        inputs.atd['ball'][:] = np.nan
        p = Params(dead_min_frames=25, w_contact=1., contact_switch_cost=2.,
                   contact_local_switch=1)
        rng = np.random.default_rng(71)
        emit, bonus = rng.normal(size=(600,5)), rng.uniform(0,15,(600,5))
        labels, _ = hsmm.Model(p).decode(emit, bonus, inputs.blocks, inputs.n)
        changed, _ = contact.refine(inputs, p, labels, emit, bonus)
        np.testing.assert_array_equal(changed, labels)


if __name__ == '__main__':
    unittest.main()
