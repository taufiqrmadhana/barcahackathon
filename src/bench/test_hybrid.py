"""Contracts for the merged Valdi validator and its independent fold decoding."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from possession import hybrid, model
from possession.config import Params
from possession.config_valdi import ParamsV2


class HybridContracts(unittest.TestCase):
    def test_legacy_parameters_remain_independent(self):
        p = ParamsV2.load(Path(__file__).resolve().parents[1]/'params_v2.json')
        self.assertGreater(p.dead_cost,0)
        self.assertGreater(p.control_radius,0)
        self.assertEqual(Params().w_contact,0)

    def test_legacy_decoder_acceleration_preserves_the_path(self):
        rng = np.random.default_rng(51)
        for cost in (0.,4.,40.):
            for tied in (False,True):
                score = (rng.integers(-1,2,(500,3)).astype(float) if tied
                         else rng.normal(size=(500,3)))
                np.testing.assert_array_equal(model.viterbi(score,cost,30.),
                                               model.viterbi_reference(score,cost,30.))

    def test_video_coverage_never_bridges_a_missing_interval_or_half(self):
        mask = np.ones(300,dtype=bool)
        mask[150:200] = False
        frames = np.array([0,5,10,100,105,149,200,205,299])
        self.assertEqual(hybrid.covered_intervals(frames,mask),
                         [(0,10),(100,105),(200,205)])

    def test_fusion_keeps_the_dead_mask_and_uncovered_frames(self):
        class Inputs:
            pass
        inputs = Inputs();inputs.n=300;inputs.blocks=[(1,10,289)]
        rng = np.random.default_rng(3)
        p = Params(dead_min_frames=10,hybrid_lock_dead=1,hybrid_weight=2.)
        emit=rng.normal(size=(300,5));emit[100:150,4]+=5
        bonus=np.zeros_like(emit)
        from possession.hsmm import Model
        baseline=Model(p).decode(emit,bonus,inputs.blocks,inputs.n)[0]
        before = emit.copy()
        mask=np.zeros(300,dtype=bool);mask[50:200]=True
        q=rng.dirichlet([1,1,1],size=300)
        labels=hybrid.decode(inputs,p,baseline,emit,bonus,q,mask)
        np.testing.assert_array_equal(labels[~mask],baseline[~mask])
        np.testing.assert_array_equal(labels=='D',baseline=='D')
        np.testing.assert_array_equal(emit,before)

    def test_heldout_labels_cannot_change_their_own_decoded_predictions(self):
        n = 500
        frames = np.arange(n)
        columns = {'x':np.sin(frames/13.),'z':np.cos(frames/29.)}
        gt = ((frames // 17) % 3).astype(np.int8)
        changed = gt.copy();changed[:100] = (changed[:100]+1)%3
        a = hybrid.fit(columns,frames,gt,('x','z'),'logreg','film',{},gap=5,train_stride=1)
        b = hybrid.fit(columns,frames,changed,('x','z'),'logreg','film',{},gap=5,train_stride=1)
        x = np.column_stack(list(columns.values()))
        np.testing.assert_array_equal(a['heads'][0].predict_proba(x),
                                      b['heads'][0].predict_proba(x))
        class Inputs:
            pass
        inputs = Inputs();inputs.n=n;inputs.blocks=[(1,0,n-1)]
        p = Params(dead_min_frames=1,switch_cost=0,dead_entry_cost=0,
                    dead_exit_cost=0,hybrid_lock_dead=0)
        labels = np.full(n,'H');emit=np.zeros((n,5));bonus=emit.copy()
        prepared = (columns,np.ones(n,dtype=bool),frames,{})
        old = hybrid.refine(inputs,p,labels,emit,bonus,None,a,prepared)
        new = hybrid.refine(inputs,p,labels,emit,bonus,None,b,prepared)
        np.testing.assert_array_equal(old[:100],new[:100])


if __name__ == '__main__':
    unittest.main()
