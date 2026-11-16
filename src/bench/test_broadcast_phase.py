"""Clock recovery and independent phase routing must not borrow held-out labels."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import broadcast, hybrid, spell_consensus
from possession.config import Params


class BroadcastPhaseContracts(unittest.TestCase):
    def test_clock_alignment_recovers_offsets_without_possession_labels(self):
        rng = np.random.default_rng(12)
        n = 48500
        knots = np.arange(0, n+75, 75)
        target = np.column_stack([np.interp(np.arange(n), knots, rng.uniform(.1,.9,len(knots)))
                                  for _ in range(2)])
        ball = np.full((n,3),np.nan)
        blocks = [(1,500,23000),(2,25000,47500)]
        for (_, first, last), offset in zip(blocks, [-50,-120]):
            take = np.arange(first,last+1)
            ball[take+offset,:2] = target[take]
        seen = np.isfinite(ball[:,:2]).all(1)
        seen[np.arange(n)%11==0] = False
        mapping, accepted, _ = broadcast.align(target,np.ones(n,bool),ball,seen,blocks,np.array([105.3,68.]))
        self.assertEqual([row['offset'] for row in accepted],[-50,-120])
        for (_,first,last),offset in zip(blocks,[-50,-120]):
            np.testing.assert_array_equal(mapping[first:last+1],np.arange(first,last+1)+offset)
        self.assertTrue((mapping[:500] == -1).all())

    def test_phase_overlay_preserves_alive_teams_unknowns_and_coverage(self):
        primary = np.array(list('HADDAXHA'))
        phase = np.array(list('AHHADDDD'))
        mask = np.array([1,1,1,1,1,1,0,1],bool)
        result = spell_consensus.overlay_dead_mask(primary,phase,mask)
        np.testing.assert_array_equal(result,list('HAHAD XHD'.replace(' ','')))
        np.testing.assert_array_equal(primary,list('HADDAXHA'))
        swap=lambda y:np.where(y=='H','A',np.where(y=='A','H',y))
        np.testing.assert_array_equal(spell_consensus.overlay_dead_mask(
            swap(primary),swap(phase),mask),swap(result))

    def test_sparse_training_purges_actual_fold_edges(self):
        frames = np.setdiff1d(np.arange(1000),np.arange(175,200))
        columns = {'frame':np.arange(1000,dtype=float)}
        class Recorder:
            def fit(self,x,y):
                self.frames = x[:,0].copy()
                return self
        with patch.object(hybrid,'factory',side_effect=lambda *a,**kw:Recorder()):
            model=hybrid.fit(columns,frames,np.arange(1000)%3,('frame',),'logreg','film',{},
                             gap=20,train_stride=1,strict_fold_edges=True)
        self.assertGreaterEqual(model['heads'][0].frames.min(),220)
        self.assertFalse(((model['heads'][1].frames>=180)&(model['heads'][1].frames<420)).any())
        self.assertTrue(model['strict_fold_edges'])

    def test_phase_routing_uses_matching_head_and_empty_evidence_abstains(self):
        n=500;frames=np.arange(n);columns={'x':np.linspace(-2,2,n),'z':np.sin(frames/17.)}
        gt=np.where(columns['x']>.5,2,np.where(columns['z']>0,0,1)).astype(np.int8)
        changed=gt.copy();changed[:100]=(changed[:100]+1)%3
        def train(target):
            result=[]
            for names,kind,c,strict in [(('z',),'logreg',.3,False),(('z',),'logreg',1.,False),
                    (('z',),'dead_logreg',1.,False),(('x','z'),'dead_logreg',.3,True)]:
                result.append(hybrid.fit(columns,frames,target,names,kind,'film',{},gap=5,
                              train_stride=1,logreg_c=c,strict_fold_edges=strict))
            return result
        members,modified=train(gt),train(changed)
        temporal=dict(oscillation_max_frames=25,consensus_weights=[.5,.5,0.,0.],
            consensus_max_frames=25,consensus_min_iou=.5,consensus_dead_weight=.5)
        dead=dict(expert=2,weight=.5)
        phase=dict(expert=3,weight=1.,dead_margin=25,keep_teams=True)
        make=lambda m,router:hybrid.combine_artifacts(m,[.75,.25,0.,0.],temporal=temporal,
                                                    dead_router=dead,phase_router=router)
        a,b=make(members,phase),make(modified,phase)
        class Inputs:
            pass
        inputs=Inputs();inputs.n=n;inputs.blocks=[(1,0,249),(2,250,499)]
        p=Params(dead_min_frames=1,switch_cost=0,dead_entry_cost=0,dead_exit_cost=0,
                 hybrid_lock_dead=0,hybrid_prune_max_frames=25)
        mask=np.ones(n,bool);mask[:20]=False
        views=[(columns,mask,frames,{}) for _ in members]
        baseline=np.full(n,'H');emit=np.zeros((n,5));bonus=emit.copy()
        run=lambda model:hybrid.refine(inputs,p,baseline,emit,bonus,None,model,views)
        first,second=run(a),run(b)
        np.testing.assert_array_equal(first[:100],second[:100])
        np.testing.assert_array_equal(first[~mask],baseline[~mask])
        old=run(make(members,None))
        self.assertGreater(int((first!=old).sum()),0)
        np.testing.assert_array_equal(run(make(members,dict(phase,weight=0.))),old)
        views[3]=(columns,np.zeros(n,bool),frames,{})
        np.testing.assert_array_equal(run(a),old)
        for invalid in [dict(phase,expert=True),dict(phase,expert=2),dict(phase,weight=np.nan),
                        dict(phase,dead_margin=0),dict(phase,keep_teams=1),dict(phase,extra=1)]:
            with self.assertRaises(ValueError):make(members,invalid)
        with self.assertRaises(ValueError):
            run(dict(a,version=hybrid.DEAD_ROUTED_ENSEMBLE_VERSION))


if __name__ == '__main__':
    unittest.main()
