"""Contracts for a fold-aligned P(D) specialist without team-score ownership."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import hybrid
from possession.config import Params
from possession.dead_head import DeadBallClassifier


class DeadHeadContracts(unittest.TestCase):
    def test_binary_probability_mapping_and_constant_target_fallback(self):
        x=np.linspace(-2,2,120)[:,None]
        y=np.where(x[:,0]>.3,2,np.where(x[:,0]>-1,0,1))
        model=DeadBallClassifier().fit(x,y);q=model.predict_proba(x)
        np.testing.assert_array_equal(model.classes_,[0,1,2])
        np.testing.assert_allclose(q.sum(axis=1),1.)
        np.testing.assert_array_equal(q[:,0],q[:,1])
        self.assertGreater(q[x[:,0]>1,2].mean(),q[x[:,0]<-1,2].mean())
        swapped=np.where(y==0,1,np.where(y==1,0,y))
        np.testing.assert_array_equal(DeadBallClassifier().fit(x,swapped).predict_proba(x),q)
        for target,dead in [(np.zeros(120),0.),(np.ones(120),0.),(np.full(120,2),1.)]:
            result=DeadBallClassifier().fit(x,target).predict_proba(x)
            np.testing.assert_array_equal(result[:,2],dead)
            np.testing.assert_array_equal(result[:,:2],(1-dead)/2)

    def test_dead_features_omit_the_existing_dead_decoder_anchors(self):
        names=hybrid.FEATURE_SETS['film_dead_motion']
        self.assertFalse(set(names)&{'current_dead','current_shape_D','v2_dead','dead_memory'})
        self.assertIn('spatial_h_speed_d75',names)
        self.assertIn('film_ball_peak',names)

    def test_dead_expert_uses_its_own_heldout_head_and_abstains_without_coverage(self):
        n=500;frames=np.arange(n)
        columns={'x':np.sin(frames/13.),'z':np.cos(frames/29.)}
        gt=((frames//17)%3).astype(np.int8)
        changed=gt.copy();changed[:100]=(changed[:100]+1)%3
        def train(target):
            return [hybrid.fit(columns,frames,target,names,kind,'film',{},gap=5,
                               train_stride=1,logreg_c=c)
                    for names,kind,c in [(('x',),'logreg',.3),(('z',),'logreg',1.),
                                         (('x','z'),'dead_logreg',1.)]]
        members=train(gt);modified=train(changed)
        temporal=dict(oscillation_max_frames=25,consensus_weights=[.5,.5,0.],
                      consensus_max_frames=25,consensus_min_iou=.5,consensus_dead_weight=.5)
        router=dict(expert=2,weight=.5)
        a=hybrid.combine_artifacts(members,[.75,.25,0.],temporal=temporal,dead_router=router)
        b=hybrid.combine_artifacts(modified,[.75,.25,0.],temporal=temporal,dead_router=router)
        self.assertEqual(a['version'],hybrid.DEAD_ROUTED_ENSEMBLE_VERSION)
        class Inputs:
            pass
        inputs=Inputs();inputs.n=n;inputs.blocks=[(1,0,n-1)]
        p=Params(dead_min_frames=1,switch_cost=0,dead_entry_cost=0,dead_exit_cost=0,
                 hybrid_lock_dead=0,hybrid_prune_max_frames=25)
        baseline=np.full(n,'H');emit=np.zeros((n,5));bonus=emit.copy()
        mask=np.ones(n,bool);mask[:20]=False
        views=[(columns,mask,frames,{}) for _ in members]
        first=hybrid.refine(inputs,p,baseline,emit,bonus,None,a,views)
        second=hybrid.refine(inputs,p,baseline,emit,bonus,None,b,views)
        np.testing.assert_array_equal(first[:100],second[:100])
        np.testing.assert_array_equal(first[~mask],baseline[~mask])
        reference=hybrid.combine_artifacts(members,[.75,.25,0.],temporal=temporal)
        legacy=hybrid.refine(inputs,p,baseline,emit,bonus,None,reference,views)
        zero=hybrid.combine_artifacts(members,[.75,.25,0.],temporal=temporal,
                                     dead_router=dict(router,weight=0.))
        np.testing.assert_array_equal(hybrid.refine(inputs,p,baseline,emit,bonus,None,zero,views),legacy)
        views[2]=(columns,np.zeros(n,bool),frames,{})
        np.testing.assert_array_equal(hybrid.refine(inputs,p,baseline,emit,bonus,None,a,views),legacy)
        for invalid in [dict(router,expert=0),dict(router,expert=True),dict(router,expert=3),
                        dict(router,weight=-.1),dict(router,weight=np.nan),
                        dict(router,extra=1)]:
            with self.assertRaises(ValueError):
                hybrid.combine_artifacts(members,[.75,.25,0.],temporal=temporal,dead_router=invalid)
        with self.assertRaises(ValueError):
            hybrid.combine_artifacts(members,[.75,.25,0.],
                temporal=dict(temporal,consensus_weights=[.25,.25,.5]),dead_router=router)
        with self.assertRaises(ValueError):
            hybrid.refine(inputs,p,baseline,emit,bonus,None,
                          dict(a,version=hybrid.TEMPORAL_ENSEMBLE_VERSION),views)


if __name__=='__main__':
    unittest.main()
