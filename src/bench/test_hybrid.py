"""Contracts for the merged Valdi validator and its independent fold decoding."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace

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
        p = Params(dead_min_frames=10,hybrid_lock_dead=1,hybrid_weight=2.,
                   hybrid_loose_threshold=.8)
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
        pp=replace(p,hybrid_prune_max_frames=25)
        old=hybrid.refine(inputs,pp,labels,emit,bonus,None,a,prepared)
        new=hybrid.refine(inputs,pp,labels,emit,bonus,None,b,prepared)
        np.testing.assert_array_equal(old[:100],new[:100])

    def test_blended_head_excludes_heldout_labels_from_both_members(self):
        n = 500
        frames = np.arange(n)
        columns = {'current_H':(frames // 31) % 2,
                   'film_near_diff':np.sin(frames / 13.),
                   'current_team':np.cos(frames / 29.)}
        gt = ((frames // 17) % 3).astype(np.int8)
        changed = gt.copy(); changed[:100] = (changed[:100] + 1) % 3
        names = tuple(columns)
        a = hybrid.fit(columns, frames, gt, names, 'logreg_blend', 'film', {},
                       gap=5, train_stride=1, logreg_c=1.)
        b = hybrid.fit(columns, frames, changed, names, 'logreg_blend', 'film', {},
                       gap=5, train_stride=1, logreg_c=1.)
        x = np.column_stack(list(columns.values()))
        np.testing.assert_array_equal(a['heads'][0].predict_proba(x),
                                      b['heads'][0].predict_proba(x))
        class Inputs:
            pass
        inputs = Inputs(); inputs.n = n; inputs.blocks = [(1, 0, n - 1)]
        p = Params(dead_min_frames=1, switch_cost=0, dead_entry_cost=0,
                   dead_exit_cost=0, hybrid_lock_dead=1, hybrid_prune_max_frames=25,
                   hybrid_loose_threshold=.8, hybrid_dead_margin=12, hybrid_dead_weight=2.)
        baseline = np.full(n, 'H'); emit = np.zeros((n, 5)); bonus = emit.copy()
        baseline[60:90] = 'D'
        prepared = (columns, np.ones(n, dtype=bool), frames, {})
        first = hybrid.refine(inputs, p, baseline, emit, bonus, None, a, prepared)
        second = hybrid.refine(inputs, p, baseline, emit, bonus, None, b, prepared)
        np.testing.assert_array_equal(first[:100], second[:100])

    def test_soft_member_ignores_hard_context_and_keeps_class_alignment(self):
        rng = np.random.default_rng(43)
        names = ('current_H', 'v2_since_change', 'film_near_diff', 'current_team')
        x = rng.normal(size=(300, len(names)))
        # Non-contiguous classes exercise sklearn voting's label encoding.
        y = np.where(x[:, 2] > 0, 2, 0)
        model = hybrid.factory('logreg_blend', names=names).fit(x, y)
        changed = x.copy(); changed[:, :2] += 100
        soft = model.named_estimators_['soft']
        np.testing.assert_array_equal(soft.predict_proba(x), soft.predict_proba(changed))
        from possession.validator import _proba3
        q = _proba3(model, x)
        np.testing.assert_array_equal(q[:, 1], 0)
        np.testing.assert_allclose(q.sum(axis=1), 1.)
        self.assertGreater(q[x[:, 2] > 1, 2].mean(), q[x[:, 2] < -1, 2].mean())

    def test_loose_eligibility_requires_confident_usable_video(self):
        q = np.array([[.9,.1,0.], [.2,.8,0.], [.04,.36,.6],
                      [0.,0.,1.], [np.nan,.8,.2], [.1,.9,0.]])
        mask = np.array([True,True,True,True,True,False])
        allowed = hybrid.loose_eligibility(q, mask, .8)
        np.testing.assert_array_equal(allowed, [[True,False],[False,False],
            [False,True],[False,False],[False,False],[False,False]])
        np.testing.assert_array_equal(hybrid.loose_eligibility(q[:,[1,0,2]],mask,.8),
                                      allowed[:,::-1])
        with self.assertRaises(ValueError):
            hybrid.loose_eligibility(q, mask, 0.)

    def test_dead_flexibility_preserves_phase_centres_and_half_video_boundaries(self):
        labels = np.array(list('A'*30+'D'*110+'H'*10+'D'*100+'A'*50+'X'*20+'D'*280))
        mask = np.ones(len(labels), dtype=bool); mask[25] = False
        blocks = [(1,0,299),(2,320,599)]
        editable = hybrid.dead_flexibility(labels, mask, blocks, 12, 100)
        self.assertTrue(editable[18:25].all())
        self.assertFalse(editable[25])
        self.assertTrue(editable[30:35].all())
        self.assertFalse(editable[35:135].any())
        self.assertTrue(editable[135:144].all())
        self.assertFalse(editable[144:146].any())
        self.assertFalse(editable[150:250].any())
        self.assertFalse(editable[300:].any())
        self.assertFalse(hybrid.dead_flexibility(labels, mask, blocks, 0, 100).any())

    def test_bounded_dead_refinement_only_changes_eligible_edges(self):
        class Inputs:
            pass
        inputs = Inputs(); inputs.n = 400; inputs.blocks = [(1,10,389)]
        p = Params(dead_min_frames=20, hybrid_loose_threshold=.8, hybrid_dead_margin=12,
                   hybrid_dead_weight=2.)
        rng = np.random.default_rng(77)
        emit = rng.normal(size=(400,5)); emit[100:240,4] += 5
        bonus = np.zeros_like(emit)
        from possession.hsmm import Model
        baseline = Model(p).decode(emit, bonus, inputs.blocks, inputs.n)[0]
        mask = np.zeros(400, dtype=bool); mask[50:350] = True
        q = rng.dirichlet([1,1,1], size=400)
        out = hybrid.decode(inputs,p,baseline,emit,bonus,q,mask)
        editable = hybrid.dead_flexibility(baseline,mask,inputs.blocks,12,20)
        self.assertFalse((((out=='D') != (baseline=='D')) & ~editable).any())
        np.testing.assert_array_equal(out[~mask],baseline[~mask])

    def test_factored_fusion_preserves_team_odds_and_shared_weight_compatibility(self):
        from possession import validator
        q = np.array([[.8,.1,.1],[.02,.02,.96],[.4,.4,.2],[np.nan,.5,.5]])
        mask = np.array([True,False,True,True])
        expected = validator.fuse(np.zeros_like(q),q,mask,validator.FusionConfig(weight=.5))
        np.testing.assert_array_equal(hybrid.probability_evidence(q,mask,.5),expected)
        np.testing.assert_array_equal(hybrid.probability_evidence(q,mask,.5,.5),expected)
        a = hybrid.probability_evidence(q,mask,.5,1.)
        b = hybrid.probability_evidence(q,mask,.5,2.)
        np.testing.assert_allclose(a[:3,0]-a[:3,1],b[:3,0]-b[:3,1])
        np.testing.assert_array_equal(b[[1,3]],0.)

    def test_ensemble_fold_alignment_coverage_and_heldout_label_isolation(self):
        n=500;frames=np.arange(n)
        cols={'x':np.sin(frames/13.),'z':np.cos(frames/29.)}
        gt=((frames//17)%3).astype(np.int8)
        changed=gt.copy();changed[:100]=(changed[:100]+1)%3
        def train(target):
            return [hybrid.fit(cols,frames,target,names,'logreg','film',{},
                               gap=5,train_stride=1,logreg_c=c)
                    for names,c in [(('x','z'),.3),(('z',),1.)]]
        members=train(gt);other=train(changed)
        a=hybrid.combine_artifacts(members,[.75,.25])
        b=hybrid.combine_artifacts(other,[.75,.25])
        class Inputs:
            pass
        inputs=Inputs();inputs.n=n;inputs.blocks=[(1,0,n-1)]
        p=Params(dead_min_frames=1,switch_cost=0,dead_entry_cost=0,dead_exit_cost=0,
                 hybrid_lock_dead=0,hybrid_dead_weight=2.)
        baseline=np.full(n,'H');emit=np.zeros((n,5));bonus=emit.copy()
        one=np.zeros(n,dtype=bool);one[20:450]=True
        two=np.zeros(n,dtype=bool);two[40:400]=True
        views=[(cols,mask,frames,{}) for mask in (one,two)]
        first=hybrid.refine(inputs,p,baseline,emit,bonus,None,a,views)
        second=hybrid.refine(inputs,p,baseline,emit,bonus,None,b,views)
        np.testing.assert_array_equal(first[:100],second[:100])
        np.testing.assert_array_equal(first[~(one&two)],baseline[~(one&two)])
        unused=hybrid.combine_artifacts(members,[1.,0.])
        np.testing.assert_array_equal(
            hybrid.refine(inputs,p,baseline,emit,bonus,None,unused,views),
            hybrid.refine(inputs,p,baseline,emit,bonus,None,members[0],views[0]))
        broken=dict(members[1]);broken['edges']=broken['edges']+1
        with self.assertRaises(ValueError):
            hybrid.combine_artifacts([members[0],broken],[.5,.5])

    def test_weak_island_is_removed_but_confident_short_possession_is_kept(self):
        labels=np.array(list('AAAHHAAA'))
        proba=np.tile([.2,.6,.2],(len(labels),1))
        mask=np.ones(len(labels),dtype=bool)
        out=hybrid.prune_unconfirmed(labels,proba,mask,[(1,0,7)],maximum=2)
        np.testing.assert_array_equal(out,np.array(list('AAAAAAAA')))
        proba[3:5]=[.8,.1,.1]
        np.testing.assert_array_equal(hybrid.prune_unconfirmed(
            labels,proba,mask,[(1,0,7)],maximum=2),labels)
        np.testing.assert_array_equal(labels,np.array(list('AAAHHAAA')))

    def test_pruning_never_crosses_dead_halftime_or_missing_video(self):
        proba=np.tile([.1,.7,.2],(9,1))
        mask=np.ones(9,dtype=bool)
        for text,blocks,maximum in [('AAHHHAAA',[(1,0,7)],2),
            ('DDHHAAA',[(1,0,6)],3),('AAHHAAA',[(1,0,3),(2,4,6)],3),
            ('AAHHXXAA',[(1,0,7)],3)]:
            labels=np.array(list(text));n=len(labels)
            np.testing.assert_array_equal(hybrid.prune_unconfirmed(
                labels,proba[:n],mask[:n],blocks,maximum),labels)
        labels=np.array(list('AAHHAAA'));mask[2]=False
        np.testing.assert_array_equal(hybrid.prune_unconfirmed(
            labels,proba[:7],mask[:7],[(1,0,6)],maximum=3),labels)

    def test_no_team_evidence_and_tied_votes_retain_the_prediction(self):
        labels=np.array(list('AAHHAAA'));mask=np.ones(7,dtype=bool)
        for probs in ([0.,0.,1.],[.3,.3,.4],[np.nan,.3,.7]):
            q=np.tile(probs,(7,1))
            np.testing.assert_array_equal(hybrid.prune_unconfirmed(
                labels,q,mask,[(1,0,6)],maximum=3),labels)
        q=np.tile([.49,.01,.5],(7,1))
        # Conditional alive confidence favours H even though its joint probability is below .5.
        np.testing.assert_array_equal(hybrid.prune_unconfirmed(
            labels,q,mask,[(1,0,6)],maximum=3),labels)

    def test_pruning_is_equivariant_to_exchanging_home_and_away(self):
        labels=np.array(list('DAAHHAAAAXAAHAAD'))
        q=np.tile([.1,.7,.2],(len(labels),1))
        mask=np.ones(len(labels),dtype=bool);blocks=[(1,0,len(labels)-1)]
        def exchange(values):
            return np.where(values=='H','A',np.where(values=='A','H',values))
        first=hybrid.prune_unconfirmed(labels,q,mask,blocks,maximum=3)
        swapped=hybrid.prune_unconfirmed(exchange(labels),q[:,[1,0,2]],mask,blocks,maximum=3)
        np.testing.assert_array_equal(swapped,exchange(first))


if __name__ == '__main__':
    unittest.main()
