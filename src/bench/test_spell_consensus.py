"""Temporal agreement must preserve stoppages, coverage and independent folds."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import hybrid, spell_consensus
from possession.config import Params


def exchange(labels):
    return np.where(labels == 'H', 'A', np.where(labels == 'A', 'H', labels))


class SpellConsensusContracts(unittest.TestCase):
    def test_dead_probability_blend_keeps_primary_team_odds_and_missing_frames(self):
        q=np.array([[.6,.2,.2],[.2,.6,.2],[.4,.4,.2],[0.,0.,1.],[.4,.4,.2]])
        auxiliary=np.array([[.1,.1,.8],[.1,.1,.8],[np.nan,.9,.1],[.4,.4,.2],[.1,.1,.8]])
        mask=np.array([True,True,True,True,False])
        out=hybrid.blend_dead_probability(q,auxiliary,mask,.5)
        np.testing.assert_allclose(out[:2,2],[.5,.5])
        np.testing.assert_allclose(out[:2,:2].sum(axis=1),.5)
        np.testing.assert_allclose(out[:2,0]/out[:2,1],q[:2,0]/q[:2,1])
        np.testing.assert_array_equal(out[2:],q[2:])
        np.testing.assert_array_equal(hybrid.blend_dead_probability(q,auxiliary,mask,0.),q)
        np.testing.assert_array_equal(hybrid.blend_dead_probability(
            q[:,[1,0,2]],auxiliary[:,[1,0,2]],mask,.5),out[:,[1,0,2]])
        np.testing.assert_array_equal(q[0],[.6,.2,.2])

    def test_pooled_confidence_removes_an_oscillation_without_changing_inputs(self):
        labels = np.array(list('DHHAAAHHHHD'))
        q = np.tile([.1, .1, .8], (len(labels), 1))
        q[1:3] = [.6, .2, .2]       # home conditional .75
        q[3:6] = [.12, .68, .2]     # away .85, stronger than only the left neighbour
        q[6:10] = [.76, .04, .2]    # home .95, pooled home .8833
        before = q.copy(); mask = np.ones(len(labels), dtype=bool)
        blocks = [(1, 0, len(labels) - 1)]
        out = spell_consensus.remove_oscillations(labels, q, mask, blocks, 4)
        np.testing.assert_array_equal(out, list('DHHHHHHHHHD'))
        np.testing.assert_array_equal(labels, list('DHHAAAHHHHD'))
        np.testing.assert_array_equal(q, before)
        swapped = spell_consensus.remove_oscillations(exchange(labels), q[:, [1, 0, 2]], mask, blocks, 4)
        np.testing.assert_array_equal(swapped, exchange(out))
        np.testing.assert_array_equal(spell_consensus.remove_oscillations(labels,q,mask,blocks,3), labels)
        # A stronger middle spell, ties and missing probabilities abstain.
        for values in ([.01,.79,.2], [.08,.72,.2], [np.nan,.7,.2], [0.,0.,1.]):
            candidate = q.copy(); candidate[3:6] = values
            np.testing.assert_array_equal(
                spell_consensus.remove_oscillations(labels,candidate,mask,blocks,4), labels)

    def test_oscillations_never_cross_a_half_or_missing_video(self):
        labels = np.array(list('HHAAHHH'))
        q = np.tile([.9, .1, 0.], (len(labels), 1)); q[2:4] = [.4,.6,0.]
        mask = np.ones(len(labels), dtype=bool)
        np.testing.assert_array_equal(spell_consensus.remove_oscillations(
            labels,q,mask,[(1,0,2),(2,3,6)],3), labels)
        mask[0] = False  # coverage is required over the entire triplet
        np.testing.assert_array_equal(spell_consensus.remove_oscillations(
            labels,q,mask,[(1,0,6)],3), labels)

    def test_oscillation_selection_is_one_pass_and_tied_confidence_abstains(self):
        labels=np.array(list('HAAHAAH'));mask=np.ones(len(labels),bool)
        q=np.tile([.4,.6,0.],(len(labels),1));q[labels=='H']=[.9,.1,0.]
        out=spell_consensus.remove_oscillations(labels,q,mask,[(1,0,6)],4)
        # Adjacent proposals share a neighbour, so only one is applied. A newly
        # enlarged H spell cannot trigger another merge during this pass.
        np.testing.assert_array_equal(out,list('HAAHHHH'))
        q[labels=='H']=[.6,.4,0.]
        np.testing.assert_array_equal(spell_consensus.remove_oscillations(
            labels,q,mask,[(1,0,6)],4),labels)

    def test_short_spell_union_requires_agreement_and_preserves_dead_and_unknown(self):
        text = 'D' + 'A'*5 + 'H'*3 + 'A'*5 + 'XD'
        auxiliary_text = 'D' + 'A'*6 + 'H'*3 + 'A'*4 + 'XD'
        labels = np.array(list(text))
        other = np.array(list(auxiliary_text))
        mask = np.ones(len(labels), dtype=bool); blocks = [(1,0,len(labels)-1)]
        out = spell_consensus.extend_agreed_spells(labels,other,mask,blocks,4)
        np.testing.assert_array_equal(out, list('D' + 'A'*5 + 'H'*4 + 'A'*4 + 'XD'))
        np.testing.assert_array_equal(labels, list(text))
        np.testing.assert_array_equal(other, list(auxiliary_text))
        np.testing.assert_array_equal(spell_consensus.extend_agreed_spells(
            exchange(labels),exchange(other),mask,blocks,4), exchange(out))
        for maximum, iou in [(3,.5),(4,.8)]:
            np.testing.assert_array_equal(spell_consensus.extend_agreed_spells(
                labels,other,mask,blocks,maximum,iou), labels)
        mask[9] = False
        np.testing.assert_array_equal(spell_consensus.extend_agreed_spells(
            labels,other,mask,blocks,4), labels)
        for barrier in ('D','X'):
            blocked=labels.copy();blocked[9]=barrier
            np.testing.assert_array_equal(spell_consensus.extend_agreed_spells(
                blocked,other,np.ones(len(labels),bool),blocks,4), blocked)
        np.testing.assert_array_equal(spell_consensus.extend_agreed_spells(
            labels,other,np.ones(len(labels),bool),[(1,0,8),(2,9,15)],4), labels)

    def test_conflicting_spell_unions_use_the_stronger_agreement_once(self):
        labels=np.array(list('DAAAHHHAAXD'));other=np.array(list('DAAAAHHHAXD'))
        mask=np.ones(len(labels),bool);blocks=[(1,0,len(labels)-1)]
        out=spell_consensus.extend_agreed_spells(labels,other,mask,blocks,4)
        np.testing.assert_array_equal(out,list('DAAAAHHAAXD'))
        np.testing.assert_array_equal(spell_consensus.extend_agreed_spells(
            exchange(labels),exchange(other),mask,blocks,4),exchange(out))

    def test_both_temporal_models_exclude_their_heldout_labels_and_allow_auxiliary_abstention(self):
        n=500;frames=np.arange(n)
        columns={'x':np.sin(frames/13.),'z':np.cos(frames/29.)}
        gt=((frames//17)%3).astype(np.int8)
        changed=gt.copy();changed[:100]=(changed[:100]+1)%3
        def train(target):
            return [hybrid.fit(columns,frames,target,names,'logreg','film',{},
                               gap=5,train_stride=1,logreg_c=c)
                    for names,c in [(('x',),.3),(('z',),1.),(('x','z'),3.)]]
        members=train(gt);modified=train(changed)
        temporal=dict(oscillation_max_frames=25,consensus_weights=[.5,0.,.5],
                      consensus_max_frames=25,consensus_min_iou=.5,consensus_dead_weight=.5)
        router=dict(expert=2,primary_max=.8,expert_min=.9)
        a=hybrid.combine_artifacts(members,[.75,.25,0.],router,temporal=temporal)
        b=hybrid.combine_artifacts(modified,[.75,.25,0.],router,temporal=temporal)
        self.assertEqual(a['version'],hybrid.TEMPORAL_ENSEMBLE_VERSION)
        class Inputs:
            pass
        inputs=Inputs();inputs.n=n;inputs.blocks=[(1,0,n-1)]
        p=Params(dead_min_frames=1,switch_cost=0,dead_entry_cost=0,dead_exit_cost=0,
                 hybrid_lock_dead=0,hybrid_prune_max_frames=25)
        baseline=np.full(n,'H');emit=np.zeros((n,5));bonus=emit.copy()
        mask=np.ones(n,dtype=bool);mask[:20]=False
        views=[(columns,mask,frames,{}) for _ in members]
        first=hybrid.refine(inputs,p,baseline,emit,bonus,None,a,views)
        second=hybrid.refine(inputs,p,baseline,emit,bonus,None,b,views)
        np.testing.assert_array_equal(first[:100],second[:100])
        np.testing.assert_array_equal(first[~mask],baseline[~mask])
        # Removing all auxiliary/expert coverage must equal the same primary-only
        # decode plus its own oscillation cleanup, not shrink primary coverage.
        views[2]=(columns,np.zeros(n,dtype=bool),frames,{})
        without=hybrid.refine(inputs,p,baseline,emit,bonus,None,a,views)
        primary_only=hybrid.combine_artifacts(members,[.75,.25,0.],router,
            temporal=dict(temporal,consensus_weights=[.75,.25,0.],consensus_dead_weight=0.))
        expected=hybrid.refine(inputs,p,baseline,emit,bonus,None,primary_only,views)
        np.testing.assert_array_equal(without,expected)
        for invalid in [dict(temporal,consensus_weights=[1.,0.]),
                        dict(temporal,consensus_weights=[0.,0.,0.]),
                        dict(temporal,consensus_weights=[np.nan,0.,1.]),
                        dict(temporal,consensus_min_iou=0.),
                        dict(temporal,consensus_dead_weight=float('nan')),
                        dict(temporal,consensus_dead_weight=1.1),
                        dict(temporal,oscillation_max_frames=-1),
                        dict(temporal,consensus_max_frames=1.5)]:
            with self.assertRaises(ValueError):
                hybrid.combine_artifacts(members,[.75,.25,0.],router,temporal=invalid)
        bad=dict(a,version=hybrid.ROUTED_ENSEMBLE_VERSION)
        with self.assertRaises(ValueError):
            hybrid.refine(inputs,p,baseline,emit,bonus,None,bad,views)


if __name__ == '__main__':
    unittest.main()
