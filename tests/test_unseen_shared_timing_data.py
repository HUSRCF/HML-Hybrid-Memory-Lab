"""Prospective cohort completeness and preservation of the full checkpoint family."""

from copy import deepcopy
from collections import Counter
import unittest

from scripts.unseen_shared_timing_data import (read_plan,validate_plan,references,make_cases,
    NATURAL,ORDER,DATA_SOURCES,unique_prefix_queries)


class UnseenTimingDataTests(unittest.TestCase):
    def test_fixed_grid_has_all_eight_new_streams_depths_styles_and_loads(self):
        p=read_plan();cases=make_cases(p)
        self.assertEqual(len(cases),48)
        self.assertEqual(Counter(c['seed'] for c in cases),{s:6 for s in range(62000,62008)})
        self.assertEqual({(c['dependency_depth'],c['question_style']) for c in cases},
            {(d,s) for d in [1,2,3] for s in ['original','entity_first']})
        for c in cases:self.assertEqual(c['distractor_updates'],8 if c['seed']%2==0 else 64)

    def test_checkpoint_baselines_schedules_and_primary_counts_cannot_be_reduced(self):
        p=read_plan();self.assertEqual(len(NATURAL),8);self.assertEqual(len(ORDER),10)
        for mutation in ['seed','depth','arm','policy','count','training','order']:
            changed=deepcopy(p)
            if mutation=='seed':changed['fresh_seeds'][0]=61000
            elif mutation=='depth':changed['dependency_depths']=[2,3]
            elif mutation=='arm':changed['evaluation_arms'].pop('gdn_all_tokens')
            elif mutation=='policy':changed['evaluation_arms']['shared_question_timing']['token_policy']='all'
            elif mutation=='count':changed['counts']['primary_physical_full_answer_readouts']=2352
            elif mutation=='training':changed['new_training']=True
            else:changed['evaluation_order'].reverse()
            with self.assertRaises(ValueError):validate_plan(changed)

    def test_every_distinct_ledger_prefix_is_selected_once_without_question_filtering(self):
        queries=[{'id':'history','source_key':'a'},{'id':'current','source_key':'a'},
            {'id':'other','source_key':'b'},{'id':'before','source_key':'c'}]
        self.assertEqual([q['id'] for q in unique_prefix_queries(queries)],['history','other','before'])

    def test_actual_references_preserve129sources_and_original_failed_gates(self):
        r=references(read_plan());self.assertEqual(len(DATA_SOURCES),132)
        self.assertEqual(len(r['training']),7)
        self.assertTrue(all(not v['passed'] for v in r['prior_fresh_gates'].values()))
        self.assertEqual(r['training']['shared']['checkpoint_state_sha256'],
            '21037c2505371f50bf76cdf5411a3bcf28b3baf5fd83da932b0e4036e3c4f197')


if __name__=='__main__':unittest.main()
