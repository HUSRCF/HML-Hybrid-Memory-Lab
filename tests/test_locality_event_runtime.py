from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from scripts.analyze_write_locality_event_paths import expected_keys, paired_outcomes
from scripts.locality_event_runtime import HistoryWeights, score_worlds, supporting_kv_equal


class LocalityEventRuntimeTests(unittest.TestCase):
    def test_one_free_generation_is_scored_against_both_fixed_worlds(self):
        calls = []

        def probe(*args, **kwargs):
            calls.append((args, kwargs))
            return {'generated': '222\n', 'query_tokens': 26,
                    'candidate_scores': {'111': {'tokens': 3, 'nll_mean': 4.},
                                         '222': {'tokens': 3, 'nll_mean': .1}}}
        query = {'question': 'current?', 'world_answers': {'left': '111', 'right': '222'}}
        result = score_worlds(SimpleNamespace(probe=probe), 'ledger', object(), query)
        self.assertEqual(len(calls), 1)
        self.assertFalse(result['world_scores']['left']['greedy_exact'])
        self.assertTrue(result['world_scores']['right']['greedy_exact'])
        self.assertEqual(result['world_scores']['left']['answer_nll'], 4.)
        self.assertFalse(calls[0][1]['score_batch'])

    def test_causal_evidence_KV_identity_rejects_any_changed_layer(self):
        def cache():
            layers = []
            for i in range(24):
                layer = SimpleNamespace()
                if i % 4 == 3:
                    layer.keys = torch.zeros(1, 1, 10, 2)
                    layer.values = torch.zeros(1, 1, 10, 2)
                layers.append(layer)
            return SimpleNamespace(layers=layers)
        left, right = cache(), cache()
        right.layers[23].values[..., 8, :] = 2
        self.assertTrue(supporting_kv_equal({'left': left, 'right': right}, [3, 6])['exact'])
        right.layers[11].keys[..., 4, :] = 1
        with self.assertRaises(AssertionError):
            supporting_kv_equal({'left': left, 'right': right}, [3, 6])

    def test_actual_probability_guard_detects_suppressed_history_access(self):
        observer = HistoryWeights.__new__(HistoryWeights)
        observer.prelude, observer.boundary = 2, 10
        observer.keep = [True] * 4 + [False] * 6
        observer.current = (10, 2)
        observer.stats = {i: {'prefill_calls': 0, 'cached_calls': 0, 'query_rows': 0,
                              'max_retained_mass': 0., 'max_suppressed_probability': 0.,
                              'max_suppressed_mass': 0.} for i in (3, 7, 11, 15, 19, 23)}
        weights = torch.zeros(1, 1, 2, 12)
        weights[..., 2] = .5
        for index in observer.stats:
            observer.after(index)(None, None, (None, weights))
        observer.validate()
        weights[..., 8] = .01
        observer.after(23)(None, None, (None, weights))
        with self.assertRaises(ValueError):
            observer.validate()

    def test_companion_and_clone_cardinality_and_paired_regressions(self):
        pairs = [{'id': 'a', 'queries': {'preservation': {}, 'current_A': {}}},
                 {'id': 'c', 'queries': {'preservation': {}, 'current_A': {}, 'current_C': {}}}]
        keys = expected_keys(pairs, {'a', 'c'})
        self.assertEqual(len(keys), 34)  # 20 factorial,4 oracle,2 C-full,8 clones.
        self.assertEqual(sum(k[1] == 'full_C_baseline' for k in keys), 2)
        self.assertEqual(paired_outcomes([True, False], [False, True]),
                         {'units': 2, 'reference_correct': 1, 'candidate_correct': 1,
                          'rescued': 1, 'regressed': 1, 'both_correct': 0, 'both_wrong': 0})


if __name__ == '__main__':
    unittest.main()
