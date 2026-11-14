from collections import Counter
from copy import deepcopy
import unittest

from scripts.prepare_order_augmented_semantics import build_augmented
from scripts.prepare_paired_semantic_updates import build_data


class OrderAugmentationTests(unittest.TestCase):
    def test_balanced_independent_updates_preserve_original_data_truth_and_siblings(self):
        original = build_data()
        frozen = deepcopy(original)
        data, orders = build_augmented(original)
        self.assertEqual(original, frozen)
        self.assertEqual(data['validation'], original['validation'])
        self.assertEqual((data, orders), build_augmented(original))
        seen = {}
        for a, b in zip(data['train'], original['train']):
            self.assertEqual(a['query'], b['query'])
            self.assertEqual(a['variant'], b['variant'])
            self.assertEqual(a['case']['variants'], b['case']['variants'])
            self.assertEqual(a['case']['prefix'][8:], b['case']['prefix'][8:])
            permutation = orders[a['case']['base_case_id']]
            self.assertEqual(a['case']['prefix'][:8], [b['case']['prefix'][i] for i in permutation])
            seen[a['case']['base_case_id']] = (a['case']['family'], a['case']['distractor_updates'], permutation != list(range(8)))
            if permutation != list(range(8)):
                self.assertTrue(all(i != j for i, j in enumerate(permutation)))
        self.assertEqual(len(seen), 24)
        self.assertEqual(set(Counter(seen.values()).values()), {1})

    def test_dependent_initial_records_and_changed_truth_are_rejected(self):
        data = build_data()
        data['train'][0]['case']['prefix'][1]['target'] = data['train'][0]['case']['prefix'][0]['target']
        with self.assertRaises(ValueError):
            build_augmented(data)
        data = build_data()
        data['train'][0]['query']['answer'] = '000'
        with self.assertRaises(ValueError):
            build_augmented(data)


if __name__ == '__main__':
    unittest.main()
