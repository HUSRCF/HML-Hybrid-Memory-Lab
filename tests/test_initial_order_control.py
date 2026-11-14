from copy import deepcopy
import unittest

from scripts.prepare_initial_order_control import build_cases
from scripts.prepare_paired_semantic_updates import build_data


class InitialOrderControlTests(unittest.TestCase):
    def test_derangement_moves_every_initial_record_and_preserves_truth(self):
        data = build_data(1, 1)
        original = deepcopy(data)
        cases, permutations = build_cases(data)
        self.assertEqual(data, original)
        self.assertEqual(len(cases), 24)
        self.assertEqual(len(permutations), 6)
        for permutation in permutations.values():
            self.assertEqual(set(permutation), set(range(8)))
            self.assertTrue(all(index != source for index, source in enumerate(permutation)))
        self.assertEqual((cases, permutations), build_cases(data))

    def test_initial_dependent_or_duplicate_entities_are_rejected(self):
        data = build_data(1, 1)
        chosen = data["validation"][0]["case"]
        chosen["prefix"][0]["target"] = chosen["prefix"][1]["target"]
        with self.assertRaises(ValueError):
            build_cases(data)


if __name__ == "__main__":
    unittest.main()
