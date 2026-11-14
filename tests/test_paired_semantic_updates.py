from copy import deepcopy
import unittest

from scripts.prepare_paired_semantic_updates import build_data, validate_data


class PairedSemanticUpdateTests(unittest.TestCase):
    def test_truth_order_preservation_and_complete_distinct_value_families(self):
        data = build_data(1, 1)
        counts = validate_data(data, 1, 1)
        self.assertEqual(counts["train"]["examples"], 6 * 4 * 9)
        self.assertEqual(counts["validation"]["query_variant_families"], 6 * 9)
        self.assertEqual(set(counts["train"]["per_family_cases"].values()), {4})

    def test_incorrect_truth_and_missing_counterfactuals_rejected(self):
        data = build_data(1, 1)
        corrupt = deepcopy(data)
        corrupt["train"][0]["query"]["answer"] = "999"
        with self.assertRaises(ValueError):
            validate_data(corrupt, 1, 1)
        with self.assertRaises(ValueError):
            validate_data({**data, "train": data["train"][:-1]}, 1, 1)


if __name__ == "__main__":
    unittest.main()
