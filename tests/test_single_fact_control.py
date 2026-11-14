import copy
import unittest

from scripts.prepare_single_fact_control import build_data, validate_data


class SingleFactControlTests(unittest.TestCase):
    def test_same_question_families_require_history_for_all_siblings(self):
        data = build_data(4, 2)
        counts = validate_data(data, 4, 2)
        self.assertEqual(counts["train"]["examples"], 16)
        self.assertEqual(counts["train"]["question_text_lookup_correct_ceiling"], 4)
        self.assertEqual(counts["validation"]["question_text_lookup_correct_ceiling"], 2)

    def test_wrong_truth_duplicate_siblings_and_missing_versions_rejected(self):
        original = build_data(4, 2)
        data = copy.deepcopy(original)
        data["train"][0]["query"]["answer"] = "999"
        with self.assertRaises(ValueError):
            validate_data(data, 4, 2)
        data = copy.deepcopy(original)
        left = data["train"][0]
        right = next(item for item in data["train"] if item is not left and item["case"]["base_case_id"] == left["case"]["base_case_id"])
        right["case"]["prefix"][0]["value"] = left["case"]["prefix"][0]["value"]
        right["query"]["answer"] = left["query"]["answer"]
        with self.assertRaisesRegex(ValueError, "distinct labels"):
            validate_data(data, 4, 2)
        data = copy.deepcopy(original)
        data["train"].pop()
        with self.assertRaises(ValueError):
            validate_data(data, 4, 2)


if __name__ == "__main__":
    unittest.main()
