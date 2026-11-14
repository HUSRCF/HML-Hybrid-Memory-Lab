import unittest

from scripts.compare_paired_semantic_controllers import functional_units, paired_counts
from scripts.prepare_paired_semantic_updates import build_data


class SemanticComparisonTests(unittest.TestCase):
    def test_bad_before_answer_cannot_fake_unchanged_fact_preservation(self):
        rows = [{"case_id": item["case"]["id"], "base_case_id": item["case"]["base_case_id"], "family": item["case"]["family"],
                 "variant": item["variant"], "role": item["query"]["role"], "answer": item["query"]["answer"], "greedy_exact": True}
                for item in build_data(1, 1)["validation"]]
        perfect = functional_units(rows)
        self.assertEqual(sum(perfect["stable_unchanged_case_role"].values()), 48)
        self.assertEqual(sum(perfect["stable_unchanged_four_values"].values()), 12)
        next(row for row in rows if row["variant"] == "before" and row["role"] == "protected_current")["greedy_exact"] = False
        changed = functional_units(rows)
        self.assertEqual(sum(changed["stable_unchanged_case_role"].values()), 47)
        self.assertEqual(sum(changed["stable_unchanged_four_values"].values()), 11)

    def test_equal_accuracy_does_not_hide_rescues_and_regressions(self):
        result = paired_counts({"a": True, "b": False}, {"a": False, "b": True})
        self.assertEqual(result["net_correct_change"], 0)
        self.assertEqual(result["rescued"], 1)
        self.assertEqual(result["regressed"], 1)
        with self.assertRaises(ValueError):
            paired_counts({"a": True}, {"b": True})


if __name__ == "__main__":
    unittest.main()
