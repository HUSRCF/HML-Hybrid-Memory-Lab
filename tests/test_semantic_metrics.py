from copy import deepcopy
import unittest

from sas_memory.semantic_metrics import summarize_semantics
from scripts.prepare_paired_semantic_updates import build_data


def correct_rows():
    return [{"case_id": example["case"]["id"], "variant": example["variant"], "role": example["query"]["role"],
             **{key: example["case"][key] for key in ("base_case_id", "family", "value_version")},
             "answer": example["query"]["answer"], "generated": example["query"]["answer"],
             "greedy_exact": True, "answer_nll": .1} for example in build_data(1, 1)["validation"]]


class SemanticMetricsTests(unittest.TestCase):
    def test_complete_truth_has_full_joint_and_family_success(self):
        rows = correct_rows()
        result = summarize_semantics(rows)
        self.assertEqual(result["correct"], 216)
        self.assertEqual(result["four_value_query_groups"]["all_four_correct"], 54)
        self.assertEqual(result["joint_roles"]["all_three_correct"], 72)
        self.assertEqual(result["all_twelve_value_variant_groups"]["all_twelve_correct"], 18)
        self.assertEqual(result["preservation_regressions"], [])
        with self.assertRaises(ValueError):
            summarize_semantics(rows[:-1])
        with self.assertRaises(ValueError):
            summarize_semantics(rows + [rows[0]])

    def test_equal_wrong_and_different_wrong_are_not_order_capability(self):
        rows = correct_rows()
        case_ids = {family: next(row["case_id"] for row in rows if row["family"] == family) for family in ("independent_swap", "dependent_swap")}
        for row in rows:
            if row["case_id"] in case_ids.values() and row["role"] == "target_current" and row["variant"] != "before":
                row.update(generated="999" if row["family"] == "independent_swap" or row["variant"] == "control" else "998", greedy_exact=False)
        result = summarize_semantics(rows)
        pairs = {item["family"]: item for item in result["order_pairs"] if item["role"] == "target_current"}
        self.assertEqual(pairs["independent_swap"]["predicted_unchanged_pairs"], 4)
        self.assertEqual(pairs["independent_swap"]["both_correct"], 3)
        self.assertEqual(pairs["dependent_swap"]["predicted_change_pairs"], 4)
        self.assertEqual(pairs["dependent_swap"]["both_correct"], 3)

    def test_unchanged_fact_regression_keeps_case_role_and_variant(self):
        rows = correct_rows()
        chosen = next(row for row in rows if row["role"] == "protected_current" and row["variant"] == "treatment")
        chosen.update(generated="999", greedy_exact=False)
        result = summarize_semantics(rows)
        self.assertEqual(len(result["preservation_regressions"]), 1)
        self.assertEqual(result["preservation_regressions"][0]["case_id"], chosen["case_id"])
        self.assertEqual(result["preservation_regressions"][0]["role"], "protected_current")


if __name__ == "__main__":
    unittest.main()
