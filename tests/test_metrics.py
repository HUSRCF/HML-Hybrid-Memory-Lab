import unittest

from sas_memory.metrics import summarize


class PreservationMetricsTests(unittest.TestCase):
    def test_only_paired_same_gold_is_compared_and_loads_stay_separate(self):
        rows = []
        for load in (4, 128):
            for variant, nll, correct in (("before", 0.1, True), ("control", 0.2, True), ("treatment", 0.7, False)):
                rows.append({"case_id": f"case-{load}", "family": "overwrite", "distractor_updates": 64,
                             "distractor_entities": load, "variant": variant, "role": "protected_current",
                             "answer": "123", "answer_nll": nll, "greedy_exact": correct,
                             "candidate_correct": correct, "ledger_tokens": 100})
        summary = summarize(rows)
        self.assertEqual(len(summary["grouped"]), 6)
        self.assertEqual(len(summary["paired_preservation"]), 2)
        for pair in summary["paired_preservation"]:
            self.assertAlmostEqual(pair["nll_treatment_minus_control"], 0.5)
        rows[-1]["answer"] = "456"
        with self.assertRaises(AssertionError):
            summarize(rows)


if __name__ == "__main__":
    unittest.main()
