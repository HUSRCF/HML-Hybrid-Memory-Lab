import unittest

from scripts.analyze_readout_counterbalance import order_consistency


class CounterbalanceMetricsTests(unittest.TestCase):
    def test_constant_wrong_predictions_are_not_functional_preservation(self):
        rows = []
        for case, answers, correct, losses in (
            ("both_right", ("101", "101\n"), (True, True), (0., 0.)),
            ("constant_wrong", ("999", "999"), (False, False), (1., 2.)),
            ("first_only", ("101", "999"), (True, False), (0., 2.)),
            ("second_only", ("999", "101"), (False, True), (3., 0.)),
        ):
            for order, answer, accuracy, loss in zip(("source_first", "unrelated_first"), answers, correct, losses):
                rows.append({"case_id": case, "variant": order, "generated": answer,
                             "greedy_exact": accuracy, "answer_nll": loss})
        result = order_consistency(rows)
        self.assertEqual(result["pairs"], 4)
        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["both_incorrect"], 1)
        self.assertEqual(result["correctness_disagreements"], 2)
        self.assertEqual(result["generated_string_disagreements"], 2)
        self.assertEqual(result["mean_absolute_answer_nll_difference"], 1.5)
        with self.assertRaisesRegex(ValueError, "Missing late-order"):
            order_consistency(rows[:-1])


if __name__ == "__main__":
    unittest.main()
