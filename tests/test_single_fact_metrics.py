import unittest

from scripts.analyze_single_fact_probe import family_metrics


class SingleFactMetricsTests(unittest.TestCase):
    def test_constant_guesses_and_wrong_value_changes_are_not_family_success(self):
        rows = [{"base_case_id": "same-question", "value_version": index, "answer": answer,
                 "generated": "200", "greedy_exact": answer == "200", "answer_nll": 1.}
                for index, answer in enumerate(("200", "300", "400", "500"))]
        result = family_metrics(rows)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["all_four_correct"], 0)
        self.assertEqual(result["identical_predictions_in_family"], 1)
        for index, row in enumerate(rows):
            row.update(generated=str(201 + index), greedy_exact=False)
        self.assertEqual(family_metrics(rows)["all_four_correct"], 0)
        self.assertEqual(family_metrics(rows)["identical_predictions_in_family"], 0)
        for row in rows:
            row.update(generated=row["answer"], greedy_exact=True)
        self.assertEqual(family_metrics(rows)["all_four_correct"], 1)
        self.assertEqual(family_metrics(rows)["more_than_one_correct"], 1)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            family_metrics(rows[:-1])
        with self.assertRaisesRegex(ValueError, "Empty"):
            family_metrics([])


if __name__ == "__main__":
    unittest.main()
