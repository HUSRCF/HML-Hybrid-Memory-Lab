import unittest

from sas_memory.tasks import Operation, execute
from sas_memory.stress_tasks import make_stress_case


class StressDesignTests(unittest.TestCase):
    def test_target_labels_are_matched_across_duration_and_load(self):
        for seed in (1, 29, 101):
            reference = make_stress_case(seed, 8, 4)
            for length, load in ((64, 4), (64, 128), (512, 128)):
                case = make_stress_case(seed, length, load)
                self.assertEqual(case["prefix"][:8], reference["prefix"][:8])
                for variant in ("before", "control", "treatment"):
                    self.assertEqual(case["variants"][variant], reference["variants"][variant])
                queried_entities = {q["entity"] for q in case["variants"]["before"]["queries"]}
                answers = {int(q["answer"]) for v in case["variants"].values() for q in v["queries"]}
                for event in case["prefix"][8:]:
                    self.assertNotIn(event["target"], queried_entities)
                    self.assertNotIn(event["value"], answers)
                for variant in case["variants"].values():
                    truth = execute([Operation(**e) for e in case["prefix"] + variant["tail"]])
                    for query in variant["queries"]:
                        self.assertEqual(query["answer"], str(truth[query["version"]][query["entity"]]))


if __name__ == "__main__":
    unittest.main()
