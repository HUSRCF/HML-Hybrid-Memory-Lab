import unittest

from sas_memory.tasks import Operation, execute
from sas_memory.training_data import build_eval_cases, build_training_examples, data_fingerprint


class TrainingDataTests(unittest.TestCase):
    def test_namespaces_are_disjoint_and_labels_remain_executable(self):
        train = [item["case"] for item in build_training_examples(20)]
        validation = build_eval_cases("validation", 4, 2)
        test = build_eval_cases("test", 8, 4)
        entity_sets, seed_sets = [], []
        for cases in (train, validation, test):
            entity_sets.append({event["target"] for case in cases for event in case["prefix"]})
            seed_sets.append({case["seed"] for case in cases})
            for case in cases:
                for variant in case["variants"].values():
                    state = execute([Operation(**event) for event in case["prefix"] + variant["tail"]])
                    for query in variant["queries"]:
                        self.assertEqual(query["answer"], str(state[query["version"]][query["entity"]]))
                        self.assertIn(query["entity"], query["question"])
        for index in range(3):
            for other in range(index):
                self.assertFalse(entity_sets[index] & entity_sets[other])
                self.assertFalse(seed_sets[index] & seed_sets[other])
        self.assertEqual(data_fingerprint(build_training_examples(20)), data_fingerprint(build_training_examples(20)))

    def test_locked_confirmation_uses_new_seeds(self):
        pilot = build_eval_cases("test", 8, 4)
        confirmation = build_eval_cases("test", 32, 16, seed_offset=10000)
        self.assertEqual(len(confirmation), 48)
        self.assertFalse({c["seed"] for c in pilot} & {c["seed"] for c in confirmation})
        self.assertEqual(min(c["seed"] for c in confirmation), 40000)
        self.assertEqual(max(c["seed"] for c in confirmation), 40047)


if __name__ == "__main__":
    unittest.main()
