import unittest

from sas_memory.generalization_tasks import build_generalization_cases, copy_variant, validate_generalization_cases, copy_update_challenges
from sas_memory.oracle import evidence_slice
from sas_memory.tasks import Operation, execute
from sas_memory.training_data import namespaced_case
from sas_memory.readout_controls import readout_event_indices


class GeneralizationTasksTests(unittest.TestCase):
    def test_readout_controls_keep_truth_and_match_two_extra_records(self):
        for case in build_generalization_cases():
            for variant in ("control", "treatment"):
                events = [Operation(**event) for event in case["prefix"] + case["variants"][variant]["tail"]]
                query = next(q for q in case["variants"][variant]["queries"] if q["role"] == "target_current")
                oracle = set(readout_event_indices(events, query["entity"], query["version"]))
                target_only = set(readout_event_indices(events, query["entity"], query["version"], "target_initial_only"))
                source_only = set(readout_event_indices(events, query["entity"], query["version"], "source_other_only"))
                both = set(readout_event_indices(events, query["entity"], query["version"], "competing_versions"))
                self.assertEqual(target_only & source_only, oracle)
                self.assertEqual(target_only | source_only, both)
                self.assertEqual(len(target_only - oracle), 1)
                self.assertEqual(len(source_only - oracle), 1)
                involved = set().union(*(events[i].reads | events[i].writes for i in oracle))
                for kind in ("competing_versions", "matched_unrelated"):
                    selected = set(readout_event_indices(events, query["entity"], query["version"], kind))
                    self.assertTrue(oracle <= selected)
                    self.assertEqual(len(selected - oracle), 2)
                    for index in selected - oracle:
                        self.assertEqual(events[index].kind, "SET")
                        self.assertEqual(bool(events[index].writes & involved), kind == "competing_versions")
                    self.assertEqual(str(execute([events[i] for i in sorted(selected)])["current"][query["entity"]]), query["answer"])

    def test_source_and_unrelated_updates_are_matched_and_preserve_snapshots(self):
        for case in build_generalization_cases():
            conditions = copy_update_challenges(case)
            source_update = conditions["source_update"][-1]
            unrelated_update = conditions["unrelated_update"][-1]
            self.assertEqual(conditions["source_update"][:-1], conditions["unrelated_update"][:-1])
            self.assertEqual(source_update.value, unrelated_update.value)
            self.assertNotEqual(source_update.target, unrelated_update.target)
            source_initial = next(event["value"] for event in case["prefix"] if event["target"] == source_update.target)
            query = next(query for query in case["variants"]["control"]["queries"] if query["role"] == "target_current")
            self.assertEqual(execute(conditions["known_accounts"])["current"][unrelated_update.target],
                             next(event["value"] for event in case["prefix"] if event["target"] == unrelated_update.target))
            for selected in conditions.values():
                self.assertEqual(execute(selected)["current"][query["entity"]], source_initial)
            self.assertEqual(len(conditions["known_accounts"]), case["dependency_depth"] + 2)

    def test_bad_gold_and_unbalanced_clusters_are_rejected(self):
        cases = build_generalization_cases(seeds=(60000, 60001))
        self.assertEqual(validate_generalization_cases(cases)["base_cases"], 2)
        with self.assertRaisesRegex(ValueError, "Unbalanced"):
            validate_generalization_cases(cases[1:])
        cases[0]["variants"]["control"]["queries"][0]["answer"] = "9999"
        with self.assertRaisesRegex(ValueError, "Gold answer"):
            validate_generalization_cases(cases)

    def test_depth_one_reproduces_original_problem(self):
        base = namespaced_case("copy", 60000, 8, "test")
        case = copy_variant(base)
        for field in ("prefix", "variants", "candidates"):
            self.assertEqual(case[field], base[field])

    def test_snapshot_truth_and_dependency_slices_at_every_depth(self):
        cases = build_generalization_cases()
        self.assertEqual(len({case["base_case_id"] for case in cases}), 8)
        self.assertEqual(len(cases), 48)
        for case in cases:
            update = case["variants"]["control"]["tail"][-1]
            old_source = next(event["value"] for event in case["prefix"] if event["target"] == update["target"])
            for name, variant in case["variants"].items():
                events = [Operation(**event) for event in case["prefix"] + variant["tail"]]
                for query in variant["queries"]:
                    if query["role"] == "target_current" and name != "before":
                        expected = old_source if name == "control" else update["value"]
                        self.assertEqual(query["answer"], str(expected))
                    else:
                        first_value = next(event["value"] for event in case["prefix"] if event["target"] == query["entity"])
                        self.assertEqual(query["answer"], str(first_value))
                    chosen = evidence_slice(events, query["entity"], query["version"])
                    self.assertEqual(str(execute(chosen)[query["version"]][query["entity"]]), query["answer"])
                    if query["role"] == "target_current" and name != "before":
                        self.assertEqual(sum(event.kind == "COPY" for event in chosen), case["dependency_depth"])
                        self.assertEqual(len(chosen), case["dependency_depth"] + 1)


if __name__ == "__main__":
    unittest.main()
