from copy import deepcopy
import unittest

from sas_memory.counterbalance_tasks import build_counterbalance_cases, counterbalance_indices, validate_counterbalance_cases, MASKS, ORDERS
from sas_memory.tasks import Operation


class CounterbalanceTasksTests(unittest.TestCase):
    def test_fresh_balanced_grid_and_snapshot_truth(self):
        cases = build_counterbalance_cases()
        validation = validate_counterbalance_cases(cases)
        self.assertEqual(validation["base_cases"], 24)
        self.assertEqual(validation["contexts"], 288)
        self.assertEqual({c["seed"] for c in cases}, set(range(70000, 70024)))
        for case in cases:
            # Independent analytic oracle: source is initialized once before the
            # chain, and COPY is a snapshot. No late record may change this value.
            expected = next(e["value"] for e in case["prefix"] if e["target"] == case["source_entity"])
            self.assertNotEqual(expected, case["new_value"])
            for order in ORDERS:
                self.assertEqual(case["variants"][order]["queries"][0]["answer"], str(expected))
                events = [Operation(**e) for e in case["prefix"] + case["variants"][order]["tail"]]
                oracle = set(counterbalance_indices(case, order, "oracle"))
                self.assertEqual(len(oracle), case["dependency_depth"] + 1)
                for kind in MASKS[1:]:
                    selected = set(counterbalance_indices(case, order, kind))
                    self.assertTrue(oracle < selected)
                    extra, = selected - oracle
                    self.assertIn(extra, (len(events) - 2, len(events) - 1))
                    expected_entity = case["source_entity"] if kind == "extra_source" else case["unrelated_entity"]
                    self.assertEqual(events[extra], Operation("SET", expected_entity, case["new_value"]))

    def test_corrupt_truth_value_chain_and_balance_are_rejected(self):
        cases = build_counterbalance_cases(seeds=range(70000, 70002))
        with self.assertRaisesRegex(ValueError, "Unbalanced"):
            validate_counterbalance_cases(cases[1:])
        for field in ("answer", "late_value", "chain"):
            changed = deepcopy(cases)
            first = changed[0]["variants"]["source_first"]
            if field == "answer":
                first["queries"][0]["answer"] = "99999"
            elif field == "late_value":
                first["tail"][-1]["value"] += 1
            else:
                first["tail"][0]["source"] = changed[0]["target_entity"]
            with self.assertRaises(ValueError):
                validate_counterbalance_cases(changed)


if __name__ == "__main__":
    unittest.main()
