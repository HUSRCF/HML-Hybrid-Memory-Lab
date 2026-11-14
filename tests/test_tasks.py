import unittest

from sas_memory.tasks import FAMILIES, Operation, execute, independent, make_case


class EventSemanticsTests(unittest.TestCase):
    def test_copy_is_a_snapshot_and_add_is_not_idempotent(self):
        events = [Operation("SET", "a", 100), Operation("COPY", "b", source="a"),
                  Operation("ADD", "a", 10), Operation("ADD", "a", 10)]
        self.assertEqual(execute(events)["current"], {"a": 120, "b": 100})
        self.assertEqual(execute(events)["initial"], {"a": 100, "b": 100})

    def test_independence_includes_reads(self):
        self.assertTrue(independent(Operation("SET", "a", 1), Operation("SET", "b", 2)))
        self.assertFalse(independent(Operation("SET", "a", 1), Operation("COPY", "b", source="a")))
        self.assertFalse(independent(Operation("ADD", "a", 1), Operation("ADD", "a", 1)))

    def test_all_labels_follow_interpreter(self):
        for seed in range(20):
            for family in FAMILIES:
                case = make_case(family, seed, 8)
                before = case["variants"]["before"]["queries"]
                for variant in case["variants"].values():
                    state = execute([Operation(**e) for e in case["prefix"] + variant["tail"]])
                    for query in variant["queries"]:
                        self.assertEqual(query["answer"], str(state[query["version"]][query["entity"]]))
                        self.assertIn(query["answer"], case["candidates"])
                        self.assertEqual(len(query["answer"]), 3)
                    self.assertEqual(variant["queries"][1]["answer"], before[1]["answer"])
                    self.assertEqual(variant["queries"][2]["answer"], before[2]["answer"])
                control, treatment = (case["variants"][v]["queries"][0]["answer"] for v in ("control", "treatment"))
                if family in ("independent_swap", "duplicate_set"):
                    self.assertEqual(control, treatment)
                else:
                    self.assertNotEqual(control, treatment)


if __name__ == "__main__":
    unittest.main()
