import unittest

from sas_memory.oracle import evidence_slice
from sas_memory.tasks import FAMILIES, Operation, execute, make_case


class EvidenceSliceTests(unittest.TestCase):
    def test_copy_uses_the_source_at_copy_time(self):
        events = [Operation("SET", "a", 100), Operation("COPY", "b", source="a"), Operation("SET", "a", 200)]
        selected = evidence_slice(events, "b", "current")
        self.assertEqual(selected, events[:2])
        self.assertEqual(execute(selected)["current"]["b"], 100)

    def test_every_query_preserves_executable_truth(self):
        for family in FAMILIES:
            for seed in range(20):
                case = make_case(family, seed, 16)
                for variant in case["variants"].values():
                    events = [Operation(**e) for e in case["prefix"] + variant["tail"]]
                    for query in variant["queries"]:
                        selected = evidence_slice(events, query["entity"], query["version"])
                        self.assertLessEqual(len(selected), len(events))
                        state = execute(selected)
                        self.assertEqual(str(state[query["version"]][query["entity"]]), query["answer"])


if __name__ == "__main__":
    unittest.main()
