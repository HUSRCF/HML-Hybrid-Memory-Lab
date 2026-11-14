from types import SimpleNamespace
import unittest

from sas_memory.record_kv_ablation import record_blocked_mask
from sas_memory.tasks import INSTRUCTIONS


class RecordKVAblationTests(unittest.TestCase):
    def test_record_values_and_question_cannot_change_allowed_history(self):
        runtime = SimpleNamespace(prompt_start="CHAT:", ids=lambda text: [ord(c) for c in text])
        prelude = runtime.prompt_start + INSTRUCTIONS
        left = prelude + "Set E123 to 777.\n"
        right = prelude + "Set E987 to 222.\n"
        a = record_blocked_mask(runtime, left, left + "Question: current E123?")
        b = record_blocked_mask(runtime, right, right + "Question: initial E987? Answer: arbitrary")
        self.assertEqual(a["keep_bits"], b["keep_bits"])
        self.assertEqual(a["keep_sha256"], b["keep_sha256"])
        self.assertEqual(a["keep_bits"][:len(prelude)], "1" * len(prelude))
        self.assertEqual(a["keep_bits"][len(prelude):], "0" * (len(left) - len(prelude)))
        self.assertEqual(a["blocked_record_tokens"], len(left) - len(prelude))
        self.assertNotEqual(a["prompt_ids_sha256"], b["prompt_ids_sha256"])

    def test_missing_records_or_incompatible_boundaries_are_rejected(self):
        runtime = SimpleNamespace(prompt_start="", ids=lambda text: [ord(c) for c in text])
        with self.assertRaises(ValueError):
            record_blocked_mask(runtime, INSTRUCTIONS, INSTRUCTIONS + "Question")
        with self.assertRaises(ValueError):
            record_blocked_mask(runtime, "No instructions", "No instructions Question")
        with self.assertRaises(ValueError):
            record_blocked_mask(runtime, INSTRUCTIONS + "record", "Different prefix")
        runtime.ids = lambda text: ([1] if text == INSTRUCTIONS else [2, 3, 4])
        with self.assertRaisesRegex(ValueError, "Tokenizer crosses"):
            record_blocked_mask(runtime, INSTRUCTIONS + "record", INSTRUCTIONS + "record question")


if __name__ == "__main__":
    unittest.main()
