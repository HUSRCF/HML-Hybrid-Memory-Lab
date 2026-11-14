import hashlib
from types import SimpleNamespace
import unittest

import torch

from scripts.run_controller_trial import reader_context, supervised_loss


class TrainingReaderPolicyTests(unittest.TestCase):
    def runtime(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(_attn_implementation="eager")
                self.embedding = torch.nn.Embedding(8, 4)
                self.output = torch.nn.Linear(4, 8)
                self.observed = None
                self.fail = False

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, logits_to_keep=1, **kwargs):
                self.observed = kwargs.get("attention_mask")
                if self.fail:
                    raise RuntimeError("intentional forward failure")
                return SimpleNamespace(logits=self.output(self.embedding(input_ids[:, -logits_to_keep:])))

        return SimpleNamespace(model=Model(), tensor=lambda ids: torch.tensor(ids))

    def encoded(self):
        return {"input_ids": [0, 1, 2, 3, 4, 5], "answer_ids": [6, 7], "ledger_tokens": 4,
                "reader_mask": {"prelude_tokens": 1, "ledger_tokens": 4, "keep_bits": "1000",
                                "keep_sha256": hashlib.sha256(bytes([1, 0, 0, 0])).hexdigest()}}

    def test_supervised_forward_is_masked_and_backward_survives_hook_removal(self):
        runtime, calls = self.runtime(), []
        loss, logits = supervised_loss(runtime, self.encoded(), reader_policy="records_blocked", reader_calls=calls)
        mask = runtime.model.observed["full_attention"][0, 0]
        self.assertEqual(logits.shape, (1, 2, 8))
        self.assertTrue(torch.all(mask[4:, 1:4] < -1e10))
        self.assertEqual(mask[3, 1].item(), 0.)  # Historical prefix rows are unchanged.
        self.assertLess(mask[4, 5].item(), -1e10)  # Future answer remains inaccessible.
        self.assertEqual(calls, [{"prefill": 1, "cached": 0}])
        self.assertEqual(len(runtime.model._forward_pre_hooks), 0)
        loss.backward()
        self.assertGreater(runtime.model.output.weight.grad.norm().item(), 0)

    def test_invalid_masks_fail_and_forward_failure_does_not_leave_a_hook(self):
        runtime, encoded = self.runtime(), self.encoded()
        encoded["reader_mask"]["keep_bits"] = "1100"
        with self.assertRaises(ValueError):
            supervised_loss(runtime, encoded, reader_policy="records_blocked")
        with self.assertRaises(ValueError):
            reader_context(runtime, "typo")
        runtime.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            supervised_loss(runtime, self.encoded(), reader_policy="records_blocked")
        self.assertEqual(len(runtime.model._forward_pre_hooks), 0)
        runtime.model.fail = False
        supervised_loss(runtime, self.encoded())
        self.assertIsNone(runtime.model.observed)


if __name__ == "__main__":
    unittest.main()
