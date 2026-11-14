import unittest
from types import SimpleNamespace

import torch

from sas_memory.kv_readout import QuestionKVReadout, evidence_token_keep, question_kv_mask


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class KVReadoutTests(unittest.TestCase):
    def test_original_spans_and_prompt_boundary(self):
        tokenizer = CharacterTokenizer()
        keep = evidence_token_keep(tokenizer, "I:A\nB\nQ?", "I:A\nB\n", "I:", ["A\n", "B\n"], [1])
        self.assertEqual(keep, [True, True, False, False, True, True])
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            evidence_token_keep(tokenizer, "other", "other", "I:", ["A\n", "B\n"], [1])

    def test_causality_prefix_identity_and_cached_continuation(self):
        history = [True, False, True, False]
        actual = question_kv_mask(history, 0, 7, 2, torch.float32, "cpu") == 0
        causal = torch.ones(7, 7, dtype=torch.bool).tril()
        self.assertTrue(torch.equal(actual[0, 0, :4], causal[:4]))
        expected = causal.clone()
        expected[4:, 1] = False
        expected[4:, 3] = False
        self.assertTrue(torch.equal(actual[0, 0], expected))
        self.assertTrue(torch.equal(actual[0], actual[1]))
        cached = question_kv_mask(history, 7, 2, 1, torch.bfloat16, "cpu") == 0
        self.assertEqual(cached[0, 0, 0].tolist(), [True, False, True, False, True, True, True, True, False])
        all_history = question_kv_mask([True] * 4, 0, 7, 1, torch.float32, "cpu") == 0
        self.assertTrue(torch.equal(all_history[0, 0], causal))

    def test_hook_applies_to_every_forward_and_detaches(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(_attn_implementation="eager")
                self.embedding = torch.nn.Embedding(10, 4)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, **kwargs):
                return kwargs.get("attention_mask")

        model = FakeModel()
        with QuestionKVReadout(model, [True, False]) as intervention:
            prefill = model(input_ids=torch.tensor([[0, 1, 2]]))
            cached = model(input_ids=torch.tensor([[3]]), past_key_values=SimpleNamespace(get_seq_length=lambda: 3))
            self.assertIsNone(prefill["linear_attention"])
            self.assertLess(cached["full_attention"][0, 0, 0, 1].item(), -1e10)
            self.assertEqual(intervention.calls, {"prefill": 1, "cached": 1})
        self.assertIsNone(model(input_ids=torch.tensor([[0]])))


if __name__ == "__main__":
    unittest.main()
