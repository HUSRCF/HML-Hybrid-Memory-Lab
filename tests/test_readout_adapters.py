import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from sas_memory.readout_adapters import (ReadoutAdapter, StableTrainableAffine,
                                         ConstantInputFrozenAffines, bits_equal)
from sas_memory.runtime import cache_tensors
from scripts.readout_adapter_runtime import regular_cache_clone
from scripts.stable_adapter_linears import StableAdapterLinears
from scripts.readout_adapter_data import load_sources, build_examples, sources
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.analyze_readout_adapter_training import expected_parameters


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.head_dim = 2
        layer.self_attn.q_proj = nn.Linear(8, 16, bias=False)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(15)] + [layer])
        self.requires_grad_(False)

    def forward(self, input_ids, past_key_values=None):
        return self.model.layers[15].self_attn.q_proj(self.embedding(input_ids))


def toy_cache():
    layers = []
    for index in range(24):
        layers.append(SimpleNamespace(**(
            {'keys': torch.randn(1, 2, 9, 2), 'values': torch.randn(1, 2, 9, 2)}
            if index % 4 == 3 else {'recurrent_states': [torch.randn(1, 2, 2, 2)],
                                   'conv_states': [torch.randn(1, 4, 2)]})))
    return SimpleNamespace(layers=layers)


class ReadoutAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1701)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.model = TinyModel()
        self.ids = torch.tensor([[2, 8, 13, 19, 25]])

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_every_head_preserves_the_inactive_half(self):
        native = self.model(input_ids=self.ids).detach()
        for mode in ('query', 'gate', 'query_gate'):
            adapter = ReadoutAdapter(self.model, mode, 4)
            try:
                adapter.set_policy(True, 0)
                with torch.no_grad():
                    for branch in adapter.branches.values():
                        branch['a'].weight.fill_(.2)
                        branch['b'].weight.fill_(.3)
                actual = self.model(input_ids=self.ids).reshape(1, 5, 4, 4)
                original = native.reshape(1, 5, 4, 4)
                for offset, branch in ((0, 'query'), (2, 'gate')):
                    for head in range(4):
                        same = bits_equal(actual[:, :, head, offset:offset + 2], original[:, :, head, offset:offset + 2])
                        self.assertEqual(same, branch not in adapter.branches)
            finally:
                adapter.close()
        self.assertFalse(self.model._forward_pre_hooks)
        self.assertFalse(self.model.model.layers[15].self_attn.q_proj._forward_hooks)

    def test_boundaries_in_full_cached_and_whole_prefix_calls(self):
        native = self.model(input_ids=self.ids).detach()
        adapter = ReadoutAdapter(self.model, 'query_gate', 4)
        try:
            with torch.no_grad():
                for branch in adapter.branches.values():
                    branch['b'].weight.fill_(.1)
            adapter.set_policy(True, 3)
            full = self.model(input_ids=self.ids)
            self.assertTrue(bits_equal(full[:, :3], native[:, :3]))
            self.assertFalse(bits_equal(full[:, 3:], native[:, 3:]))
            cache = SimpleNamespace(get_seq_length=lambda: 2)
            adapter.set_policy(False, 3)
            native_cached = self.model(input_ids=self.ids[:, 2:], past_key_values=cache).detach()
            adapter.set_policy(True, 3)
            cached = self.model(input_ids=self.ids[:, 2:], past_key_values=cache)
            # Native FP32 matmul itself may round differently at M=5 versus M=3.
            # Scope is exact against the SAME native call, not across native shapes.
            self.assertTrue(bits_equal(cached[:, :1], native_cached[:, :1]))
            self.assertFalse(bits_equal(cached[:, 1:], native_cached[:, 1:]))
            adapter.set_policy(True, 20)
            prefix = self.model(input_ids=self.ids)
            self.assertTrue(bits_equal(prefix, native))
            self.assertGreater(adapter.scope['prefix_calls'], 0)
        finally:
            adapter.close()

    def test_zero_identity_gradients_through_attention_and_output_gate(self):
        native = self.model(input_ids=self.ids).detach()
        for mode in ('query', 'gate', 'query_gate'):
            torch.manual_seed(1701)
            adapter = ReadoutAdapter(self.model, mode, 4)
            adapter.set_policy(True, 0)
            optimizer = torch.optim.SGD(adapter.parameters(), lr=.05)
            try:
                actual = self.model(input_ids=self.ids)
                self.assertTrue(bits_equal(actual, native))
                def objective():
                    projected = self.model(input_ids=self.ids).reshape(1, 5, 4, 4)
                    q, gate = projected[..., :2], projected[..., 2:]
                    q = q / torch.sqrt(q.square().mean(-1, keepdim=True) + 1e-6)
                    keys = torch.tensor([[.3, -.7], [-.4, .2], [.8, .5]])
                    values = torch.tensor([[.1, .8], [-.4, .9], [.2, -.3]])
                    weights = torch.softmax(torch.einsum('bthd,nd->bthn', q, keys) / math.sqrt(2), -1)
                    read = torch.einsum('bthn,nd->bthd', weights, values) * gate.sigmoid()
                    return (read - .15).square().sum()
                objective().backward()
                for branch in adapter.branches.values():
                    self.assertGreater(branch['b'].weight.grad.norm().item(), 0)
                    self.assertEqual(branch['a'].weight.grad.norm().item(), 0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                objective().backward()
                for branch in adapter.branches.values():
                    self.assertGreater(branch['a'].weight.grad.norm().item(), 0)
                self.assertTrue(all(p.grad is None for p in self.model.parameters()))
            finally:
                adapter.close()

    def test_matched_parameters_paired_initialization_and_policy_validation(self):
        hashes = []
        for mode, rank in (('query', 32), ('gate', 32), ('query_gate', 16)):
            torch.manual_seed(1701)
            adapter = ReadoutAdapter(self.model, mode, rank)
            try:
                self.assertEqual(adapter.description()['parameters'], 512)
                actual_shapes = expected_parameters(mode, rank)
                self.assertEqual(sum(math.prod(v) for v in actual_shapes.values()), 131072)
                if mode != 'query_gate':
                    hashes.append(next(iter(adapter.branches.values()))['a'].weight.detach().clone())
                with self.assertRaises(ValueError): adapter.set_policy(True, -1)
                with self.assertRaises(ValueError): adapter.set_policy('question', 0)
                with self.assertRaises(TypeError): adapter.set_policy(True, 0, answer='123')
            finally:
                adapter.close()
        self.assertTrue(torch.equal(*hashes))

    def test_constant_frozen_affines_match_old_profile_with_global_grad(self):
        module = nn.Sequential(nn.Linear(8, 4, bias=False), nn.SiLU(), nn.Linear(4, 2, bias=False)).requires_grad_(False)
        x = torch.randn(1, 27, 8)
        old = StableAdapterLinears(module)
        with torch.inference_mode():
            expected = module(x).clone()
        old.close()
        current = ConstantInputFrozenAffines(module)
        try:
            with torch.enable_grad():
                actual = module(x)
                self.assertFalse(actual.requires_grad)
                self.assertTrue(bits_equal(actual, expected))
                with self.assertRaises(ValueError): module(x.clone().requires_grad_(True))
        finally:
            current.close()
        self.assertTrue(all(not v._forward_hooks for v in module.modules()))

    def test_all48_inference_cache_fields_are_regular_and_isolated(self):
        with torch.inference_mode():
            original = toy_cache()
        clone = regular_cache_clone(original)
        before = dict(cache_tensors(original))
        after = dict(cache_tensors(clone))
        self.assertEqual(len(after), 48)
        for key, value in after.items():
            self.assertFalse(torch.is_inference(value))
            self.assertNotEqual(value.untyped_storage().data_ptr(), before[key].untyped_storage().data_ptr())
            self.assertTrue(torch.equal(value, before[key]))
            value.add_(1)
            self.assertFalse(torch.equal(value, before[key]))

    def test_causal_interface_rejects_checkpointing_and_unfrozen_backbone(self):
        self.model.requires_grad_(True)
        with self.assertRaises(ValueError): ReadoutAdapter(self.model, 'query', 4)
        self.model.requires_grad_(False)
        self.model.is_gradient_checkpointing = True
        with self.assertRaises(ValueError): ReadoutAdapter(self.model, 'query', 4)
        self.assertFalse(self.model._forward_pre_hooks)

    def test_all_observed_fit_inputs_and_old_errors_without_model_loading(self):
        with (patch('sas_memory.runtime.TextRuntime.__init__', side_effect=AssertionError('No model construction')),
              patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU initialization'))):
            p, m, cases, registry, queries, rows = load_sources()
            examples = build_examples(tokenizer_runtime(p['model_path']), cases, registry, queries)
            self.assertEqual(len(sources()), 276)
        self.assertEqual([e['id'] for e in examples], list(queries))
        self.assertEqual(len(examples), 1728)
        self.assertEqual(sum(not r['greedy_exact'] for r in rows), 95)
        self.assertEqual({len(e['teacher_ids']) for e in examples}, {27})
        self.assertEqual({len(e['answer_ids']) for e in examples}, {4})
        self.assertEqual(len({e['context_id'] for e in examples}), 288)
        self.assertEqual(p['confirmation']['set_seeds'], list(range(88000, 88008)))
        self.assertEqual(p['confirmation']['copy_seeds'], list(range(89000, 89008)))


if __name__ == '__main__':
    unittest.main()
