from types import SimpleNamespace
import unittest

import torch
from torch import nn

from sas_memory.gdn_query_control import GDNQueryController


class QueryToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, layer_types=["linear_attention"] * 3 + ["full_attention"])
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(3):
            layer = nn.Module()
            layer.linear_attn = nn.Module()
            layer.linear_attn.key_dim, layer.linear_attn.value_dim = 4, 4
            layer.linear_attn.in_proj_qkv = nn.Linear(8, 12, bias=False)
            self.model.layers.append(layer)
        self.model.layers.append(nn.Identity())

    def forward(self, hidden, past_key_values=None):
        for layer in self.model.layers[:3]:
            packed = layer.linear_attn.in_proj_qkv(hidden)
            hidden = hidden + packed[..., :4].mean(-1, keepdim=True)
        return hidden


class GDNQueryControlTests(unittest.TestCase):
    def test_identity_learning_prefix_and_direct_write_channel_preservation(self):
        torch.manual_seed(61)
        model = QueryToyModel().requires_grad_(False)
        hidden = torch.randn(1, 5, 8)
        native = model(hidden).detach()
        controller = GDNQueryController(model, 0, 3)
        controller.set_token_policy("question", 3)
        self.assertEqual(controller.description()["trainable_parameters"], 3 * 3 * (8 + 4))
        actual = model(hidden)
        self.assertTrue(torch.equal(actual, native))
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(module.weight.grad.norm() > 0 for module in controller.outputs.values()))
        optimizer = torch.optim.AdamW(controller.parameters(), lr=.02, weight_decay=0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        changed = model(hidden)
        self.assertTrue(torch.equal(changed[:, :3], native[:, :3]))
        self.assertFalse(torch.equal(changed[:, 3:], native[:, 3:]))
        changed.square().mean().backward()
        self.assertTrue(all(module.weight.grad.norm() > 0 for module in controller.encoders.values()))
        projection = model.model.layers[0].linear_attn.in_proj_qkv
        controller.set_token_policy("none")
        packed_native = projection(hidden).detach()
        controller.set_token_policy("question", 3)
        packed_changed = projection(hidden)
        self.assertTrue(torch.equal(packed_changed[..., 4:], packed_native[..., 4:]))
        self.assertTrue(torch.equal(packed_changed[:, :3], packed_native[:, :3]))
        self.assertFalse(torch.equal(packed_changed[:, 3:, :4], packed_native[:, 3:, :4]))
        controller.close()
        self.assertTrue(torch.equal(model(hidden), native))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertTrue(all(len(layer.linear_attn.in_proj_qkv._forward_hooks) == 0 for layer in model.model.layers[:3]))

    def test_cached_offsets_and_invalid_scope(self):
        torch.manual_seed(17)
        model = QueryToyModel().requires_grad_(False)
        hidden = torch.randn(1, 1, 8)
        native = model(hidden).detach()
        controller = GDNQueryController(model, 0, 3)
        with torch.no_grad():
            for module in controller.outputs.values():
                module.weight.fill_(.5)
        controller.set_token_policy("question", 3)
        self.assertTrue(torch.equal(model(hidden, past_key_values=SimpleNamespace(get_seq_length=lambda: 2)), native))
        self.assertFalse(torch.equal(model(hidden, past_key_values=SimpleNamespace(get_seq_length=lambda: 3)), native))
        controller.set_enabled_outputs([])
        self.assertTrue(torch.equal(model(hidden, past_key_values=SimpleNamespace(get_seq_length=lambda: 3)), native))
        with self.assertRaises(ValueError):
            controller.set_token_policy("all")
        controller.close()


if __name__ == "__main__":
    unittest.main()
