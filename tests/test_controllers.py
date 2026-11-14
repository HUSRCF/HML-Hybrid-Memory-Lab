from types import SimpleNamespace
import unittest

import torch
from torch import nn

from sas_memory.controllers import GroupController, query_residual


class ToyGDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_a, self.in_proj_b = nn.Linear(8, 2, bias=False), nn.Linear(8, 2, bias=False)
        self.key_dim = 4
        self.in_proj_qkv = nn.Linear(8, 12, bias=False)

    def forward(self, hidden_states):
        return (hidden_states + (self.in_proj_a(hidden_states).sigmoid() + self.in_proj_b(hidden_states).sigmoid()).mean(-1, keepdim=True)
                + self.in_proj_qkv(hidden_states)[..., 4:8].mean(-1, keepdim=True))


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.q_proj, self.k_proj = nn.Linear(8, 8, bias=False), nn.Linear(8, 2, bias=False)

    def forward(self, hidden_states):
        return hidden_states + self.q_proj(hidden_states).mean(-1, keepdim=True) + self.k_proj(hidden_states).mean(-1, keepdim=True)


class ToyLayer(nn.Module):
    def __init__(self, recurrent):
        super().__init__()
        if recurrent:
            self.linear_attn = ToyGDN()
        else:
            self.self_attn = ToyAttention()

    def forward(self, hidden_states):
        return self.linear_attn(hidden_states) if hasattr(self, "linear_attn") else self.self_attn(hidden_states)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, layer_types=["linear_attention"] * 3 + ["full_attention"])
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(ToyLayer(index < 3) for index in range(4))

    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


class ControllerTests(unittest.TestCase):
    def test_query_gate_is_preserved_for_every_head(self):
        output = torch.arange(24.0).reshape(1, 3, 8)
        delta = torch.ones(1, 3, 4)
        changed = query_residual(output, delta, 2).reshape(1, 3, 2, 4)
        original = output.reshape(1, 3, 2, 4)
        torch.testing.assert_close(changed[..., :2], original[..., :2] + 1)
        torch.testing.assert_close(changed[..., 2:], original[..., 2:])

    def test_zero_identity_gradients_and_detach_restore(self):
        for mode in GroupController.MODES:
            torch.manual_seed(42)
            model = ToyModel().requires_grad_(False)
            hidden = torch.randn(1, 5, 8)
            baseline = model(hidden).detach()
            controller = GroupController(model, 0, 3, mode)
            actual = model(hidden)
            self.assertTrue(torch.equal(actual, baseline))
            actual.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(m.weight.grad is not None and torch.isfinite(m.weight.grad).all()
                                and m.weight.grad.abs().sum() > 0 for m in controller.outputs.values()))
            optimizer = torch.optim.AdamW(controller.parameters(), lr=0.01)
            optimizer.step()
            self.assertFalse(torch.equal(model(hidden), baseline))
            controller.set_enabled_outputs([])
            self.assertTrue(torch.equal(model(hidden), baseline))
            controller.set_enabled_outputs(controller.outputs)
            self.assertFalse(torch.equal(model(hidden), baseline))
            controller.close()
            self.assertTrue(torch.equal(model(hidden), baseline))

    def test_gdn_key_adapter_keeps_query_and_value_channels(self):
        torch.manual_seed(7)
        model = ToyModel().requires_grad_(False)
        controller = GroupController(model, 0, 3, "shared", adapt_gdn_keys=True)
        hidden = torch.randn(1, 5, 8)
        controller._context = hidden
        output = model.model.layers[0].linear_attn.in_proj_qkv(hidden)
        with torch.no_grad():
            controller.outputs["gdn_0_key"].weight.fill_(1)
        changed = model.model.layers[0].linear_attn.in_proj_qkv(hidden)
        self.assertTrue(torch.equal(output[..., :4], changed[..., :4]))
        self.assertTrue(torch.equal(output[..., 8:], changed[..., 8:]))
        self.assertFalse(torch.equal(output[..., 4:8], changed[..., 4:8]))
        model(hidden).square().mean().backward()
        self.assertGreater(controller.outputs["gdn_0_key"].weight.grad.abs().sum().item(), 0)
        controller.close()

    def test_temporal_masks_restrict_direct_changes_to_selected_tokens(self):
        torch.manual_seed(23)
        model = ToyModel().requires_grad_(False)
        hidden = torch.randn(1, 5, 8)
        baseline = model(hidden).detach()
        controller = GroupController(model, 0, 3, "shared")
        with torch.no_grad():
            for module in controller.outputs.values():
                module.weight.fill_(0.1)
        controller.set_token_policy("ledger", 3)
        ledger = model(hidden)
        self.assertTrue(torch.equal(ledger[:, 3:], baseline[:, 3:]))
        self.assertFalse(torch.equal(ledger[:, :3], baseline[:, :3]))
        controller.set_token_policy("question", 3)
        question = model(hidden)
        self.assertTrue(torch.equal(question[:, :3], baseline[:, :3]))
        self.assertFalse(torch.equal(question[:, 3:], baseline[:, 3:]))
        controller.set_token_policy("none")
        self.assertTrue(torch.equal(model(hidden), baseline))
        controller.close()


if __name__ == "__main__":
    unittest.main()
