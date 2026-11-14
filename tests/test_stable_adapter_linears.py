"""Cancellation, cleanup and freeze controls for the new diagnostic profile."""

import unittest
import torch
from scripts.stable_adapter_linears import StableAdapterLinears


class StableAdapterTests(unittest.TestCase):
    def test_cancellation_reference_parameter_identity_and_cleanup(self):
        module = torch.nn.Sequential(torch.nn.Linear(3, 1, bias=False)).requires_grad_(False)
        module[0].weight.copy_(torch.tensor([[1., 1., 1.]]))
        identity, before = module[0].weight, module[0].weight.clone()
        inputs = torch.tensor([[1e8, 1., -1e8]])
        with torch.inference_mode():
            native = module(inputs)
            with StableAdapterLinears(module) as arithmetic:
                self.assertEqual(module(inputs).item(), 1.)
                self.assertEqual(arithmetic.calls, {"0": 1})
                self.assertIs(module[0].weight, identity)
                self.assertTrue(torch.equal(module[0].weight, before))
            self.assertTrue(torch.equal(module(inputs), native))
        self.assertFalse(module[0]._forward_hooks)
        self.assertIsNone(module[0].weight.grad)

    def test_rejects_training_and_noncanonical_adapters(self):
        with self.assertRaises(ValueError): StableAdapterLinears(torch.nn.Linear(3, 1, bias=False))
        with self.assertRaises(ValueError): StableAdapterLinears(torch.nn.Linear(3, 1).requires_grad_(False))
        module = torch.nn.Linear(3, 1, bias=False).requires_grad_(False)
        with self.assertRaises(ValueError):
            with StableAdapterLinears(module): module(torch.ones(1, 3))
        self.assertFalse(module._forward_hooks)


if __name__ == "__main__":
    unittest.main()
