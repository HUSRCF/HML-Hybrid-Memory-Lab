from types import SimpleNamespace
import unittest

import torch
from transformers.cache_utils import LinearAttentionLayer

from scripts.readout_adapter_cache_commit import copy_on_write_commit, install_teacher_commits


def recurrent(value=None):
    layer = LinearAttentionLayer()
    layer.update_recurrent_state(torch.arange(1, 5, dtype=torch.float32).reshape(2, 2)
                                 if value is None else value)
    return layer


def cache():
    return SimpleNamespace(layers=[SimpleNamespace(keys=torch.zeros(1), values=torch.zeros(1))
                                  if i % 4 == 3 else recurrent() for i in range(24)])


class CacheCommitCorrectionTests(unittest.TestCase):
    def test_reproduce_native_failure_and_recover_exact_analytical_gradient(self):
        native = recurrent()
        x = torch.tensor([[.2, .4], [.6, .8]], requires_grad=True)
        initial = native.recurrent_states[0]
        loss = (initial * x).square().sum()
        native.update_recurrent_state(initial + 3)
        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
            loss.backward()
        safe = recurrent()
        initial = safe.recurrent_states[0]
        old = initial.clone()
        x = x.detach().clone().requires_grad_(True)
        loss = (initial * x).square().sum()
        final = copy_on_write_commit(safe, initial + x)
        (loss + final.sum()).backward()
        self.assertTrue(torch.equal(x.grad, 2 * old.square() * x.detach() + 1))
        self.assertTrue(torch.equal(initial, old))
        self.assertEqual(initial._version, 1)
        self.assertTrue(final.requires_grad)

    def test_exact_native_values_without_aliasing_and_repeated_saved_states(self):
        native, safe = recurrent(), recurrent()
        old = safe.recurrent_states[0]
        incoming = old + .125
        expected = native.update_recurrent_state(incoming)
        actual = copy_on_write_commit(safe, incoming)
        self.assertTrue(torch.equal(actual, expected))
        self.assertNotEqual(actual.data_ptr(), incoming.data_ptr())
        self.assertNotEqual(actual.data_ptr(), old.data_ptr())
        x = torch.tensor(.7, requires_grad=True)
        initial = safe.recurrent_states[0]
        first = copy_on_write_commit(safe, initial + x)
        loss = (first * x).square().sum()
        copy_on_write_commit(safe, first + 2 * x)
        loss.backward()
        expected_grad = (2 * (initial + x.detach()) * x.detach() * (initial + 2 * x.detach())).sum()
        self.assertTrue(torch.allclose(x.grad, expected_grad, rtol=1e-6, atol=1e-6))

    def test_only18_disposable_instances_change_and_native_class_stays_intact(self):
        native, safe = cache(), cache()
        original = LinearAttentionLayer.update_recurrent_state
        initial = [layer.recurrent_states[0] for layer in safe.layers if hasattr(layer, 'recurrent_states')]
        self.assertIs(install_teacher_commits(safe), safe)
        self.assertIs(LinearAttentionLayer.update_recurrent_state, original)
        for i, layer in enumerate(safe.layers):
            if i % 4 == 3:
                self.assertNotIn('update_recurrent_state', vars(layer))
                continue
            self.assertIn('update_recurrent_state', vars(layer))
            self.assertNotIn('update_recurrent_state', vars(native.layers[i]))
            before = layer.recurrent_states[0]
            layer.update_recurrent_state(before + 2)
            self.assertTrue(torch.equal(before, native.layers[i].recurrent_states[0]))
        self.assertEqual(len(initial), 18)

    def test_reject_inference_uninitialized_incompatible_and_duplicate_policies(self):
        with torch.inference_mode():
            value = torch.ones(2, 2)
        with self.assertRaises(ValueError):
            copy_on_write_commit(recurrent(), value)
        with self.assertRaises(ValueError):
            copy_on_write_commit(LinearAttentionLayer(), torch.ones(2, 2))
        with self.assertRaises(ValueError):
            copy_on_write_commit(recurrent(), torch.ones(3, 2))
        with self.assertRaises(ValueError):
            copy_on_write_commit(recurrent(), torch.ones(2, 2, dtype=torch.float64))
        incomplete = cache()
        incomplete.layers[0] = LinearAttentionLayer()
        with self.assertRaises(ValueError):
            install_teacher_commits(incomplete)
        self.assertTrue(all('update_recurrent_state' not in vars(layer) for layer in incomplete.layers))
        valid = install_teacher_commits(cache())
        with self.assertRaises(ValueError):
            install_teacher_commits(valid)


if __name__ == '__main__':
    unittest.main()
