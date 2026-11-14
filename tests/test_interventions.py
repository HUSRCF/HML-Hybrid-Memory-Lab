from types import SimpleNamespace
import unittest

import torch

from sas_memory.interventions import transplant
from sas_memory.runtime import assert_cache_equal, fork_cache


class FakeCache:
    def __init__(self, value, length=5):
        self.length = length
        self.layers = [SimpleNamespace(
            conv_states={0: torch.full((1, 2, 4), value)},
            recurrent_states={0: torch.full((1, 2, 3, 3), value)},
            is_conv_states_initialized={0: True}, is_recurrent_states_initialized={0: True},
            has_previous_state={0: True}, conv_kernel_size={0: 4}) for _ in range(3)]
        self.layers.append(SimpleNamespace(keys=torch.full((1, 2, length, 3), value),
                                           values=torch.full((1, 2, length, 3), value)))

    def get_seq_length(self):
        return self.length


class TransplantTests(unittest.TestCase):
    def test_paths_are_separate_and_sources_immutable(self):
        recipient, donor = FakeCache(1.0), FakeCache(2.0)
        saved_recipient, saved_donor = fork_cache(recipient), fork_cache(donor)
        layer_types = ["linear_attention"] * 3 + ["full_attention"]
        for path in ("recurrent", "kv", "both"):
            result = transplant(recipient, donor, layer_types, [0], path)
            expected_recurrent = 2.0 if path != "kv" else 1.0
            expected_kv = 2.0 if path != "recurrent" else 1.0
            self.assertEqual(result.layers[0].conv_states[0].mean().item(), expected_recurrent)
            self.assertEqual(result.layers[0].recurrent_states[0].mean().item(), expected_recurrent)
            self.assertEqual(result.layers[3].keys.mean().item(), expected_kv)
            result.layers[0].conv_states[0].zero_()
            assert_cache_equal(recipient, saved_recipient)
            assert_cache_equal(donor, saved_donor)
        with self.assertRaises(ValueError):
            transplant(recipient, FakeCache(2.0, length=6), layer_types, [0], "both")


if __name__ == "__main__":
    unittest.main()
