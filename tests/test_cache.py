import unittest
from types import SimpleNamespace

import torch

from sas_memory.runtime import assert_cache_equal, cache_tensors, fork_cache


class MixedCacheTests(unittest.TestCase):
    def test_fork_and_batch_cover_every_state_and_do_not_alias(self):
        source = SimpleNamespace(layers=[
            SimpleNamespace(recurrent_states={0: torch.randn(1, 2, 3, 3)},
                            conv_states={0: torch.randn(1, 6, 4)}, has_previous_state={0: True}),
            SimpleNamespace(keys=torch.randn(1, 2, 5, 3), values=torch.randn(1, 2, 5, 3)),
        ])
        saved = fork_cache(source)
        fork = fork_cache(source, 3)
        self.assertEqual(len(dict(cache_tensors(fork))), 4)
        for key, tensor in cache_tensors(fork):
            self.assertEqual(tensor.shape[0], 3)
            torch.testing.assert_close(tensor[1], dict(cache_tensors(source))[key][0])
            tensor.zero_()
        assert_cache_equal(source, saved)
        self.assertTrue(fork.layers[0].has_previous_state[0])
        source.layers[0].recurrent_states[0].add_(1)
        with self.assertRaises(AssertionError):
            assert_cache_equal(source, saved)


if __name__ == "__main__":
    unittest.main()
