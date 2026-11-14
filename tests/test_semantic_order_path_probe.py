"""Controls for cache provenance and paired four-condition NLL effects."""

from types import SimpleNamespace
import unittest

import torch

from sas_memory.interventions import transplant
from sas_memory.runtime import fork_cache
from scripts.semantic_order_path_common import (CONDITIONS, assert_complete_cache_equal,
    cache_fingerprint, check_origins, nll_contrasts)


class Cache:
    def __init__(self, number):
        self.layers = [SimpleNamespace(conv_states={0:torch.full((1,2,4),number)},
            recurrent_states={0:torch.full((1,2,3,3),number)}, is_conv_states_initialized={0:True},
            is_recurrent_states_initialized={0:True}, has_previous_state={0:True},conv_kernel_size={0:4}) for _ in range(3)]
        self.layers.append(SimpleNamespace(keys=torch.full((1,2,5,3),number),values=torch.full((1,2,5,3),number),is_initialized=True))

    def get_seq_length(self):
        return 5


class NativePathControls(unittest.TestCase):
    def test_complete_cache_guard_detects_bookkeeping_changes(self):
        original = Cache(1.)
        saved = fork_cache(original)
        first = cache_fingerprint(original)
        assert_complete_cache_equal(original,saved)
        original.layers[0].has_previous_state[0] = False
        self.assertNotEqual(cache_fingerprint(original),first)
        with self.assertRaises(AssertionError):
            assert_complete_cache_equal(original,saved)

    def test_four_condition_provenance_covers_conv_recurrent_and_kv(self):
        originals = {"canonical":Cache(1.),"permuted":Cache(2.)}
        types = ["linear_attention"]*3+["full_attention"]
        actual = {"canonical_natural":originals["canonical"],"permuted_natural":originals["permuted"],
            "canonical_state_permuted_kv":transplant(originals["canonical"],originals["permuted"],types,[0],"kv"),
            "permuted_state_canonical_kv":transplant(originals["canonical"],originals["permuted"],types,[0],"recurrent")}
        for name,origins in CONDITIONS.items():
            check_origins(actual[name],originals,origins)
        mixed = actual["canonical_state_permuted_kv"]
        mixed.layers[1].conv_states[0].fill_(2.)
        with self.assertRaises(AssertionError):
            check_origins(mixed,originals,CONDITIONS["canonical_state_permuted_kv"])

    def test_nll_factorial_is_paired_by_identity_with_explicit_sign(self):
        effects = dict(zip(CONDITIONS,(1.,9.,3.,4.)))
        rows = {name:[{"case_id":"a","variant":"control","role":"target_current","answer_nll":loss},
                     {"case_id":"b","variant":"control","role":"target_current","answer_nll":loss+2}]
                for name,loss in effects.items()}
        rows["permuted_natural"].reverse()
        report = nll_contrasts(rows)
        self.assertEqual(report["means"]["interaction_difference_in_differences"],3.)
        self.assertEqual(report["means"]["recurrent_permuted_minus_canonical_at_canonical_kv"],3.)
        self.assertEqual(report["means"]["kv_permuted_minus_canonical_at_canonical_state"],2.)
        rows["permuted_natural"].pop()
        with self.assertRaises(ValueError):
            nll_contrasts(rows)


if __name__ == '__main__':
    unittest.main()
