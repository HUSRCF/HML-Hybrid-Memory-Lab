"""Detect arithmetic-reference mixing, missing outputs and incomplete order controls."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from sas_memory.training_data import data_fingerprint
from scripts.analyze_canonical_copy_necessity import audit_full_rows
from scripts.canonical_necessity_common import (MASKED_REGIMES, REGIMES, audit_affines,
    check_affine_scope, model_order)
from scripts.compare_canonical_necessity_smokes import compare_observations
from scripts.copy_necessity_data import MODELS


class CanonicalNecessityTests(unittest.TestCase):
    def full_fixture(self):
        q = {"id": "q", "answer": "123", "question_tokens": 8}
        row = {"id": "q", "answer": "123", "generated": "123\n", "greedy_exact": True,
               "answer_nll": .1, "query_tokens": 8, "model": "native", "regime": "full",
               "token_policy": "none", "prefill": "full", "registration_sha256": data_fingerprint(q),
               "mask_registration_sha256": None, "keep_bits_sha256": None, "hook_calls": None}
        return q, row

    def test_full_reference_is_own_world_unmasked_and_finite(self):
        q, row = self.full_fixture()
        audit_full_rows([row], [q], "native", "none")
        for field, value in (("answer_nll", float("nan")), ("answer", "124"), ("greedy_exact", False),
                             ("hook_calls", {"prefill": 1}), ("mask_registration_sha256", "old-mask")):
            wrong = {**row, field: value}
            with self.assertRaises(ValueError):
                audit_full_rows([wrong], [q], "native", "none")
        with self.assertRaises(ValueError):
            audit_full_rows([], [q], "native", "none")

    def order_fixture(self):
        _, template = self.full_fixture()
        ids = [f"q{i}" for i in range(18)]
        rows = {m: {r: [{**template, "id": key, "model": m, "regime": r} for key in ids]
                    for r in REGIMES} for m in MODELS}
        first = {"model_order": model_order(), "selected_physical_ids": ids,
                 "native_anchors": [{"id": key, "readout": template} for key in ids[:10]], "models": {}}
        for m in MODELS:
            first["models"][m] = {"proofs": [{"id": key, "regimes": {r: {
                "prefix_sha256": "a" * 64, "masked_prefix_sha256": "a" * 64} for r in MASKED_REGIMES}}
                for key in ids[:10]], "prefix_checks": [{"source_key": key, "prefix_sha256": "b" * 64}
                for key in ids[:6]] if m in ("local_question", "gdn_question_only") else []}
        second = deepcopy(first)
        second["model_order"] = model_order(True)
        return first, second, rows, deepcopy(rows)

    def test_cross_order_checks_every_readout_without_accuracy_filter(self):
        first, second, a, b = self.order_fixture()
        result = compare_observations(first, second, a, b)
        self.assertEqual(result["exact_readouts"], 432)
        self.assertEqual(result["exact_complete_prefix_samples"], 80)
        # Same answer/correctness, one small NLL discrepancy must stop the primary.
        b["attention_only"]["full"][3]["answer_nll"] += 1e-12
        with self.assertRaises(ValueError): compare_observations(first, second, a, b)

    def test_masked_rows_and_complete_prefixes_cannot_be_omitted(self):
        first, second, a, b = self.order_fixture()
        b["gdn_question_only"][MASKED_REGIMES[1]] = b["gdn_question_only"][MASKED_REGIMES[1]][:-1]
        with self.assertRaises(ValueError): compare_observations(first, second, a, b)
        first, second, a, b = self.order_fixture()
        second["models"]["shared"]["proofs"][9]["regimes"][MASKED_REGIMES[1]]["masked_prefix_sha256"] = "c" * 64
        with self.assertRaises(ValueError): compare_observations(first, second, a, b)
        first, second, a, b = self.order_fixture()
        second["models"]["local_question"]["prefix_checks"][5]["prefix_sha256"] = "c" * 64
        with self.assertRaises(ValueError): compare_observations(first, second, a, b)

    def test_both_model_orders_are_required(self):
        first, second, a, b = self.order_fixture()
        second["model_order"] = model_order()
        with self.assertRaises(ValueError): compare_observations(first, second, a, b)

    def controller_fixture(self):
        controller = nn.Module()
        controller.encoders = nn.ModuleDict({"shared": nn.Linear(4, 2, bias=False)})
        controller.outputs = nn.ModuleDict({"attention_q_proj": nn.Linear(2, 4, bias=False)})
        description = {"encoder_shapes": {"shared": [2, 4]}, "output_shapes": {"attention_q_proj": [4, 2]}}
        controller.description = lambda: description
        return controller, description

    def test_native_model_modules_cannot_receive_canonical_hooks(self):
        controller, _ = self.controller_fixture()
        native = nn.Linear(4, 4, bias=False)
        self.assertEqual(check_affine_scope(controller, native)["native_modules_wrapped"], 0)
        native.shared = controller.encoders["shared"]
        with self.assertRaises(ValueError): check_affine_scope(controller, native)

    def test_all_canonical_affines_must_execute_and_keep_storage(self):
        controller, description = self.controller_fixture()
        scope = check_affine_scope(controller, nn.Linear(4, 4, bias=False))
        names = scope["linear_modules"]
        state = {"description": description, "affine_scope": scope,
                 "arithmetic": {"profile": "canonical", "linear_modules": names,
                                "parameter_storage_and_dtype_changed": False, "backbone_arithmetic_changed": False},
                 "actual_affine_calls": {n: 10 for n in names}}
        audit_affines(state, "canonical")
        wrong = deepcopy(state); wrong["actual_affine_calls"][names[1]] = 0
        with self.assertRaises(ValueError): audit_affines(wrong, "canonical")
        wrong = deepcopy(state); wrong["arithmetic"]["parameter_storage_and_dtype_changed"] = True
        with self.assertRaises(ValueError): audit_affines(wrong, "canonical")
        with self.assertRaises(ValueError): audit_affines(state, "legacy_fp32")


if __name__ == "__main__":
    unittest.main()
