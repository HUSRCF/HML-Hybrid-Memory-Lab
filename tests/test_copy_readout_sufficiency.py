"""Actual-token mask budgets, observed access and non-vacuous recovery decisions."""

from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from sas_memory.kv_readout import QuestionKVReadout
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.analyze_copy_readout_sufficiency import audit_proofs, audit_rows
from scripts.copy_readout_data import MODELS, REGIMES, build_masks, nonslice_keep, read_plan, sample_ids, selected_inputs
from scripts.copy_readout_metrics import sufficiency_checks, summarize
from scripts.copy_readout_runtime import SelectionWeights, check_weight_stats, readout


class FakeAttention(torch.nn.Module):
    def forward(self, weights):
        return None, weights


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)
        self.config = SimpleNamespace(_attn_implementation="eager", layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6)
        self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=FakeAttention()) for _ in range(24)])
        self.bad_layout = False

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, past_key_values=None, attention_mask=None):
        past, length = (past_key_values.get_seq_length() if past_key_values else 0), input_ids.shape[1]
        q, k = torch.arange(past, past + length), torch.arange(past + length)
        bias = torch.zeros(length, past + length).masked_fill(k[None, :] > q[:, None], torch.finfo(torch.float32).min)[None, None]
        if attention_mask is not None:
            bias = attention_mask["full_attention"]
        weights = bias.softmax(-1).expand(1, 2, length, past + length)
        if self.bad_layout:
            weights = weights[0]
        for index, layer in enumerate(self.model.layers):
            if self.config.layer_types[index] == "full_attention":
                layer.self_attn(weights)


class ReadoutSufficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read_plan()
        cls.source_plan = json.loads(Path(cls.plan["source_transfer_plan"]).read_text())
        source = Path(cls.plan["source_dataset"])
        cls.cases = json.loads((source / "cases.json").read_text())
        cls.registry = json.loads((source / "registry.json").read_text())
        cls.masks = build_masks(cls.cases, cls.registry, tokenizer_runtime(cls.plan["model_path"]))

    def test_all_actual_masks_and_equal_budgets(self):
        self.assertEqual(len(self.masks), 336)
        budgets = set()
        for mask in self.masks.values():
            p, n = mask["prelude_tokens"], mask["retained_record_tokens"]
            left, right = [mask["keep_bits"][r] for r in REGIMES]
            self.assertEqual(len(left), mask["ledger_tokens"])
            self.assertEqual(len(left), len(right))
            self.assertEqual(left[:p], "1" * p)
            self.assertEqual(right[:p], "1" * p)
            self.assertEqual(left.count("1"), right.count("1"))
            self.assertEqual(left.count("1") - p, n)
            self.assertTrue(all(a == "0" or b == "0" for a, b in zip(left[p:], right[p:])))
            self.assertEqual([b == "1" for b in right], nonslice_keep([b == "1" for b in left], p))
            budgets.add(n)
        self.assertEqual(len(budgets), 4)

    def test_nearby_nonslice_control_deterministic_and_valid(self):
        # Four kept record tokens, with ties resolved toward earlier positions.
        oracle = [True, True, False, False, True, True, True, True, False, False]
        self.assertEqual(nonslice_keep(oracle, 2), [True, True, True, True, False, False, False, False, True, True])
        for bad, prelude in (([True] * 5, 1), ([True, True, True, False], 1), ([False, True, False], 1), ([1, 0, 1, 0], 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                nonslice_keep(bad, prelude)

    def test_prescribed_samples_cover_roles_depths_and_orders(self):
        samples = sample_ids(self.cases, self.registry)
        physical, bindings = selected_inputs(self.registry, True)
        self.assertEqual((len(physical), len(bindings), len(samples)), (18, 18, 10))
        self.assertTrue(set(samples) <= {q["id"] for q in physical})
        selected = [b for b in bindings if b["physical_id"] in samples]
        self.assertEqual({b["role"] for b in selected}, {"target_current", "protected_current", "target_initial"})
        self.assertEqual({b["variant"] for b in selected}, {"before", "control", "treatment"})

    def test_actual_six_layer_probabilities_and_cached_masking(self):
        model, keep = FakeModel(), [True, False, True, False]
        for masked in (False, True):
            with (QuestionKVReadout(model, keep) if masked else nullcontext()):
                with SelectionWeights(model, {"test": keep}, 1) as observer:
                    model(input_ids=torch.zeros(1, 7, dtype=torch.long))
                    model(input_ids=torch.zeros(1, 1, dtype=torch.long), past_key_values=SimpleNamespace(get_seq_length=lambda: 7))
            check_weight_stats(observer.stats["test"], masked)
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertTrue(all(not layer.self_attn._forward_hooks for layer in model.model.layers))

    def test_observer_failure_removes_all_hooks(self):
        model = FakeModel(); model.bad_layout = True
        with self.assertRaises(ValueError):
            with QuestionKVReadout(model, [True, False, True, False]):
                with SelectionWeights(model, {"oracle": [True, False, True, False]}, 1):
                    model(input_ids=torch.zeros(1, 7, dtype=torch.long))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertTrue(all(not layer.self_attn._forward_hooks for layer in model.model.layers))
        with self.assertRaises(ValueError):
            SelectionWeights(model, {}, 1)

    def proof_fixture(self):
        row = {"answer": "123", "generated": "123", "greedy_exact": True, "answer_nll": .1, "query_tokens": 4,
               "hook_calls": {"prefill": 1, "cached": 1}}
        def weights(masked):
            return {i: {"prefill_calls": 1, "cached_calls": 1, "query_rows": 4,
                "max_suppressed_probability": 0. if masked else .1, "max_suppressed_mass": 0. if masked else .3,
                "max_kept_record_mass": .5} for i in (3, 7, 11, 15, 19, 23)}
        p = {"id": "q", "unmasked_reference_exact": True, "unmasked_readout": readout(row),
            "all_history_cache_logits_readout_exact": True, "all_history_hook_calls": {"prefill": 2, "cached": 1},
            "hooks_restored": True, "unmasked_weights": {r: weights(False) for r in REGIMES},
            "regimes": {r: {"complete_prefix_memory_exact": True, "prefix_logits_exact": True, "historical_kv_fields_exact": 12,
                "prefix_sha256": "a" * 64, "masked_prefix_sha256": "a" * 64, "proof_hook_calls": {"prefill": 2, "cached": 0},
                "readout": readout(row), "measure_hook_calls": row["hook_calls"], "attention_stats": weights(True)} for r in REGIMES}}
        return p, {"q": row}, {r: {"q": row} for r in REGIMES}

    def test_independent_proof_audit_rejects_memory_or_probability_drift(self):
        p, full, rows = self.proof_fixture()
        audit_proofs([p], ["q"], full, rows)
        changes = [("complete_prefix_memory_exact", False), ("historical_kv_fields_exact", 10),
                   ("masked_prefix_sha256", "b" * 64)]
        for field, value in changes:
            wrong = deepcopy(p); wrong["regimes"]["oracle"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_proofs([wrong], ["q"], full, rows)
        wrong = deepcopy(p); wrong["regimes"]["oracle"]["attention_stats"][3]["max_suppressed_probability"] = .001
        with self.assertRaises(ValueError): audit_proofs([wrong], ["q"], full, rows)
        wrong = deepcopy(p); wrong["unmasked_weights"]["oracle"][3]["max_suppressed_mass"] = 0.
        with self.assertRaises(ValueError): audit_proofs([wrong], ["q"], full, rows)
        with self.assertRaises(ValueError): audit_proofs([], ["q"], full, rows)

    def test_preservation_floor_losses_and_stream_coverage_required(self):
        def stats(current, historical=96, unrelated=96):
            return {"role_variant_counts": {r: {"control": {"correct": n // 2}, "treatment": {"correct": n - n // 2}}
                for r, n in (("target_current", current), ("target_initial", historical), ("protected_current", unrelated))}}
        comp = {"signed_net_gain": 50, "positive_base_streams": [str(i) for i in range(4)]}
        self.assertTrue(all(sufficiency_checks(stats(40), stats(90), comp).values()))
        for oracle in (stats(89), stats(90, 95), stats(90, unrelated=95)):
            self.assertFalse(all(sufficiency_checks(stats(40), oracle, comp).values()))
        for comparison in ({**comp, "signed_net_gain": 0}, {**comp, "positive_base_streams": ["one", "two", "three"]}):
            self.assertFalse(all(sufficiency_checks(stats(40), stats(96), comparison).values()))

    def test_sufficiency_and_address_specificity_are_separate(self):
        source_run = Path(self.plan["source_run"])
        full = {m: [json.loads(s) for s in (source_run / (m + ".jsonl")).read_text().splitlines()] for m in MODELS}
        perfect = [{"id": q["id"], "greedy_exact": True, "answer_nll": .01} for q in self.registry["physical"]]
        rows = {m: {r: deepcopy(perfect) for r in REGIMES} for m in MODELS}
        result = summarize(rows, full, self.cases, self.registry, self.source_plan, self.plan)
        self.assertTrue(result["frozen_prefix_sufficiency"]["passed"])
        self.assertFalse(result["address_specific_gain"]["passed"])
        self.assertTrue(result["prior_shared_history_repeat_primaries_remain_failed"])

    def test_independent_grading_rejects_wrong_masks_and_missing_rows(self):
        physical, _ = selected_inputs(self.registry, True)
        rows = [{"id": q["id"], "model": "native", "regime": "oracle", "token_policy": "none", "prefill": "full",
            "registration_sha256": data_fingerprint(q), "mask_registration_sha256": data_fingerprint(self.masks[q["id"]]),
            "keep_bits_sha256": data_fingerprint(self.masks[q["id"]]["keep_bits"]["oracle"]),
            "answer": q["answer"], "generated": q["answer"], "greedy_exact": True, "answer_nll": .1,
            "query_tokens": q["question_tokens"], "hook_calls": {"prefill": 1, "cached": 1}} for q in physical]
        audit_rows(rows, physical, self.masks, "native", "oracle", "none")
        wrong = deepcopy(rows); wrong[0]["keep_bits_sha256"] = "bad"
        for bad in (wrong, rows[:-1], rows[::-1]):
            with self.assertRaises(ValueError): audit_rows(bad, physical, self.masks, "native", "oracle", "none")


if __name__ == "__main__":
    unittest.main()
