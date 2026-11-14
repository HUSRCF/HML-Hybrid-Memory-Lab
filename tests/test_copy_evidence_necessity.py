"""Whole-record semantics, actual token controls and non-vacuous necessity gates."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from sas_memory.tasks import Operation
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.analyze_copy_evidence_necessity import audit_proofs, audit_rows
from scripts.copy_necessity_data import MODELS, REGIMES, ROLES, build_masks, choose_records, event_token_spans, read_plan, sample_ids, selected_inputs
from scripts.copy_necessity_metrics import necessity_checks, summarize
from scripts.run_copy_dependency_transfer import actual_prompt


class WholeRecordNecessityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read_plan()
        cls.source_plan = json.loads(Path(cls.plan["source_transfer_plan"]).read_text())
        source = Path(cls.plan["source_dataset"])
        cls.cases = json.loads((source / "cases.json").read_text())
        cls.registry = json.loads((source / "registry.json").read_text())
        cls.runtime = tokenizer_runtime(cls.plan["model_path"])
        cls.masks = build_masks(cls.cases, cls.registry, cls.runtime)

    def test_every_actual_mask_removes_one_whole_matched_record(self):
        self.assertEqual(len(self.masks), 336)
        self.assertEqual({m["removed_record_tokens"] for m in self.masks.values()}, {17})
        for m in self.masks.values():
            self.assertTrue(m["support_leaf_symbolically_necessary"])
            self.assertIn(m["support_event_index"], m["oracle_event_indices"])
            self.assertNotIn(m["other_event_index"], m["oracle_event_indices"])
            for regime, span in zip(REGIMES, (m["support_token_span"], m["other_token_span"])):
                keep = m["keep_bits"][regime]
                self.assertEqual([i for i, bit in enumerate(keep) if bit == "0"], list(range(*span)))
                self.assertEqual(keep[:m["prelude_tokens"]], "1" * m["prelude_tokens"])
                self.assertEqual(len(keep), m["ledger_tokens"])
                self.assertEqual(keep.count("1") - m["prelude_tokens"], m["retained_record_tokens"])
            self.assertTrue(set(range(*m["support_token_span"])).isdisjoint(range(*m["other_token_span"])))
        smoke, _ = selected_inputs(self.registry, True)
        self.assertEqual(len(smoke), 18)
        self.assertEqual(len(sample_ids(self.cases, self.registry)), 10)
        self.assertTrue(set(sample_ids(self.cases, self.registry)) <= {q["id"] for q in smoke})

    def test_nearest_control_tie_semantic_scope_and_equal_width(self):
        events = [Operation("SET", "X", 100), Operation("SET", "A", 200), Operation("SET", "Y", 300),
                  Operation("SET", "A", 400), Operation("COPY", "T", source="A")]
        spans = [[85, 102], [102, 119], [119, 136], [136, 153], [153, 174]]
        self.assertEqual(choose_records(events, [1, 4], spans), (1, 0, ["A", "T"]))
        wrong = deepcopy(spans); wrong[0][1] -= 1
        self.assertEqual(choose_records(events, [1, 4], wrong)[1], 2)
        wrong[2][1] -= 1
        with self.assertRaises(ValueError): choose_records(events, [1, 4], wrong)
        with self.assertRaises(ValueError): choose_records(events, [0, 1, 4], spans)

    def test_actual_offset_crossing_record_boundary_is_rejected(self):
        q = self.registry["physical"][0]
        case = next(c for c in self.cases if c["id"] == q["representative"]["case_id"])
        events = [Operation(**e) for e in case["prefix"]]
        ledger = actual_prompt(self.runtime, {c["id"]: c for c in self.cases}, q)
        prompt = ledger + self.runtime.query_suffix(q["question"])
        spans = event_token_spans(self.runtime, ledger, prompt, events, q["prelude_tokens"])
        def crossing(*args, **kwargs):
            result = self.runtime.tokenizer(*args, **kwargs)
            t = spans[0][1] - 1
            left, right = result["offset_mapping"][t]
            result["offset_mapping"][t] = (left, right + 1)
            return result
        proxy = SimpleNamespace(prompt_start=self.runtime.prompt_start, ids=self.runtime.ids, tokenizer=crossing)
        with self.assertRaises(ValueError): event_token_spans(proxy, ledger, prompt, events, q["prelude_tokens"])

    def test_control_preservation_and_stream_coverage_required(self):
        def stats(current=40, history=96, unrelated=96, before=48):
            return {"before_all_correct": before, "role_variant_counts": {r: {
                "control": {"correct": n // 2}, "treatment": {"correct": n - n // 2}}
                for r, n in ((ROLES[0], current), (ROLES[2], history), (ROLES[1], unrelated))}}
        contrast = {"signed_net_gain": 10, "positive_base_streams": list("abcd")}
        self.assertTrue(all(necessity_checks(stats(), stats(), contrast).values()))
        for control in (stats(39), stats(history=95), stats(unrelated=95), stats(before=47)):
            self.assertFalse(all(necessity_checks(stats(), control, contrast).values()))
        for effect in ({**contrast, "signed_net_gain": 0}, {**contrast, "positive_base_streams": list("abc")}):
            self.assertFalse(all(necessity_checks(stats(), stats(), effect).values()))

    def test_complete_summary_accepts_harm_and_rejects_no_necessity(self):
        perfect = [{"id": q["id"], "greedy_exact": True, "answer_nll": .01} for q in self.registry["physical"]]
        full = {m: deepcopy(perfect) for m in MODELS}
        rows = {m: {r: deepcopy(perfect) for r in REGIMES} for m in MODELS}
        _, bindings = selected_inputs(self.registry)
        targets = {b["physical_id"] for b in bindings if b["variant"] != "before" and b["role"] == ROLES[0]}
        for m in MODELS:
            for row in rows[m]["remove_support"]:
                if row["id"] in targets: row.update(greedy_exact=False, answer_nll=2.)
        result = summarize(rows, full, self.cases, self.registry, self.source_plan, self.plan)
        self.assertTrue(result["whole_record_necessity"]["passed"])
        self.assertEqual(result["support_removal_effects"]["local_question"]["gains"], 96)
        same = {m: {r: deepcopy(perfect) for r in REGIMES} for m in MODELS}
        result = summarize(same, full, self.cases, self.registry, self.source_plan, self.plan)
        self.assertFalse(result["whole_record_necessity"]["passed"])
        self.assertTrue(result["prior_sufficiency_and_address_gates_unchanged_passed"])
        self.assertTrue(result["prior_shared_history_repeat_primaries_remain_failed"])

    def test_grading_requires_exact_removal_and_complete_inputs(self):
        physical, _ = selected_inputs(self.registry, True)
        regime = REGIMES[0]
        rows = [{"id": q["id"], "model": "native", "regime": regime, "token_policy": "none", "prefill": "full",
            "registration_sha256": data_fingerprint(q), "mask_registration_sha256": data_fingerprint(self.masks[q["id"]]),
            "keep_bits_sha256": data_fingerprint(self.masks[q["id"]]["keep_bits"][regime]), "answer": q["answer"],
            "generated": q["answer"], "greedy_exact": True, "answer_nll": .1, "query_tokens": q["question_tokens"],
            "hook_calls": {"prefill": 1, "cached": 1}} for q in physical]
        audit_rows(rows, physical, self.masks, "native", regime, "none")
        wrong = deepcopy(rows); wrong[0]["keep_bits_sha256"] = "bad"
        for bad in (wrong, rows[:-1], rows[::-1]):
            with self.assertRaises(ValueError): audit_rows(bad, physical, self.masks, "native", regime, "none")

    def test_both_new_regimes_require_complete_memory_proofs(self):
        from tests.test_copy_readout_sufficiency import ReadoutSufficiencyTests
        p, full, rows = ReadoutSufficiencyTests().proof_fixture()
        for field in ("unmasked_weights", "regimes"):
            p[field] = {new: value for new, value in zip(REGIMES, p[field].values())}
        rows = {new: value for new, value in zip(REGIMES, rows.values())}
        audit_proofs([p], ["q"], full, rows)
        wrong = deepcopy(p); wrong["regimes"][REGIMES[1]]["masked_prefix_sha256"] = "b" * 64
        with self.assertRaises(ValueError): audit_proofs([wrong], ["q"], full, rows)
        with self.assertRaises(ValueError): audit_proofs([], ["q"], full, rows)


if __name__ == "__main__":
    unittest.main()
