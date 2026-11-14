"""Protect complete COPY coverage, exact deduplication and paired decision units."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
import zlib

from sas_memory.generalization_tasks import build_generalization_cases
from sas_memory.training_data import data_fingerprint
from scripts.analyze_copy_dependency_transfer import audit_rows
from scripts.copy_transfer_data import (build_registry, read_plan, safety_samples, selected_inputs, validate_cases, MODELS)
from scripts.copy_transfer_metrics import paired_outcomes, summarize


class WordRuntime:
    prompt_start = "Start:\n"

    def ids(self, text):
        return [zlib.crc32(word.encode()) for word in text.split()]

    def query_suffix(self, question):
        return "\nQuestion: " + question + "\nAnswer: "

    def continuation_ids(self, prompt, answer):
        return [int(digit) for digit in answer]


class CopyTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read_plan()
        cls.cases = build_generalization_cases()
        cls.registry = build_registry(cls.cases, WordRuntime())

    def test_preserve_original_cases_and_executable_order(self):
        self.assertEqual(validate_cases(self.cases, self.plan)["cases"], 48)
        for mutation in ("answer", "source", "missing"):
            data = deepcopy(self.cases)
            if mutation == "answer":
                data[0]["variants"]["control"]["queries"][0]["answer"] = "999"
            elif mutation == "source":
                data[0]["variants"]["control"]["tail"][0]["source"] = data[0]["variants"]["control"]["tail"][0]["target"]
            else:
                data.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_cases(data, self.plan)

    def test_deduplicate_only_complete_identical_before_prompts(self):
        physical, bindings = selected_inputs(self.registry)
        self.assertEqual((len(physical), len(bindings)), (336, 432))
        counts = {}
        for b in bindings:
            counts.setdefault(b["physical_id"], []).append(b)
        duplicated = [group for group in counts.values() if len(group) > 1]
        self.assertEqual(len(duplicated), 48)
        self.assertTrue(all(len(g) == 3 and {b["variant"] for b in g} == {"before"} for g in duplicated))
        self.assertEqual(len({b["base_case_id"] for b in bindings}), 8)

    def test_conflicting_gold_for_identical_prompt_is_rejected(self):
        data = deepcopy(self.cases)
        data[2]["variants"]["before"]["queries"][0]["answer"] = "999"
        with self.assertRaises(ValueError):
            build_registry(data, WordRuntime())

    def test_smoke_and_safety_keep_both_orders_and_boundaries(self):
        physical, bindings = selected_inputs(self.registry, True)
        self.assertEqual((len(physical), len(bindings)), (18, 18))
        samples = safety_samples(self.registry)
        self.assertEqual(len(samples), 6)
        self.assertTrue(set(samples) <= {q["id"] for q in physical})
        self.assertEqual({b["variant"] for b in bindings if b["physical_id"] in samples}, {"before", "control", "treatment"})
        self.assertEqual({b["dependency_depth"] for b in bindings}, {1, 3})
        self.assertEqual({b["question_style"] for b in bindings}, {"original", "entity_first"})
        self.assertEqual({b["distractor_updates"] for b in bindings}, {8, 64})

    def outcome_pair(self):
        reference = [{"case_id": c["id"], "base_case_id": c["base_case_id"], "dependency_depth": c["dependency_depth"],
            "question_style": c["question_style"], "joint_after": False} for c in self.cases]
        return deepcopy(reference), reference

    def test_one_stream_many_conditions_cannot_pass(self):
        candidate, reference = self.outcome_pair()
        for c in candidate[:6]:
            c["joint_after"] = True
        r = paired_outcomes(candidate, reference)
        self.assertEqual(r["signed_net_gain"], 6)
        self.assertTrue(r["candidate_gate_checks"]["positive_at_every_depth"])
        self.assertFalse(r["candidate_gate_checks"]["at_least_three_positive_streams"])
        self.assertFalse(r["candidate_passed"])

    def test_losses_cancel_gains_and_every_depth_is_required(self):
        candidate, reference = self.outcome_pair()
        for i in (0, 6, 12):
            candidate[i]["joint_after"] = True
            reference[i+1]["joint_after"] = True
        r = paired_outcomes(candidate, reference)
        self.assertEqual((r["gains"], r["losses"], r["signed_net_gain"]), (3, 3, 0))
        self.assertFalse(r["candidate_passed"])
        candidate, reference = self.outcome_pair()
        for i in (0, 6, 12):
            candidate[i]["joint_after"] = True
        r = paired_outcomes(candidate, reference)
        self.assertTrue(r["candidate_gate_checks"]["at_least_three_positive_streams"])
        self.assertFalse(r["candidate_gate_checks"]["positive_at_every_depth"])

    def test_all_three_control_comparisons_required(self):
        physical, _ = selected_inputs(self.registry)
        all_rows = {name: [{"id": q["id"], "greedy_exact": True, "answer_nll": .1} for q in physical] for name in MODELS}
        result = summarize(all_rows, self.cases, self.registry, self.plan)
        self.assertFalse(result["descriptive_shared_candidate_gate"]["passed"])
        self.assertEqual(result["models"]["native"]["joint_all_nine"], 48)
        self.assertEqual(result["models"]["native"]["binding_correct"], 432)
        positive = deepcopy(all_rows)
        for model in ("independent", "local_question", "gdn_question_only"):
            for row in positive[model]:
                row["greedy_exact"] = False
        self.assertTrue(summarize(positive, self.cases, self.registry, self.plan)["descriptive_shared_candidate_gate"]["passed"])
        del all_rows["independent"]
        with self.assertRaises(ValueError):
            summarize(all_rows, self.cases, self.registry, self.plan)

    def test_independent_grading_rejects_wrong_truth_or_filtered_rows(self):
        physical, _ = selected_inputs(self.registry, True)
        rows = [{"id": q["id"], "model": "native", "prefill": "full", "reader_policy": "full", "token_policy": "none",
            "registration_sha256": data_fingerprint(q), "query_tokens": q["question_tokens"], "answer": q["answer"],
            "generated": q["answer"], "greedy_exact": True, "answer_nll": .1} for q in physical]
        audit_rows(rows, physical, "native", {"token_policy": "none"})
        wrong = deepcopy(rows); wrong[0]["answer"] = "999"
        for bad in (wrong, rows[:-1], rows[::-1]):
            with self.assertRaises(ValueError):
                audit_rows(bad, physical, "native", {"token_policy": "none"})


if __name__ == "__main__":
    unittest.main()
