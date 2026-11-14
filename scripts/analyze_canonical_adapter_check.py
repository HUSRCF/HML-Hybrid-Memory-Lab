"""Audit the bounded arithmetic check; never substitute it for scientific smoke."""

from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.check_canonical_adapter_readout import ADDENDUM, ADDENDUM_SHA256, OUTPUT
from scripts.copy_necessity_data import registered_inputs, safety_samples, read_plan
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import write_json


def audit():
    report = json.loads(OUTPUT.read_text())
    registered, source_plan, cases, registry, masks, full_rows = registered_inputs()
    plan = read_plan()
    addendum = json.loads(ADDENDUM.read_text())
    queries = [q for q in registry["physical"] if q["id"] in safety_samples(registry)]
    query_map = {q["id"]: q for q in queries}
    names = ["attention_only", "local_question", "gdn_question_only"]
    refs = {m: {r["id"]: r for r in rows} for m, rows in full_rows.items()}
    expected_keys = [(order, m, q["id"], repeat) for order, seq in enumerate((names, names[::-1]))
                     for m in seq for q in queries for repeat in range(3)]
    weights = report["weights_before"]
    if (report["status"] != "completed_arithmetic_diagnostic_only" or report["training"] or
        streaming_hash(ADDENDUM) != ADDENDUM_SHA256 or report["precision_addendum_sha256"] != ADDENDUM_SHA256 or
        report["profile"] != addendum["profile"] or report["query_ids"] != list(query_map) or
        report["orders"] != [names, names[::-1]] or report["weights_after"] != weights or
        weights["sha256"] != plan["native_loaded_weights_sha256"] or
        not weights["all_parameters_frozen"] or not weights["no_parameter_gradients"] or
        [(r["order"], r["model"], r["id"], r["repeat"]) for r in report["records"]] != expected_keys or
        streaming_hash(OUTPUT.with_suffix(".source.py")) != report["source_sha256"]["scripts/check_canonical_adapter_readout.py"] or
        streaming_hash(ROOT / addendum["profile_source"]) != addendum["profile_source_sha256"] or
        not datetime.fromisoformat(addendum["registered_utc"]) < datetime.fromisoformat(report["started_utc"]) < datetime.fromisoformat(report["completed_utc"])):
        raise ValueError("Actual canonical arithmetic check scope, execution or frozen state changed")
    for path, digest in report["source_sha256"].items():
        if streaming_hash(ROOT / path) != digest:
            raise ValueError("Actual arithmetic-check source changed")
    first = {}
    changed_legacy = set()
    for row in report["records"]:
        q, actual = query_map[row["id"]], row["actual"]
        match = re.fullmatch(r"\s*(-?\d+)\s*", actual["generated"])
        if (actual["answer"] != q["answer"] or actual["query_tokens"] != q["question_tokens"] or
            actual["greedy_exact"] != bool(match and match.group(1) == q["answer"]) or
            row["legacy_fp32"] != readout(refs[row["model"]][row["id"]])):
            raise ValueError("Canonical unrestricted output or labeled legacy comparison changed")
        key = (row["model"], row["id"])
        if key in first and first[key] != actual:
            raise ValueError("Canonical repeats/cross-order outputs are not exact")
        first[key] = actual
        if actual != row["legacy_fp32"]:
            changed_legacy.add(key)
    expected_arms = [(o, m) for o, seq in enumerate((names, names[::-1])) for m in seq]
    if [(r["order"], r["model"]) for r in report["adapter_checks"]] != expected_arms:
        raise ValueError("Incomplete actual adapter coverage")
    for r in report["adapter_checks"]:
        expected = registered["references"]["training"][r["model"]]["checkpoint_state_sha256"]
        if (r["before_sha256"] != expected or r["after_sha256"] != expected or
            r["arithmetic"]["profile"] != addendum["profile"] or
            r["arithmetic"]["linear_modules"] != list(r["actual_calls"]) or
            any(v < 1 for v in r["actual_calls"].values()) or
            r["arithmetic"]["parameter_storage_and_dtype_changed"] or r["arithmetic"]["backbone_arithmetic_changed"]):
            raise ValueError("Frozen adapter bytes or canonical affine scope changed")
    prefixes = {}
    expected_prefixes = [(o, m, q["source_key"]) for o, seq in enumerate((names, names[::-1]))
                         for m in seq if m != "attention_only" for q in queries]
    if [(p["order"], p["model"], p["source_key"]) for p in report["prefix_checks"]] != expected_prefixes:
        raise ValueError("Complete24native-prefix checks required")
    for p in report["prefix_checks"]:
        if not p["complete_KV_recurrent_conv_bookkeeping_exact"] or not p["last_prefix_logits_exact"]:
            raise ValueError("Canonical question-only controller changed native prefix")
        key = (p["model"], p["source_key"])
        if key in prefixes and prefixes[key] != p["prefix_sha256"]:
            raise ValueError("Canonical native prefix differs across model orders")
        prefixes[key] = p["prefix_sha256"]
    if [(r["order"], r["after_model"]) for r in report["native_replays"]] != expected_arms or any(not r["exact"] for r in report["native_replays"]):
        raise ValueError("Complete detached-native replay controls required")
    return {"status": "audited_arithmetic_diagnostic_not_scientific_smoke", "source_report": str(OUTPUT),
        "source_report_sha256": streaming_hash(OUTPUT), "precision_addendum_sha256": ADDENDUM_SHA256,
        "exact_readouts": 108, "unique_model_queries": len(first), "prefix_identity_checks": 24, "native_replays": 6,
        "actual_unchanged_weights_sha256": weights["sha256"], "unique_readouts_different_from_legacy_fp32": len(changed_legacy),
        "required432_query_smokes_completed": False, "canonical8064_primary_readouts_started": False,
        "interpretation": "Only three models and six prescribed queries each. This supports testing the registered numerical profile, not universal determinism, scientific necessity, a new architecture or vendor-library defect.",
        "analysis_source_sha256": streaming_hash(Path(__file__))}


if __name__ == "__main__":
    result = audit()
    destination = OUTPUT.with_name("canonical-adapter-readout-check-audit-20261114-01.json")
    write_json(destination, result)
    destination.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)
