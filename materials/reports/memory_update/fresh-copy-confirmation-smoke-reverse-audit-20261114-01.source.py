"""Independent fixed fresh-cohort natural controls and three-mask readout audit."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_copy_evidence_necessity import audit_rows
from scripts.canonical_necessity_common import ADDENDUM_SHA256, audit_affines
from scripts.fresh_copy_data import (MODELS, PRIMARY, PROTOCOL, MASKED_REGIMES, read_plan, registered_inputs,
    sample_ids, safety_samples, selected_inputs, model_order, sources)
from scripts.fresh_copy_runtime import verify_prior_index, checked_smoke_equivalence
from scripts.fresh_copy_metrics import summarize
from scripts.copy_readout_runtime import readout, check_weight_stats
from scripts.run_controller_trial import write_json


def audit_readout(actual, query):
    match = re.fullmatch(r"\s*(-?\d+)\s*", actual["generated"])
    if (actual["answer"] != query["answer"] or actual["query_tokens"] != query["question_tokens"] or
        actual["greedy_exact"] != bool(match and match.group(1) == query["answer"]) or
        not math.isfinite(actual["answer_nll"]) or actual["answer_nll"] < 0):
        raise ValueError("Own-world unrestricted grading, token boundary or finite gold NLL changed")


def audit_full_rows(rows, physical, model, policy):
    if [r["id"] for r in rows] != [q["id"] for q in physical]:
        raise ValueError("Fresh full-history reference coverage/order changed")
    for row, q in zip(rows, physical):
        audit_readout(readout(row), q)
        if (row["model"] != model or row["regime"] != "full" or row["token_policy"] != policy or
            row["prefill"] != "full" or row["registration_sha256"] != data_fingerprint(q) or
            row["mask_registration_sha256"] is not None or row["keep_bits_sha256"] is not None or row["hook_calls"] is not None):
            raise ValueError("Fresh full-history reference unexpectedly masked or mislabeled")


def audit_proofs(proofs, sample_keys, full_reference, indexed_rows, masked_regimes):
    if [p['id'] for p in proofs] != sample_keys:
        raise ValueError('Complete prescribed all-role exact-control samples required')
    for p in proofs:
        if (not p['unmasked_reference_exact'] or p['unmasked_readout'] != readout(full_reference[p['id']]) or
            not p['all_history_cache_logits_readout_exact'] or p['all_history_hook_calls']['prefill'] != 2 or
            p['all_history_hook_calls']['cached'] < 1 or not p['hooks_restored'] or
            list(p['regimes']) != list(masked_regimes) or list(p['unmasked_weights']) != list(masked_regimes)):
            raise ValueError('Actual exact unmasked/all-history controls or mask coverage changed')
        hashes = set()
        for regime in masked_regimes:
            check_weight_stats(p['unmasked_weights'][regime], False)
            state = p['regimes'][regime]; row = indexed_rows[regime][p['id']]
            if (not state['complete_prefix_memory_exact'] or not state['prefix_logits_exact'] or
                state['historical_kv_fields_exact'] != 12 or state['prefix_sha256'] != state['masked_prefix_sha256'] or
                not re.fullmatch(r'[0-9a-f]{64}', state['prefix_sha256']) or
                state['proof_hook_calls'] != {'prefill': 2, 'cached': 0} or state['readout'] != readout(row) or
                state['measure_hook_calls'] != row['hook_calls']):
                raise ValueError('Fresh complete memory/conv/historicalKV or observed mask proof changed')
            check_weight_stats(state['attention_stats'], True); hashes.add(state['prefix_sha256'])
        if masked_regimes and len(hashes) != 1:
            raise ValueError('Different masks used different original prefix states')


def audit_run(path, allow_smoke=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed" or (manifest["smoke"] and not allow_smoke):
        raise ValueError("Completed canonical full evaluation required; smoke is operational only")
    plan = read_plan()
    dataset = Path(manifest["dataset"])
    registered, source_plan, cases, registry, masks = registered_inputs(dataset)
    profile = json.loads(Path(plan["precision_addendum"]).read_text())
    novelty = verify_prior_index(dataset, registered)
    physical, bindings = selected_inputs(registry, manifest["smoke"])
    samples, prefix_samples = sample_ids(cases, registry), safety_samples(registry)
    queries = {q["id"]: q for q in physical}
    order = model_order(manifest["reverse_model_order"])
    weights = manifest["weights_after"]
    if (manifest["training"] or manifest["new_adapters"] or
        (manifest["reverse_model_order"] and not manifest["smoke"]) or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or
        manifest["precision_addendum_sha256"] != ADDENDUM_SHA256 or manifest["arithmetic_profile"] != profile["profile"] or
        manifest["independent_novelty_recheck"] != novelty or manifest["prior_inputs_sha256"] != registered["prior_inputs_sha256"] or
        manifest["primary_reference"] != "fresh_full_same_canonical_arithmetic_profile" or manifest["cross_profile_rows_in_primary"] != 0 or
        manifest["dataset_manifest_sha256"] != streaming_hash(dataset / "manifest.json") or
        manifest["mask_registry_sha256"] != registered["mask_registry_sha256"] or manifest["references"] != registered["references"] or
        json.loads((path / "plan.json").read_text()) != plan or json.loads((path / "precision_addendum.json").read_text()) != profile or
        manifest["model_path"] != plan["model_path"] or manifest["packages"] != plan["packages"] or
        manifest["selected_physical_ids"] != list(queries) or manifest["bindings_per_model_regime"] != len(bindings) or
        manifest["sample_query_ids"] != samples or manifest["native_prefix_sample_ids"] != prefix_samples or
        manifest["generation"] != plan["generation"] or manifest["model_order"] != order or
        manifest["completed_models"] != order or list(manifest["models"]) != order or
        set(manifest["source_sha256"]) != sources() or manifest["weights_before"] != weights or
        weights["sha256"] != plan["native_loaded_weights_sha256"] or
        not weights["all_parameters_frozen"] or not weights["no_parameter_gradients"] or not manifest["hooks_restored"] or
        not (datetime.fromisoformat(plan["registered_utc"]) < datetime.fromisoformat(registered["registered_utc"]) <
             datetime.fromisoformat(manifest["started_utc"]) < datetime.fromisoformat(manifest["completed_utc"])) or
        not datetime.fromisoformat(profile["registered_utc"]) < datetime.fromisoformat(manifest["started_utc"])):
        raise ValueError("Registered canonical scope, runtime, source coverage, chronology or frozen native tensors changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived canonical runtime source changed")
    if manifest["generation_config_sha256"] != {p.name: streaming_hash(p) for p in Path(plan["model_path"]).glob("generation_config.json")}:
        raise ValueError("Actual native generation config changed")
    if manifest["smoke"]:
        if manifest["smoke_equivalence_gate"] is not None:
            raise ValueError("Operational smoke has an unexpected primary gate")
    else:
        gate = manifest["smoke_equivalence_gate"]
        if checked_smoke_equivalence(Path(gate["path"]), manifest["dataset_manifest_sha256"]) != gate:
            raise ValueError("Pre-primary independent smoke equivalence gate changed")
        eq = json.loads(Path(gate["path"]).read_text())
        if (eq["runtime_source_sha256"] != manifest["source_sha256"] or
            any(datetime.fromisoformat(t) >= datetime.fromisoformat(manifest["started_utc"]) for t in eq["smoke_completed_utc"])):
            raise ValueError("Primary run preceded operational smoke or changed its execution source")
    if [r["id"] for r in manifest["native_anchors"]] != samples:
        raise ValueError("Complete independent pre-controller native anchors required")
    anchors = {r["id"]: r["readout"] for r in manifest["native_anchors"]}
    for key, actual in anchors.items():
        audit_readout(actual, queries[key])
    all_rows = {}
    proof_count = prefix_count = detach_count = mask_count = 0
    for name in order:
        state, spec = manifest["models"][name], source_plan["models"][name]
        regimes = plan["regimes_by_model"][name]
        masked_regimes = regimes[1:]
        policy = spec["token_policy"] if spec else "none"
        if (state["completed_queries_per_regime"] != len(physical) or not state["hooks_restored"] or
            state["token_policy"] != policy or list(state["results_sha256"]) != regimes):
            raise ValueError("Actual fixed model or three-regime coverage changed")
        if spec:
            training = json.loads((Path(spec["training_run"]) / "manifest.json").read_text())
            if (state["description"] != training["arms"][spec["training_arm"]]["description"] or
                state["adapter_parameters"] != spec["parameters"] or state["checkpoint_sha256"] != spec["checkpoint_sha256"] or
                state["adapter_state_sha256_before"] != state["adapter_state_sha256_after"] or
                state["adapter_state_sha256_after"] != registered["references"]["training"][name]["checkpoint_state_sha256"] or
                not state["all_adapter_parameters_frozen"] or not state["arithmetic_hooks_removed"]):
                raise ValueError("Actual frozen adapter tensors, gradients, allocation or cleanup changed")
            audit_affines(state, profile["profile"])
        elif any(k.startswith(("adapter_", "checkpoint_", "arithmetic", "affine_", "actual_affine")) for k in state):
            raise ValueError("Native arm unexpectedly has adapter tensors or canonical hooks")
        all_rows[name] = {}
        for regime in regimes:
            result_path = path / f"{name}.{regime}.jsonl"
            if streaming_hash(result_path) != state["results_sha256"][regime]:
                raise ValueError("Actual readout file changed")
            rows = [json.loads(line) for line in result_path.read_text().splitlines()]
            if any(r["arithmetic_profile"] != profile["profile"] or
                   r["reference_policy"] != "fresh_full_same_canonical_arithmetic_profile" for r in rows):
                raise ValueError("Cross-profile rows entered primary references or contrasts")
            if regime == "full":
                audit_full_rows(rows, physical, name, policy)
            else:
                audit_rows(rows, physical, masks, name, regime, policy)
            all_rows[name][regime] = rows
        indexed = {r: {row["id"]: row for row in rows} for r, rows in all_rows[name].items()}
        audit_proofs(state["proofs"], samples, indexed["full"], indexed, masked_regimes)
        proof_count += len(state["proofs"])
        mask_count += len(state["proofs"]) * len(masked_regimes)
        if name == "native" and any(readout(indexed["full"][key]) != value for key, value in anchors.items()):
            raise ValueError("Native arm failed pre-controller anchor replay")
        expected_prefixes = [queries[i]["source_key"] for i in prefix_samples] if policy == "question" else []
        if [p["source_key"] for p in state["prefix_checks"]] != expected_prefixes:
            raise ValueError("Question-only native-prefix coverage incomplete")
        prefix_hashes = {}
        for p in state["prefix_checks"]:
            q = next(q for q in physical if q["source_key"] == p["source_key"])
            if (p["ledger_tokens"] != q["ledger_tokens"] or not p["complete_KV_recurrent_conv_bookkeeping_exact"] or
                not p["last_prefix_logits_exact"] or not re.fullmatch(r"[0-9a-f]{64}", p["prefix_sha256"])):
                raise ValueError("Actual question-only native prefix/logits changed")
            prefix_hashes[p["source_key"]] = p["prefix_sha256"]
        for p in state["proofs"]:
            source = queries[p["id"]]["source_key"]
            if source in prefix_hashes and any(p["regimes"][r]["prefix_sha256"] != prefix_hashes[source] for r in masked_regimes):
                raise ValueError("Question-only masked prefix differs from actual native prefix")
        prefix_count += len(state["prefix_checks"])
        if ([p["id"] for p in state["native_restore_checks"]] != samples or
            any(not p["exact"] or p["readout"] != anchors[p["id"]] for p in state["native_restore_checks"])):
            raise ValueError("Detached original native full-history readouts failed exact replay")
        detach_count += len(state["native_restore_checks"])
    if (proof_count, prefix_count, detach_count, mask_count) != (80, 12, 80, 60):
        raise ValueError("Actual complete-memory/native-proof coverage does not match registration")
    metrics = ({"primary_evaluated": False} if manifest["smoke"] else
               summarize({m: all_rows[m] for m in MODELS}, cases, registry, source_plan, plan))
    return {"status": "audited_fresh_copy_smoke" if manifest["smoke"] else "completed_and_audited_fresh_copy",
        "source_run": str(path), "manifest_sha256": streaming_hash(path / "manifest.json"),
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"], "mask_registry_sha256": registered["mask_registry_sha256"],
        "precision_addendum_sha256": ADDENDUM_SHA256, "arithmetic_profile": profile["profile"],
        "weights_sha256": weights["sha256"], "references": registered["references"], "model_order": order,
        "new_physical_readouts_audited": len(physical) * 14, "fresh_full_reference_readouts_audited": len(physical) * 8,
        "new_masked_readouts_audited": len(physical) * 6, "new_related_bindings_audited": len(bindings) * 14,
        "cross_profile_rows_in_primary": 0, "native_anchors": len(anchors),
        "all_history_exact_controls": proof_count, "actual_masked_attention_checks": mask_count,
        "complete_prefix_and_historical_KV_proofs": mask_count, "detached_native_exact_replays": detach_count,
        "question_only_native_prefix_checks": prefix_count, "metrics": metrics, "interpretation": plan["preserve"],
        "analysis_source_sha256": streaming_hash(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior independent audit files")
    report = audit_run(args.run, args.allow_smoke)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: report[k] for k in ("status", "new_physical_readouts_audited", "fresh_full_reference_readouts_audited",
        "actual_masked_attention_checks", "complete_prefix_and_historical_KV_proofs")}), flush=True)


if __name__ == "__main__":
    main()
