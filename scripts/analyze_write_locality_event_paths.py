"""Independently reconstruct the complete registered event-path observations."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.locality_event_data import registered_inputs
from scripts.locality_event_runtime import replay_pair_ids, same_readout, selected_pairs
from scripts.run_controller_trial import write_json
from scripts.run_write_locality_event_paths import RUNTIME_SOURCES


def row_key(row):
    return (row["pair_id"], row["kind"], row["condition"], row["label"], row.get("clone_kind", ""))


def expected_keys(pairs, replay_ids):
    keys = set()
    for pair in pairs:
        key = pair["id"]
        for label in pair["queries"]:
            keys.update((key, "cache_condition", index, label, "") for index in range(4))
        for side in ("left", "right"):
            keys.add((key, "oracle", side, "preservation", ""))
            if "current_C" in pair["queries"]:
                keys.add((key, "full_C_baseline", side, "current_C", ""))
            if key in replay_ids:
                keys.update((key, "replay", side, "preservation", kind) for kind in ("self", "all_path_donor"))
    return keys


def paired_outcomes(before, after):
    if len(before) != len(after):
        raise ValueError("Mismatched paired outcome lists")
    return {"units": len(before), "reference_correct": sum(before), "candidate_correct": sum(after),
            "rescued": sum(not a and b for a, b in zip(before, after)),
            "regressed": sum(a and not b for a, b in zip(before, after)),
            "both_correct": sum(a and b for a, b in zip(before, after)),
            "both_wrong": sum(not a and not b for a, b in zip(before, after))}


def summarize(pairs, indexed, calibrations):
    groups = {}
    for pair in pairs:
        groups.setdefault((pair["cohort"], pair["stratum"]), []).append(pair)
    results = []
    for (cohort, stratum), members in groups.items():
        def row(pair, index, label="preservation"):
            return indexed[(pair["id"], "cache_condition", index, label, "")]

        def correct(pair, index, label="preservation"):
            return row(pair, index, label)["world_scores"]["right"]["greedy_exact"]

        comparisons, nll_rows = [], []
        for index in range(4):
            labels = list(members[0]["queries"])
            item = {"condition": index, "versus_natural_right": {}}
            for label in labels:
                before = [correct(p, 1, label) for p in members]
                after = [correct(p, index, label) for p in members]
                item["versus_natural_right"][label] = paired_outcomes(before, after)
            reference_joint = [all(correct(p, 1, label) for label in labels) for p in members]
            candidate_joint = [all(correct(p, index, label) for label in labels) for p in members]
            item["versus_natural_right"]["all_roles_joint"] = paired_outcomes(reference_joint, candidate_joint)
            item["preservation_rescue_but_right_update_failure"] = [p["case_id"] for p in members
                if not correct(p, 1) and correct(p, index) and
                not all(correct(p, index, label) for label in labels if label != "preservation")]
            comparisons.append(item)
        for pair in members:
            ll, rr, lr, rl = [row(pair, index)["world_scores"]["right"]["answer_nll"] for index in range(4)]
            nll_rows.append({"case_id": pair["case_id"], "seed": pair["seed"], "value_version": pair["value_version"],
                "recurrent_right_minus_left_at_left_KV": rl - ll,
                "recurrent_right_minus_left_at_right_KV": rr - lr,
                "KV_right_minus_left_at_left_recurrent": lr - ll,
                "KV_right_minus_left_at_right_recurrent": rr - rl,
                "interaction_difference_in_differences": rr - lr - rl + ll})
        fields = [key for key in nll_rows[0] if key not in ("case_id", "seed", "value_version")]
        oracles = []
        for side, index in (("left", 0), ("right", 1)):
            before = [correct(p, index) for p in members]
            after = [indexed[(p["id"], "oracle", side, "preservation", "")]["world_scores"][side]["greedy_exact"] for p in members]
            oracles.append({"side": side, **paired_outcomes(before, after)})
        results.append({"cohort": cohort, "stratum": stratum, "pairs": len(members),
                        "base_streams": sorted({p["base_stream_id"] for p in members}),
                        "conditions": comparisons, "oracle_vs_cached_natural": oracles,
                        "preservation_NLL_contrasts": {"means": {key: mean(r[key] for r in nll_rows) for key in fields},
                                                       "individual_pairs": nll_rows}})
    calibration_groups = {}
    by_id = {pair["id"]: pair for pair in pairs}
    for item in calibrations:
        pair = by_id[item["pair_id"]]
        key = (pair["cohort"], pair["stratum"], item["label"], item["side"])
        calibration_groups.setdefault(key, []).append(item)
    calibration_result = []
    for key, members in calibration_groups.items():
        calibration_result.append({**dict(zip(("cohort", "stratum", "label", "side"), key)),
            **paired_outcomes([r["full_own_world_correct"] for r in members], [r["cached_own_world_correct"] for r in members]),
            "different_generation": sum(r["full_generated"] != r["cached_generated"] for r in members),
            "mean_absolute_nll_delta": mean(abs(r["cached_minus_full_nll"]) for r in members),
            "max_absolute_nll_delta": max(abs(r["cached_minus_full_nll"]) for r in members)})
    return {"by_cohort_and_stratum": results, "cached_vs_full_calibration": calibration_result,
            "world_scoring": "Every comparison above uses the fixed right-world truth. Left natural endpoints are explicit counterfactual scores for changed A/C, never a claimed own-world baseline failure. Calibration uses each endpoint's own-world truth.",
            "NLL_interpretation": "Positive differences mean greater preservation-answer loss. Mixed caches can be incompatible; mediator effects do not prove semantic destruction or shared-controller necessity.",
            "sampling": "Discovery and replica remain separate. Four value siblings share a base stream; no independent-readout confidence interval is asserted."}


def audit_run(path, allow_smoke=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed" or (manifest["smoke"] and not allow_smoke):
        raise ValueError("A completed full event-path run is required")
    registered, contexts, registry = registered_inputs(Path(manifest["dataset"]))
    pairs = selected_pairs(registry, manifest["smoke"])
    pair_index = {p["id"]: p for p in pairs}
    replay_ids = replay_pair_ids(registry, pairs, manifest["smoke"])
    required_sources = set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}
    reference = json.loads(Path(registered["references"]["replica"]["audit_path"]).read_text())
    weights = manifest["weights_after"]
    if (manifest["training"] or manifest["adapters"] or not manifest["hooks_restored"] or
        manifest["dataset_manifest_sha256"] != streaming_hash(Path(manifest["dataset"]) / "manifest.json") or
        manifest["model_path"] != registered["model_path"] or manifest["packages"] != reference["packages"] or
        manifest["selected_pair_ids"] != [p["id"] for p in pairs] or manifest["replay_pair_ids"] != sorted(replay_ids) or
        set(manifest["source_sha256"]) != required_sources or manifest["weights_before"] != weights or
        weights["sha256"] != reference["weights_sha256"] or not weights["all_parameters_frozen"] or
        not weights["no_parameter_gradients"] or manifest["decoding"] != {
            "maximum_new_tokens": 12, "candidate_restriction": False, "gold_tokens": 3,
            "one_generation_per_query_and_cache": True, "separate_world_scoring": True}):
        raise ValueError("Native scope, loaded weights, runtime or complete event cohort changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual event runtime or archived sources changed")
    for name, digest in manifest["results_sha256"].items():
        if streaming_hash(path / name) != digest:
            raise ValueError("Actual event observations changed")
    if set(manifest["results_sha256"]) != {"readouts.jsonl", "calibrations.jsonl"}:
        raise ValueError("Incomplete result file coverage")
    if manifest["smoke"]:
        if manifest["smoke_gate"] is not None:
            raise ValueError("Smoke incorrectly asserts a full-run gate")
    else:
        gate = manifest["smoke_gate"]
        gate_path = Path(gate["path"])
        report = json.loads(gate_path.read_text())
        if (streaming_hash(gate_path) != gate["sha256"] or report["status"] != "audited_smoke" or
            gate["manifest_sha256"] != report["manifest_sha256"] or
            report["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"] or
            audit_run(Path(report["source_run"]), allow_smoke=True) != report):
            raise ValueError("Complete independent native smoke gate invalid")
    rows = [json.loads(line) for line in (path / "readouts.jsonl").read_text().splitlines()]
    indexed = {row_key(row): row for row in rows}
    if len(indexed) != len(rows) or indexed.keys() != expected_keys(pairs, replay_ids):
        raise ValueError("Missing, duplicated, filtered or extra registered event readout")
    counts = dict(Counter(f"{row['kind']}/{row['label']}" for row in rows))
    if manifest["readout_counts"] != counts:
        raise ValueError("Observed readout counts disagree with completion manifest")
    for row in rows:
        pair = pair_index[row["pair_id"]]
        query = pair["queries"][row["label"]]
        if (row["query_tokens"] != len(query["endpoints"]["left"]["question_ids"]) or
            set(row["world_scores"]) != {"left", "right"}):
            raise ValueError("Question boundaries or separate-world score coverage changed")
        match = re.fullmatch(r"\s*(-?\d+)\s*", row["generated"])
        for side, score in row["world_scores"].items():
            if (score["answer"] != query["world_answers"][side] or
                score["greedy_exact"] != bool(match and match.group(1) == score["answer"]) or
                not math.isfinite(score["answer_nll"]) or score["answer_nll"] < 0):
                raise ValueError("Recorded free-generation grade or world gold score invalid")
        if row["kind"] == "cache_condition":
            spec = registry["conditions"][row["condition"]]
            if row["cache_origins"] != {"recurrent_and_convolution": spec["recurrent_conv"], "kv": spec["kv"]} or not row["source_cache_unchanged"]:
                raise ValueError("Mixed source provenance or immutable-source check missing")
        elif row["kind"] == "oracle":
            if row["mask_calls"]["prefill"] != 0 or row["mask_calls"]["cached"] < 1 or not row["source_cache_unchanged"]:
                raise ValueError("Oracle recomputed ledger or bypassed question-time mask")
        elif row["kind"] == "replay":
            side = row["condition"]
            natural = indexed[(row["pair_id"], "cache_condition", 0 if side == "left" else 1, "preservation", "")]
            if not row["exact_cache_and_metadata"] or not same_readout(natural, row):
                raise ValueError("Self/all-path clone does not replay its complete donor")
    source_checks = manifest["source_checks"]
    if [item["pair_id"] for item in source_checks] != list(pair_index):
        raise ValueError("Missing complete per-pair source checks")
    fingerprints = {}
    for item in source_checks:
        pair = pair_index[item["pair_id"]]
        if (item["source_keys"] != pair["source_keys"] or not item["all_condition_origins_verified"] or
            not item["sources_unchanged_after_each_query"] or not item["supporting_KV"]["exact"] or
            item["supporting_KV"]["support_span"] != pair["alignment"]["support_span"] or
            item["supporting_KV"]["checked_KV_tensors"] != [[i, field, 0] for i in (3, 7, 11, 15, 19, 23) for field in ("keys", "values")]):
            raise ValueError("Causal evidence identity or complete origin checks missing")
        for side, key in item["source_keys"].items():
            expected_layout = {"layers": 24, "GDN_recurrent_layers": 18, "GDN_convolution_layers": 18,
                               "KV_layers": 6, "tensor_count": 48,
                               "sequence_length": registry["prefixes"][key]["ledger_tokens"]}
            digest = item["prefix_sha256"][side]
            if item["layouts"][side] != expected_layout or fingerprints.setdefault(key, digest) != digest or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Source cache layout or repeated-prefix identity changed")
    if (manifest["physical_ledger_prefills"] != 2 * len(pairs) or manifest["unique_natural_prefixes"] != len(fingerprints) or
        manifest["full_prefill_C_baselines"] != sum("current_C" in p["queries"] for p in pairs) * 2):
        raise ValueError("Physical prefix computations or source coverage misreported")
    observed = {(r["pair_id"], r["kind"], r["condition"]): r for r in manifest["attention_checks"]}
    expected_observations = {(key, kind, condition) for key in replay_ids for kind, conditions in (("full", range(4)), ("oracle", ("left", "right"))) for condition in conditions}
    if len(observed) != len(manifest["attention_checks"]) or observed.keys() != expected_observations:
        raise ValueError("Actual all-layer full/oracle probability checks incomplete")
    for item in observed.values():
        if set(item["stats"]) != {str(i) for i in (3, 7, 11, 15, 19, 23)} or any(
            s["prefill_calls"] != 0 or s["cached_calls"] < 1 or s["query_rows"] < 1 or s["max_retained_mass"] <= 0 or
            s["max_suppressed_probability"] != 0 or s["max_suppressed_mass"] != 0 for s in item["stats"].values()):
            raise ValueError("Actual retained/suppressed historical probability check failed")
    calibrations = [json.loads(line) for line in (path / "calibrations.jsonl").read_text().splitlines()]
    ckeys = {(r["pair_id"], r["label"], r["side"]) for r in calibrations}
    expected_calibrations = {(p["id"], label, side) for p in pairs for label in p["queries"] for side in ("left", "right")}
    if len(ckeys) != len(calibrations) or ckeys != expected_calibrations:
        raise ValueError("Natural own-world cached/full calibration incomplete")
    for item in calibrations:
        pair = pair_index[item["pair_id"]]
        label, side = item["label"], item["side"]
        cached = indexed[(pair["id"], "cache_condition", 0 if side == "left" else 1, label, "")]
        if label == "current_C":
            baseline = indexed[(pair["id"], "full_C_baseline", side, label, "")]
            full = baseline["world_scores"][side]
            generated = baseline["generated"]
            reference_hash = data_fingerprint({key: baseline[key] for key in ("generated", "query_tokens", "world_scores")})
        else:
            descriptor = pair["queries"][label]["endpoints"][side]["full_prefill_reference"]
            full = contexts[pair["cohort"]]["raw"][tuple(descriptor["key"])]
            generated, reference_hash = full["generated"], data_fingerprint(full)
        score = cached["world_scores"][side]
        expected = {"pair_id": pair["id"], "label": label, "side": side, "full_reference_sha256": reference_hash,
                    "full_generated": generated, "full_own_world_correct": full["greedy_exact"], "full_own_world_nll": full["answer_nll"],
                    "cached_generated": cached["generated"], "cached_own_world_correct": score["greedy_exact"],
                    "cached_own_world_nll": score["answer_nll"], "cached_minus_full_nll": score["answer_nll"] - full["answer_nll"]}
        if item != expected:
            raise ValueError("Cached/full calibration used wrong world, source or reference")
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
            "manifest_sha256": streaming_hash(path / "manifest.json"), "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
            "registry_sha256": registered["registry_sha256"], "weights_sha256": weights["sha256"],
            "readouts_audited": len(rows), "calibrations_audited": len(calibrations), "pairs_audited": len(pairs),
            "readout_counts": counts, "physical_ledger_prefills": manifest["physical_ledger_prefills"],
            "unique_natural_prefixes": len(fingerprints), "source_KV_identity_pairs": len(source_checks),
            "actual_attention_probability_checks": len(observed), "metrics": summarize(pairs, indexed, calibrations),
            "analysis_source_sha256": streaming_hash(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report = audit_run(args.run, args.allow_smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: report[key] for key in ("status", "readouts_audited", "calibrations_audited", "pairs_audited")}))


if __name__ == "__main__":
    main()
