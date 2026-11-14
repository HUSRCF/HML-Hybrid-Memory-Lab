"""Independently audit every native cache-origin readout and replay control."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_paired_semantic_probe import identity
from scripts.compare_paired_semantic_controllers import functional_units, paired_counts
from scripts.run_controller_trial import write_json
from scripts.semantic_order_path_common import (CONDITIONS, functional_report, nll_contrasts,
    reference_gate, registered_probe_inputs, selected_examples)


def full_prefill_calibration(plan, expected, evaluated):
    native_run = Path(plan["full_reference_native_run"])
    permuted_run = Path(plan["full_reference_order_run"])
    canonical_manifest = json.loads((native_run / "manifest.json").read_text())
    permuted_manifest = json.loads((permuted_run / "manifest.json").read_text())
    sources = {"canonical": (native_run / "validation.native.jsonl", canonical_manifest["result_sha256"]),
               "permuted": (permuted_run / "native.jsonl", permuted_manifest["models"]["native"]["results_sha256"])}
    report = {}
    for origin, (path, digest) in sources.items():
        if file_hash(path) != digest:
            raise ValueError("Audited full-prefill natural generations changed")
        all_rows = [json.loads(line) for line in path.read_text().splitlines()]
        all_expected = {(row["case_id"],row["variant"],row["role"]): row["answer"] for row in all_rows}
        rows = [row for row in load_rows(path, all_expected) if (row["case_id"],row["variant"],row["role"]) in expected]
        cached = evaluated[f"{origin}_natural"]
        original_units, cached_units = functional_units(rows), functional_units(cached)
        indexed = {(row["case_id"],row["variant"],row["role"]):row for row in rows}
        errors = [abs(row["answer_nll"] - indexed[row["case_id"],row["variant"],row["role"]]["answer_nll"]) for row in cached]
        report[origin] = {"full_prefill_source": str(path), "full_prefill_sha256": digest,
            "full_to_cached": {metric: paired_counts(original_units[metric], cached_units[metric]) for metric in original_units},
            "nll_mean_absolute_difference": sum(errors)/len(errors), "nll_max_absolute_difference": max(errors)}
    return report


def audit_run(path, allow_smoke=False):
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    base, data, cases, registry = registered_probe_inputs(plan)
    reference = reference_gate(plan)
    if (manifest["status"] != "completed" or manifest["training"] or manifest["reader_policy"] != "full" or
        manifest["completed_conditions"] != list(CONDITIONS) or manifest["plan_sha256"] != data_fingerprint(plan) or
        manifest["reference_audit"] != reference or manifest["model_path"] != base["model_path"] or
        not manifest["hooks_restored"] or not manifest["frozen_parameters_and_no_gradients"]):
        raise ValueError("Incomplete or changed native cache-origin diagnostic")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Operational path smoke is not the registered full cohort")
    if datetime.fromisoformat(plan["registered_utc"]) >= datetime.fromisoformat(manifest["started_utc"]):
        raise ValueError("Native path plan was not fixed before inference")
    native_manifest = json.loads((Path(plan["full_reference_native_run"]) / "manifest.json").read_text())
    if manifest["packages"] != native_manifest["packages"]:
        raise ValueError("Native path runtime packages changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest or file_hash(ROOT / relative) != digest:
            raise ValueError("Actual or archived native path source changed")
    selected = selected_examples(data, manifest["smoke"])
    keys = [tuple(identity(example)) for example in selected]
    expected = {key: example["query"]["answer"] for key,example in zip(keys,selected)}
    if (manifest["evaluated_keys"] != [list(key) for key in keys] or manifest["natural_prefix_prefills"] != 2*len(keys) or
        manifest["unique_prefixes_per_order"] != len({(key[0],key[1]) for key in keys})):
        raise ValueError("Native path cohort, query order or prefix accounting changed")
    evaluated = {}
    prefix_checks = manifest["prefix_checks"]
    if [check["key"] for check in prefix_checks] != manifest["evaluated_keys"]:
        raise ValueError("Missing cache-origin/immutability controls")
    sample_keys = {min(keys),max(keys)}
    seen_hashes = {}
    for key, check in zip(keys, prefix_checks):
        if (not check["all_condition_origins_verified"] or not check["sources_unchanged_after_each_query"] or
            check["ledger_tokens"] != registry["canonical"][key]["ledger_tokens"] or
            set(check["prefix_sha256"]) != {"canonical","permuted"} or
            any(len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in check["prefix_sha256"].values())):
            raise ValueError("Native cache-origin/source controls failed")
        prefix_key = key[:2]
        if prefix_key in seen_hashes and check["prefix_sha256"] != seen_hashes[prefix_key]:
            raise ValueError("Repeated natural prefix recomputation differs across queries")
        seen_hashes[prefix_key] = check["prefix_sha256"]
        replay = check["replay_checks"]
        if key in sample_keys:
            if ({(row["origin"],row["kind"]) for row in replay} != {(origin,kind) for origin in ("canonical","permuted") for kind in ("self","all_path_donor")} or
                len(replay) != 4 or any(not row["readout_matches"] or not 0 <= row["nll_absolute_error"] <= 1e-5 for row in replay)):
                raise ValueError("Prescribed self/all-path donor readout replay failed")
        elif replay:
            raise ValueError("Unexpected outcome-dependent replay selection")
    prescribed = {(key,name) for key in sample_keys for name in CONDITIONS}
    checks = manifest["safety_checks"]
    if len(checks) != len(prescribed) or {(tuple(check["key"]),check["condition"]) for check in checks} != prescribed:
        raise ValueError("Missing prescribed cached attention observations")
    for check in checks:
        if set(check["full_attention_weights"]) != {str(i) for i in (3,7,11,15,19,23)}:
            raise ValueError("Missing observed historical attention layer")
        if any(s["prefill_calls"] != 0 or s["cached_calls"] < 1 or s["max_record_mass"] <= 0 for s in check["full_attention_weights"].values()):
            raise ValueError("Prescribed query used a wrong prefill path or no observed historical attention")
    for name, origins in CONDITIONS.items():
        source = path / f"{name}.jsonl"
        if file_hash(source) != manifest["results_sha256"][name]:
            raise ValueError("Raw native path generation changed")
        rows = load_rows(source, expected)
        if [(r["case_id"],r["variant"],r["role"]) for r in rows] != keys:
            raise ValueError("Native path free generations changed query order")
        for row, example, check in zip(rows,selected,prefix_checks):
            key = tuple(identity(example))
            if (row["condition"] != name or row["cache_origins"] != origins or row["reader_policy"] != "full" or row["prefill"] != "cached" or
                not row["source_cache_unchanged"] or row["prefix_sha256"] != check["prefix_sha256"] or
                row["registrations"] != {origin:registry[origin][key] for origin in registry} or
                row["query_tokens"] != registry["canonical"][key]["prompt_tokens"]-registry["canonical"][key]["ledger_tokens"] or
                any(row[field] != example["case"][field] for field in ("family","seed","base_case_id","value_version","distractor_updates"))):
                raise ValueError("Native path input/metadata/origin changed")
        evaluated[name] = rows
    functional = functional_report(evaluated)
    if functional != manifest["functional_summary"]:
        raise ValueError("Native path summary differs from raw free generation")
    return {"status": "audited_operational_smoke" if manifest["smoke"] else "completed_and_audited",
        "source_run": str(path), "manifest_sha256": file_hash(path / "manifest.json"),
        "records_audited": len(keys)*len(CONDITIONS), "functional": functional,
        "full_prefill_calibration": full_prefill_calibration(plan,expected,evaluated),
        "nll_source_contrasts": nll_contrasts(evaluated), "interpretation": plan["interpretation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-smoke",action="store_true")
    args = parser.parse_args()
    report = audit_run(args.run,args.allow_smoke)
    report["analysis_source_sha256"] = file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status":report["status"],"records":report["records_audited"],
                      "correct":{name:metrics["individual"]["correct"] for name,metrics in report["functional"]["counts"].items()}}))


if __name__ == '__main__':
    main()
