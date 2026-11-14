"""Audit the matched GDN question/answer timing control and native prefix identity."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.training_data import data_fingerprint
from scripts.analyze_order_augmented_semantics import audit_order_evaluation, order_inputs
from scripts.analyze_paired_semantic_controllers import audit_checkpoint, audit_evaluation, audit_safety
from scripts.analyze_semantic_single_path_controls import (audit_run as audit_components, check_scope,
    fresh_initial_fingerprint, model_config)
from scripts.analyze_semantic_order_path_probe import audit_run as audit_native_paths
from scripts.prepare_order_augmented_semantics import registered_augmented_dataset
from scripts.run_controller_trial import write_json

ARM = "gdn_question"
SPEC = {"mode":"gdn_only","rank":84,"token_policy":"question","parameters":696192}


def check_plan_pair(plan,original,component_manifest):
    changed = {"status","registered_utc","purpose","profile","arms","initial_state_sha256","gate","requirements","interpretation"}
    if (plan["profile"] != "gdn_question_timing_control" or list(plan["arms"]) != [ARM] or plan["arms"][ARM] != SPEC or
        original["arms"]["gdn_only"] != {**SPEC,"token_policy":"all"} or
        any(plan.get(key) != value for key,value in original.items() if key not in changed)):
        raise ValueError("Question timing changed scope or a matched model/data/training field")
    initial = component_manifest["arms"]["gdn_only"]["initial_state_sha256"]
    if plan["initial_state_sha256"] != {ARM:initial}:
        raise ValueError("Question timing initialization is not paired with the all-token GDN reference")


def gated_references(plan):
    # The training engine also uses four CPU threads. Cap reference tensor
    # audits here, before loading a model, to avoid excessive OpenMP overhead.
    import torch
    torch.set_num_threads(4)
    component = Path(plan["component_reference_run"])
    native_path = Path(plan["native_path_reference_run"])
    for path,manifest_key,plan_key in ((component,"component_reference_manifest_sha256","component_reference_plan_sha256"),
                                       (native_path,"native_path_reference_manifest_sha256","native_path_plan_sha256")):
        if (file_hash(path/"manifest.json") != plan[manifest_key] or file_hash(path/"plan.json") != plan[plan_key]):
            raise ValueError("Pinned timing-control reference changed")
    original = json.loads((component/"plan.json").read_text())
    component_manifest = json.loads((component/"manifest.json").read_text())
    check_plan_pair(plan,original,component_manifest)
    components = audit_components(component)
    native = audit_native_paths(native_path)
    native_plan = json.loads((native_path/"plan.json").read_text())
    native_manifest = json.loads((native_path/"manifest.json").read_text())
    if (components["status"] != "completed_and_audited" or components["validation_records_audited"] != 1728 or
        components["teacher_train_records_audited"] != 1728 or native["status"] != "completed_and_audited" or
        native["records_audited"] != 1728 or native_plan["base_data_sha256"] != plan["base_data_sha256"] or
        native_manifest["model_path"] != component_manifest["model_path"] or native_manifest["packages"] != component_manifest["packages"]):
        raise ValueError("Full native/component timing-control gates failed")
    return {"component":components,"native_paths":native,"native":component_manifest["native_reference"]}


def check_prefix_samples(checks,registry):
    prescribed = [(row["ledger_tokens"],row["case_id"],row["variant"]) for row in registry]
    actual = [(row["ledger_tokens"],row["case_id"],row["variant"]) for row in checks]
    if actual != [min(prescribed),max(prescribed)] or any(row["kv_recurrent_conv_and_logits_exact"] is not True for row in checks):
        raise ValueError("Native prefix identity was not checked on the prescribed shortest/longest ledgers")
    return checks


def audit_run(path,allow_smoke=False):
    manifest = json.loads((path/"manifest.json").read_text())
    plan = json.loads((path/"plan.json").read_text())
    references = gated_references(plan)
    registered,data,registry = registered_augmented_dataset(Path(plan["data_source"]))
    if (manifest["status"] != "completed" or manifest["completed_arms"] != [ARM] or manifest["reader_policy"] != "full" or
        manifest["plan_sha256"] != data_fingerprint(plan) or manifest["native_reference"] != references["native"] or
        json.loads((path/"reference_audits.json").read_text()) != references or manifest["model_path"] != registered["model_path"] or
        manifest["data_manifest_sha256"] != plan["data_manifest_sha256"]):
        raise ValueError("Incomplete or changed GDN timing trial")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("One-update safety is not completed timing-control training")
    for stamp in (registered["registered_utc"],plan["registered_utc"]):
        if datetime.fromisoformat(stamp) >= datetime.fromisoformat(manifest["started_utc"]):
            raise ValueError("Timing protocol/data were not registered before inference")
    original_manifest = json.loads((Path(plan["component_reference_run"])/"manifest.json").read_text())
    if manifest["packages"] != original_manifest["packages"]:
        raise ValueError("Timing trial runtime packages changed")
    for relative,digest in manifest["source_sha256"].items():
        if file_hash(path/"sources"/relative) != digest or file_hash(ROOT/relative) != digest:
            raise ValueError("Actual/archived timing trial source changed")
    for key in ("data_sha256","registry_sha256","tokenized_input_answer_sha256"):
        if plan[key] != registered[key]:
            raise ValueError("Timing trial data changed")
    encoded = json.loads((Path(plan["data_source"])/"tokenized_train.json").read_text())
    if encoded != json.loads((path/"tokenized_train.json").read_text()):
        raise ValueError("Actual timing trial optimized tokens changed")
    config = model_config(registered["model_path"])
    safety = json.loads((path/ARM/"safety_check.json").read_text())
    audit_safety(safety,SPEC)
    check_scope(safety,SPEC,config,plan["group_index"])
    initial = fresh_initial_fingerprint(config,SPEC,plan["initialization_seed"])
    if initial != plan["initial_state_sha256"][ARM]:
        raise ValueError("Reconstructed timing initialization differs from paired reference")
    result = {"status":"audited_smoke" if manifest["smoke"] else "completed_and_audited","source_run":str(path),
        "manifest_sha256":file_hash(path/"manifest.json"),"parameters":SPEC["parameters"],"initial_state_sha256":initial,
        "initialization_reconstructed":True,"paired_initialization_verified":True,
        "component_reference_audit":references["component"],"native_path_reference_audit":references["native_paths"],
        "validation_records_audited":0,"teacher_train_records_audited":0,"checkpoints_audited":0,
        "interpretation":plan["interpretation"]}
    if manifest["smoke"]:
        result["safety_status"] = "scope_gradients_zero_identity_and_native_prefix_passed"
        return result
    state = manifest["arms"][ARM]
    check_scope(state["description"],SPEC,config,plan["group_index"])
    if (state["initial_state_sha256"] != initial or state["completed_checkpoints"] != plan["checkpoint_steps"] or
        not state["native_restore_exact"] or len((path/ARM/"training.jsonl").read_text().splitlines()) != plan["maximum_optimizer_steps"]):
        raise ValueError("Timing initialization/schedule/native restoration failed")
    for step in plan["checkpoint_steps"]:
        audit_checkpoint(path,ARM,step,plan,encoded)
    canonical = audit_evaluation(path,ARM,plan,data,encoded,registry)
    permuted = audit_order_evaluation(path,ARM,plan)
    if state["extra_validation"] != permuted:
        raise ValueError("Timing permuted-evaluation descriptor changed")
    checkpoint = path/ARM/f"step-{plan['maximum_optimizer_steps']:06d}"
    canonical_summary = json.loads((checkpoint/"evaluation/summary.json").read_text())
    permuted_summary = json.loads((checkpoint/"initial_order_evaluation/summary.json").read_text())
    _,permuted_registry = order_inputs(plan)
    prefix_checks = {"canonical":check_prefix_samples(canonical_summary["pre_question_prefix_checks"],registry["validation"]),
                     "permuted":check_prefix_samples(permuted_summary["pre_question_prefix_checks"],permuted_registry)}
    result.update(canonical=canonical,permuted=permuted,teacher_train_fit=canonical["teacher_train_fit"],native_prefix_checks=prefix_checks,
                  checkpoints_audited=len(plan["checkpoint_steps"]),validation_records_audited=2*len(data["validation"]),
                  teacher_train_records_audited=len(data["train"]))
    return result


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
    print(json.dumps({"status":report["status"],"validation_records":report["validation_records_audited"],
        "correct":{condition:report[condition]["metrics"]["correct"] for condition in ("canonical","permuted")} if "canonical" in report else None}))


if __name__ == '__main__':
    main()
