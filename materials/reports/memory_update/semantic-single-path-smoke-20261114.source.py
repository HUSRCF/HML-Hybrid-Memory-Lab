"""Audit component-specific semantic controls against the fixed augmentation trial."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.training_data import data_fingerprint
from scripts.analyze_order_augmented_semantics import audit_run as audit_augmentation, audit_order_evaluation
from scripts.analyze_paired_semantic_controllers import audit_checkpoint, audit_evaluation, audit_safety
from scripts.prepare_order_augmented_semantics import registered_augmented_dataset
from scripts.run_controller_trial import state_fingerprint, write_json


ARMS = ["gdn_only", "attention_only"]
SPECS = {"gdn_only": {"mode": "gdn_only", "rank": 84, "token_policy": "all", "parameters": 696192},
         "attention_only": {"mode": "attention_only", "rank": 150, "token_policy": "all", "parameters": 691200}}


def expected_shapes(config, spec):
    rank = spec["rank"]
    branch = "gdn" if spec["mode"] == "gdn_only" else "attention"
    if spec["mode"] not in ARMS:
        raise ValueError("Not a registered single-component controller")
    encoders = {branch: [rank, config["hidden_size"]]}
    outputs = {}
    if branch == "gdn":
        for offset in range(3):
            for field in ("a", "b"):
                outputs[f"gdn_{offset}_{field}"] = [config["linear_num_value_heads"], rank]
            outputs[f"gdn_{offset}_key"] = [config["linear_num_key_heads"] * config["linear_key_head_dim"], rank]
    else:
        outputs["attention_q_proj"] = [config["num_attention_heads"] * config["head_dim"], rank]
        outputs["attention_k_proj"] = [config["num_key_value_heads"] * config["head_dim"], rank]
    return encoders, outputs


def check_scope(description, spec, config, group):
    encoders, outputs = expected_shapes(config, spec)
    count = sum(rows * columns for rows, columns in list(encoders.values()) + list(outputs.values()))
    if (description["mode"] != spec["mode"] or description["rank"] != spec["rank"] or description["group_index"] != group or
        description["layers"] != list(range(4*group, 4*group+4)) or description["adapt_gdn_keys"] is not True or
        description["cache_expansion"] is not False or description["encoder_shapes"] != encoders or
        description["output_shapes"] != outputs or description["trainable_parameters"] != count or count != spec["parameters"]):
        raise ValueError("Actual controller contains a wrong projection scope, shape or parameter count")
    return count


def fresh_initial_fingerprint(config, spec, seed):
    """Reconstruct the controller's CPU initialization without loading a backbone."""
    import torch
    from torch import nn
    encoders, outputs = expected_shapes(config, spec)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        module = nn.Module()
        module.encoders = nn.ModuleDict({name: nn.Linear(shape[1], shape[0], bias=False, device="cpu", dtype=torch.float32)
                                         for name, shape in encoders.items()})
        module.outputs = nn.ModuleDict({name: nn.Linear(shape[1], shape[0], bias=False, device="cpu", dtype=torch.float32)
                                        for name, shape in outputs.items()})
        for output in module.outputs.values():
            nn.init.zeros_(output.weight)
        return state_fingerprint(module)


def model_config(model_path):
    config = json.loads((Path(model_path) / "config.json").read_text())
    return config.get("text_config", config)


def gated_reference(plan):
    reference = Path(plan["augmentation_reference_run"])
    if (plan["profile"] != "single_path_semantic_controls" or list(plan["arms"]) != ARMS or plan["arms"] != SPECS or
        file_hash(reference / "manifest.json") != plan["augmentation_reference_manifest_sha256"] or
        file_hash(reference / "plan.json") != plan["augmentation_plan_sha256"] or
        file_hash(Path(plan["data_source"]) / "manifest.json") != plan["data_manifest_sha256"]):
        raise ValueError("Registered component scopes or augmentation reference changed")
    original = json.loads((reference / "plan.json").read_text())
    allowed_changes = {"status", "registered_utc", "purpose", "arms", "gate", "requirements", "interpretation", "profile", "initial_state_sha256"}
    if any(plan.get(key) != value for key, value in original.items() if key not in allowed_changes):
        raise ValueError("Component controls changed a matched data/model/training/evaluation field")
    audit = audit_augmentation(reference)
    if audit["status"] != "completed_and_audited" or audit["validation_records_audited"] != 3456:
        raise ValueError("All four augmented controls must finish independent audit before new inference")
    manifest = json.loads((reference / "manifest.json").read_text())
    return {"augmentation": audit, "native": manifest["native_reference"]}


def audit_run(path, allow_smoke=False):
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    registered, data, registry = registered_augmented_dataset(Path(plan["data_source"]))
    references = gated_reference(plan)
    if (manifest["status"] != "completed" or manifest["completed_arms"] != ARMS or manifest["reader_policy"] != "full" or
        manifest["plan_sha256"] != data_fingerprint(plan) or manifest["native_reference"] != references["native"] or
        json.loads((path / "reference_audits.json").read_text()) != references or manifest["model_path"] != registered["model_path"] or
        manifest["data_manifest_sha256"] != plan["data_manifest_sha256"]):
        raise ValueError("Incomplete or changed component-control trial")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Safety smoke is not completed scheduled training")
    for stamp in (registered["registered_utc"], plan["registered_utc"]):
        if datetime.fromisoformat(stamp) >= datetime.fromisoformat(manifest["started_utc"]):
            raise ValueError("Component data/protocol was not fixed before inference")
    reference_manifest = json.loads((Path(plan["augmentation_reference_run"]) / "manifest.json").read_text())
    if manifest["packages"] != reference_manifest["packages"]:
        raise ValueError("Component-control runtime packages differ from matched reference")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest or file_hash(ROOT / relative) != digest:
            raise ValueError("Actual/archived component-control source changed")
    for key in ("data_sha256", "registry_sha256", "tokenized_input_answer_sha256"):
        if plan[key] != registered[key]:
            raise ValueError("Registered component-control data changed")
    encoded = json.loads((Path(plan["data_source"]) / "tokenized_train.json").read_text())
    if encoded != json.loads((path / "tokenized_train.json").read_text()):
        raise ValueError("Actual component-control optimized tokens changed")
    config = model_config(registered["model_path"])
    models = {}
    for arm, spec in plan["arms"].items():
        safety = json.loads((path / arm / "safety_check.json").read_text())
        audit_safety(safety, spec)
        check_scope(safety, spec, config, plan["group_index"])
        if manifest["smoke"]:
            models[arm] = {"parameters": spec["parameters"], "status": "scope_and_safety_passed"}
            continue
        state = manifest["arms"][arm]
        check_scope(state["description"], spec, config, plan["group_index"])
        initial = fresh_initial_fingerprint(config, spec, plan["initialization_seed"])
        if (state["initial_state_sha256"] != initial or state["completed_checkpoints"] != plan["checkpoint_steps"] or
            not state["native_restore_exact"] or len((path / arm / "training.jsonl").read_text().splitlines()) != plan["maximum_optimizer_steps"]):
            raise ValueError("Component initialization/schedule/native restoration failed")
        for step in plan["checkpoint_steps"]:
            audit_checkpoint(path, arm, step, plan, encoded)
        canonical = audit_evaluation(path, arm, plan, data, encoded, registry)
        permuted = audit_order_evaluation(path, arm, plan)
        if state["extra_validation"] != permuted:
            raise ValueError("Component permuted-evaluation descriptor changed")
        models[arm] = {"parameters": spec["parameters"], "initial_state_sha256": initial, "initialization_reconstructed": True,
                       "checkpoints_audited": len(plan["checkpoint_steps"]), "teacher_train_fit": canonical["teacher_train_fit"],
                       "canonical": canonical, "permuted": permuted}
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
            "manifest_sha256": file_hash(path / "manifest.json"), "arms": models,
            "validation_records_audited": 0 if manifest["smoke"] else len(ARMS)*2*len(data["validation"]),
            "teacher_train_records_audited": 0 if manifest["smoke"] else len(ARMS)*len(data["train"]),
            "augmentation_reference_audit": references["augmentation"], "interpretation": plan["interpretation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report = audit_run(args.run, args.allow_smoke)
    report["analysis_source_sha256"] = file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": report["status"], "arms": {arm:{condition:value[condition]["metrics"]["correct"] for condition in ("canonical", "permuted")} if "canonical" in value else value for arm,value in report["arms"].items()}}))


if __name__ == "__main__":
    main()
