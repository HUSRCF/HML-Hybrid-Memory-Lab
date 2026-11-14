"""Five existing final checkpoints on the identical sealed natural-query registry."""

from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.locality_transfer_data import (DATA_SOURCES as TRANSFER_DATA_SOURCES,
    registered_inputs as transfer_inputs, validate_training_reference)

PROTOCOL = ROOT / "configs/write_locality_control_panel_plan.json"
PROTOCOL_SHA256 = "5e77a11a20437a4bc29802cb7923cbcf753cbb580b774c0228bc3eefc350d958"
DESTINATION = Path("/media/990Pro/Code/SAS/datasets/memory_update/write-locality-control-panel-v1")
MODELS = ("shared", "independent", "local", "local_question", "attention_only")
DATA_SOURCES = tuple(sorted(set(TRANSFER_DATA_SOURCES) | {
    "configs/write_locality_control_panel_plan.json", "scripts/locality_panel_data.py",
    "scripts/prepare_write_locality_control_panel.py", "scripts/run_write_locality_adapter_transfer.py",
    "scripts/analyze_write_locality_adapter_transfer.py", "scripts/locality_transfer_metrics.py",
}))


def read_plan():
    plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or list(plan["models"]) != list(MODELS) or
        plan["profile"] != "fixed_checkpoint_hybrid_control_panel_on_identical_natural_endpoints" or
        plan["training"] or plan["new_adapters"] or not plan["local_files_only"] or
        plan["model_key"] != "it" or plan["reader_policy"] != "full" or plan["counts"] != {
            "models": 5, "reported_pair_endpoint_readouts_per_model": 448,
            "unique_physical_queries_per_model": 384, "total_reported_bindings": 2240,
            "total_physical_queries": 1920, "pairs_per_cohort_stratum": 16,
            "independent_base_streams_per_cohort": 4}):
        raise ValueError("Registered five-checkpoint panel scope changed")
    scopes = (("shared", 64, "all", 694272), ("independent", 54, "all", 696384),
              ("local", 22, "all", 689216), ("local", 22, "question", 689216),
              ("attention_only", 150, "all", 691200))
    for name, (mode, rank, policy, parameters) in zip(MODELS, scopes):
        spec = plan["models"][name]
        expected = {"mode": mode, "rank": rank, "token_policy": policy, "parameters": parameters,
                    "final_step": 432, "group_index": 3, "adapt_gdn_keys": True, "training_arm": name}
        if any(spec[key] != value for key, value in expected.items()):
            raise ValueError("Registered final checkpoint/projection allocation changed")
    if plan["models"]["local"]["initial_state_sha256"] != plan["models"]["local_question"]["initial_state_sha256"]:
        raise ValueError("Local timing controls lost their paired initialization")
    return plan


def checked_references(plan, reexecute_transfer=False):
    source = Path(plan["source_dataset"])
    if streaming_hash(source / "manifest.json") != plan["source_dataset_manifest_sha256"]:
        raise ValueError("Original natural-query registration changed")
    original, contexts, registry = transfer_inputs(source)
    audit_path = Path(plan["prior_timing_audit"])
    audit = json.loads(audit_path.read_text())
    if (streaming_hash(audit_path) != plan["prior_timing_audit_sha256"] or
        streaming_hash(audit_path.with_suffix(".source.py")) != audit["analysis_source_sha256"] or
        audit["status"] != "completed_and_audited" or audit["physical_readouts_audited"] != 768 or
        audit["reported_bindings_audited"] != 896 or audit["pairs_per_model_audited"] != 96 or
        audit["registry_sha256"] != plan["source_registry_sha256"] or
        original["registry_sha256"] != plan["source_registry_sha256"] or
        data_fingerprint(registry) != plan["source_registry_sha256"] or
        audit["dataset_manifest_sha256"] != plan["source_dataset_manifest_sha256"] or
        audit["weights_sha256"] != plan["native_loaded_weights_sha256"]):
        raise ValueError("Completed prior transfer reference changed")
    if reexecute_transfer:
        from scripts.analyze_write_locality_adapter_transfer import audit_run
        if audit_run(Path(audit["source_run"])) != audit:
            raise ValueError("Actual complete prior transfer audit no longer reproduces")
    training = {}
    for name, spec in plan["models"].items():
        training[name] = validate_training_reference(spec)
        old_audit = json.loads(Path(spec["audit"]).read_text())
        old_plan = json.loads((Path(spec["training_run"]) / "plan.json").read_text())
        if (old_audit["arms"][spec["training_arm"]]["teacher_train_fit"] != spec["teacher_train_fit"] or
            old_plan["unique_training_examples"] != 864 or old_plan["maximum_optimizer_steps"] != 432 or
            old_plan["learning_rate"] != .001 or
            old_plan["optimized_input_token_budgets"]["432"] != 527808 or
            old_plan["supervised_answer_token_budgets"]["432"] != 3456 or
            training[name]["packages"] != original["references"]["event"]["packages"] or
            training[name]["model_path"] != original["model_path"]):
            raise ValueError("Training distribution/budget, fit caveat or native runtime changed")
    refs = {"event": original["references"]["event"], "training": training,
            "prior_timing": {"audit_path": str(audit_path), "audit_sha256": streaming_hash(audit_path),
                             "manifest_sha256": audit["manifest_sha256"]}}
    return original, contexts, registry, refs


def registered_inputs(path=DESTINATION):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    plan = read_plan()
    original, contexts, registry, refs = checked_references(plan)
    if (manifest["status"] != "registered_before_inference" or manifest["training"] or manifest["model_inference"] or
        not manifest["complete_prior_transfer_audit_reexecuted_at_registration"] or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or
        manifest["registry_sha256"] != data_fingerprint(registry) or manifest["references"] != refs or
        manifest["model_path"] != original["model_path"] or manifest["counts"] != plan["counts"] or
        manifest["source_dataset"] != plan["source_dataset"] or
        manifest["source_dataset_manifest_sha256"] != plan["source_dataset_manifest_sha256"] or
        json.loads((path / "plan.json").read_text()) != plan or
        set(manifest["source_sha256"]) != set(DATA_SOURCES) or
        datetime.fromisoformat(plan["registered_utc"]) >= datetime.fromisoformat(manifest["registered_utc"])):
        raise ValueError("Panel registration, actual source cohort or fixed references changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived panel data provider changed")
    return manifest, contexts, registry
