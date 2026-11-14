"""Explicit fresh-seed providers for the sealed native locality engine."""

import importlib.util
import json
from pathlib import Path
from statistics import mean

from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import (DATA_SOURCES as ORIGINAL_DATA_SOURCES, CELLS, ROLES,
                                           build_data, streaming_hash, token_registry, validate_data)
from sas_memory.write_locality_metrics import all_correct_units, paired

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/semantic_write_locality_replication_plan.json"
DATA_SOURCES = tuple(sorted(set(ORIGINAL_DATA_SOURCES) | {
    "sas_memory/locality_replication.py", "scripts/prepare_write_locality_replication.py",
    "configs/semantic_write_locality_replication_plan.json", "configs/semantic_write_locality_load_plan.json"}))
RUNTIME_SOURCES = tuple(sorted(set(DATA_SOURCES) | {
    "scripts/run_write_locality_replication.py", "scripts/analyze_write_locality_replication.py",
    "scripts/run_write_locality_load_screen.py", "scripts/analyze_write_locality_load_screen.py"}))


def read_plan():
    plan = json.loads(PROTOCOL.read_text())
    original_path = ROOT / plan["source_protocol"]
    original = json.loads(original_path.read_text())
    grid = {**original["development_grid"], "base_seeds": [97100, 97101, 97102, 97103]}
    if (plan["profile"] != "fixed_length_write_locality_independent_seed_replication" or
        streaming_hash(original_path) != plan["source_protocol_sha256"] or plan["development_grid"] != grid or
        streaming_hash(ROOT / plan["generation"]["module"]) != plan["generation"]["source_sha256"] or
        any(plan[key] != original[key] for key in ("model_config", "model_key", "model_class", "local_files_only", "reader_policy", "training", "adapters"))):
        raise ValueError("Replica changed the registered seed grid, generation recipe or model scope")
    expected = [("unrelated_B_at32", 32, "protected_current"), ("initial_A_at128", 128, "target_initial")]
    if plan["primary_contrasts"] != [
        {"id": label, "active_accounts": load, "axis": "A", "fixed_other_bit": 1, "role": role,
         "left_cell": "A0_C1", "right_cell": "A1_C1"} for label, load, role in expected]:
        raise ValueError("Replica primary contrasts changed")
    return plan


def discovery_reference(plan, *, reexecute=False):
    path = Path(plan["discovery_audit"])
    report = json.loads(path.read_text())
    run = Path(report["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    if (streaming_hash(path) != plan["discovery_audit_sha256"] or report["status"] != "completed_and_audited" or
        report["records_audited"] != 576 or not report["metrics"]["screen_signal"]["present"] or
        streaming_hash(run / "manifest.json") != report["manifest_sha256"] or
        report["manifest_sha256"] != plan["discovery_manifest_sha256"] or manifest["status"] != "completed" or
        streaming_hash(run / "validation.native.jsonl") != manifest["result_sha256"] or
        streaming_hash(path.with_suffix(".source.py")) != report["analysis_source_sha256"]):
        raise ValueError("Completed discovery audit or its source observations changed")
    for relative, digest in manifest["source_sha256"].items():
        if streaming_hash(run / "sources" / relative) != digest or streaming_hash(ROOT / relative) != digest:
            raise ValueError("Sealed discovery data/runtime implementation changed")
    if reexecute:
        from scripts.analyze_write_locality_load_screen import audit_run
        actual = audit_run(run)
        actual["analysis_source_sha256"] = streaming_hash(ROOT / "scripts/analyze_write_locality_load_screen.py")
        if actual != report:
            raise ValueError("Reexecuted complete discovery audit disagrees with pinned report")
    return report


def gate_descriptor(plan):
    report = discovery_reference(plan)
    return {"report_path": plan["discovery_audit"], "report_sha256": plan["discovery_audit_sha256"],
            "manifest_sha256": report["manifest_sha256"], "status": report["status"]}


def load_engine(relative, name):
    """Load an isolated copy; providers never mutate the historical module."""
    plan = read_plan()
    report = discovery_reference(plan)
    manifest = json.loads((Path(report["source_run"]) / "manifest.json").read_text())
    if streaming_hash(ROOT / relative) != manifest["source_sha256"][relative]:
        raise ValueError("Native engine is no longer the sealed discovery implementation")
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def evaluated_members(data, smoke):
    if not smoke:
        return data["validation"]
    grid = read_plan()["development_grid"]
    return [row for row in data["validation"] if row["case"]["seed"] == grid["base_seeds"][0] and
            row["case"]["value_version"] == 0 and row["case"]["active_accounts"] in (grid["active_accounts"][0], grid["active_accounts"][-1]) and
            row["variant"] in (CELLS[0], CELLS[-1])]


def registered_dataset(path):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    path = Path(path)
    plan = read_plan()
    manifest = json.loads((path / "manifest.json").read_text())
    data = json.loads((path / "data.json").read_text())
    registry = json.loads((path / "registry.json").read_text())
    reference = discovery_reference(plan)
    if (path != Path(plan["data_source"]) or manifest["status"] != "registered_before_inference" or
        manifest["training"] or manifest["model_inference"] or json.loads((path / "plan.json").read_text()) != plan or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or manifest["counts"] != validate_data(data, plan) or
        manifest["data_sha256"] != data_fingerprint(data) or manifest["registry_sha256"] != data_fingerprint(registry) or
        set(manifest["source_sha256"]) != set(DATA_SOURCES) or manifest["reference_gate"] != gate_descriptor(plan) or
        not manifest["complete_discovery_audit_reexecuted_at_registration"] or
        streaming_hash(path / "discovery_audit.json") != plan["discovery_audit_sha256"] or
        manifest["model_path"] != json.loads((ROOT / plan["model_config"]).read_text())[plan["model_key"]] or
        manifest["evaluation_keys_sha256"] != data_fingerprint([[r["case_id"], r["variant"], r["role"]] for r in registry["validation"]])):
        raise ValueError("Replica registration, true seed identities, data or gate changed")
    model = Path(manifest["model_path"])
    expected_config = {file.name for file in model.iterdir() if file.is_file() and
                       ("token" in file.name or file.name in ("config.json", "generation_config.json") or file.suffix == ".jinja")}
    if (set(manifest["tokenizer_sha256"]) != expected_config or not manifest["checkpoint_sha256"] or
        set(manifest["checkpoint_sha256"]) != {file.name for file in model.glob("*.safetensors")}):
        raise ValueError("Replica tokenizer/config/checkpoint coverage incomplete")
    for relative, digest in manifest["source_sha256"].items():
        if streaming_hash(path / "sources" / relative) != digest or streaming_hash(ROOT / relative) != digest:
            raise ValueError("Actual/archived replica data provider changed")
    for group in ("tokenizer_sha256", "checkpoint_sha256"):
        for name, digest in manifest[group].items():
            if streaming_hash(model / name) != digest:
                raise ValueError("Replica local model/tokenizer changed")
    rebuilt = token_registry(data, plan, tokenizer_runtime(str(model)))
    if rebuilt != registry or {row["prompt_tokens"] for row in registry["validation"]} != {2865}:
        raise ValueError("Replica token positions, sibling controls or complete prompt registry changed")
    if manifest["discovery_weights_sha256"] != reference["weights_sha256"]:
        raise ValueError("Replica native-model weight reference changed")
    return manifest, data, registry


def summarize(rows, plan=None):
    """Same cell/contrast definitions, parameterized by actual registered seeds."""
    plan = read_plan() if plan is None else plan
    grid = plan["development_grid"]
    if len(rows) != 576 or len({(r['case_id'], r['variant'], r['role']) for r in rows}) != 576:
        raise ValueError("Complete registered576-query grid required")
    if {row["seed"] for row in rows} != set(grid["base_seeds"]):
        raise ValueError("Metrics contain another cohort's seed identities")
    cells, contrasts, target_updates, units, streams = [], [], [], [], []
    for load in grid["active_accounts"]:
        members = [row for row in rows if row["active_accounts"] == load]
        if len(members) != 192:
            raise ValueError("Replica load cell incomplete")
        for cell in CELLS:
            for role in ROLES:
                group = [row for row in members if row["variant"] == cell and row["role"] == role]
                if len(group) != 16:
                    raise ValueError("Replica role/tail cell incomplete")
                cells.append({"active_accounts": load, "cell": cell, "role": role,
                              "correct": sum(row["greedy_exact"] for row in group), "examples": len(group),
                              "answer_nll": mean(row["answer_nll"] for row in group)})
        for bit in (0, 1):
            for role in ROLES[1:]:
                group = [row for row in members if row["role"] == role]
                contrasts.append({"active_accounts": load, "axis": "A", "fixed_other_bit": bit, "role": role,
                                  **paired(group, f"A0_C{bit}", f"A1_C{bit}")})
            for role in ROLES:
                group = [row for row in members if row["role"] == role]
                contrasts.append({"active_accounts": load, "axis": "C", "fixed_other_bit": bit, "role": role,
                                  **paired(group, f"A{bit}_C0", f"A{bit}_C1")})
            target_updates.append({"active_accounts": load, "fixed_C_bit": bit,
                                   **paired([row for row in members if row["role"] == ROLES[0]],
                                            f"A0_C{bit}", f"A1_C{bit}", invariant=False)})
        units.append({"active_accounts": load,
                      "individual": {"correct": sum(row["greedy_exact"] for row in members), "units": len(members)},
                      "all_three_roles": all_correct_units(members, ["case_id", "variant"]),
                      "all_four_values": all_correct_units(members, ["base_case_id", "variant", "role"]),
                      "all_twelve_cell_role_readouts": all_correct_units(members, ["case_id"]),
                      "all_48_value_cell_role_readouts": all_correct_units(members, ["base_case_id"])})
        for seed in grid["base_seeds"]:
            group = [row for row in members if row["seed"] == seed]
            if len(group) != 48:
                raise ValueError("Replica base-stream value/tail/role cluster incomplete")
            streams.append({"seed": seed, "active_accounts": load, "correct": sum(row["greedy_exact"] for row in group),
                            "examples": len(group), "answer_nll": mean(row["answer_nll"] for row in group)})
    signals = [row for row in contrasts if len(row["regressed_base_streams"]) >= 2]
    return {"examples": len(rows), "correct": sum(row["greedy_exact"] for row in rows),
            "cell_role_counts": cells, "preservation_contrasts": contrasts, "target_A_update_pairs": target_updates,
            "functional_units_by_load": units, "base_stream_results": streams,
            "screen_signal": {"present": bool(signals), "contrasts": signals,
                              "definition": "The same axis, fixed-other-bit, role and load regresses in at least two distinct base streams. Multiple value siblings from one stream count once.",
                              "interpretation": "Descriptive development screen only; no significance, recurrent-damage or architecture-necessity claim."}}


def primary_recurrence(metrics, plan=None):
    plan = read_plan() if plan is None else plan
    result = []
    for spec in plan["primary_contrasts"]:
        matches = [row for row in metrics["preservation_contrasts"] if
                   all(row[key] == spec[key] for key in ("active_accounts", "axis", "fixed_other_bit", "role"))]
        if len(matches) != 1:
            raise ValueError("Missing or duplicate registered primary contrast")
        result.append({"id": spec["id"], **matches[0], "recurs": len(matches[0]["regressed_base_streams"]) >= 2})
    return {"contrasts": result, "event_path_gate_passed": any(row["recurs"] for row in result),
            "interpretation": "Apply only the two registered primaries; secondary signals cannot replace failed primaries."}
