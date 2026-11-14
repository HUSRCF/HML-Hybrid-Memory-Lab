"""Exact existing COPY cohort and deduplicated, locally tokenized transfer inputs."""

from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.generalization_tasks import build_generalization_cases, validate_generalization_cases
from sas_memory.tasks import INSTRUCTIONS
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.history_no_repeat_data import model_artifacts
from scripts.locality_transfer_data import validate_training_reference
from scripts.run_controller_trial import prompt_for
from scripts.run_write_locality_control_panel import RUNTIME_SOURCES as SEALED_SOURCES
from scripts.run_semantic_overwrite_history import RUNTIME_SOURCES as NATIVE_SOURCES

PROTOCOL = ROOT / "configs/copy_dependency_transfer_plan.json"
PROTOCOL_SHA256 = "84d864ce4ad6373059fc49a559a18b5b39d497bbcaa6544afef5bb6fb4888772"
DESTINATION = Path("/media/990Pro/Code/SAS/datasets/memory_update/copy-dependency-transfer-v1")
MODELS = ("native", "shared", "independent", "local", "local_question", "attention_only", "gdn_all_tokens", "gdn_question_only")
VARIANTS = ("before", "control", "treatment")
ROLES = ("target_current", "protected_current", "target_initial")
DATA_SOURCES = tuple(sorted(set(SEALED_SOURCES) | set(NATIVE_SOURCES) | {
    "configs/copy_dependency_transfer_plan.json", "scripts/copy_transfer_data.py",
    "scripts/prepare_copy_dependency_transfer.py", "scripts/history_no_repeat_data.py",
    "sas_memory/generalization_tasks.py", "scripts/analyze_blocked_reader_adaptation.py",
}))


def read_plan():
    plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or plan["training"] or plan["new_adapters"] or
        not plan["local_files_only"] or plan["reader_policy"] != "full" or plan["model_key"] != "it" or
        list(plan["models"]) != list(MODELS) or plan["models"]["native"] is not None or
        plan["variants"] != list(VARIANTS) or plan["roles"] != list(ROLES)):
        raise ValueError("Frozen COPY transfer scope changed")
    return plan


def validate_cases(cases, plan):
    if (cases != build_generalization_cases() or
        data_fingerprint(cases) != plan["source_cases_canonical_sha256"] or
        streaming_hash(Path(plan["source_cases"])) != plan["source_cases_file_sha256"] or
        json.loads(Path(plan["source_cases"]).read_text()) != cases):
        raise ValueError("Existing COPY cohort, executable truths or complete selection changed")
    counts = validate_generalization_cases(cases)
    if counts["cases"] != 48 or counts["base_cases"] != 8:
        raise ValueError("Incomplete existing depth/wording grid")
    for case in cases:
        queries = {v: {q["role"]: q for q in case["variants"][v]["queries"]} for v in VARIANTS}
        if queries["control"][ROLES[0]]["answer"] == queries["treatment"][ROLES[0]]["answer"]:
            raise ValueError("COPY order must change current-target truth")
        for role in ROLES[1:]:
            if len({queries[v][role]["answer"] for v in VARIANTS}) != 1:
                raise ValueError("COPY order changed a protected or historical answer")
    return counts


def reference_gate(plan, reexecute=False):
    path = Path(plan["previous_negative_audit"])
    report = json.loads(path.read_text())
    if (streaming_hash(path) != plan["previous_negative_audit_sha256"] or
        report["status"] != "completed_and_audited" or report["records_audited"] != 384 or
        report["metrics"]["primary_signal"]["cohort_passed"] != {"discovery": False, "replica": False} or
        report["reference_gate"]["previous_history_primary_passed"] or
        report["weights_sha256"] != plan["native_loaded_weights_sha256"] or report["packages"] != plan["packages"]):
        raise ValueError("Prior negative native evidence changed")
    if reexecute:
        from scripts.analyze_semantic_overwrite_history import audit_run
        if audit_run(Path(report["source_run"])) != report:
            raise ValueError("Complete previous native audit no longer reproduces")
    training = {name: validate_training_reference(spec) for name, spec in plan["models"].items() if spec}
    if any(r["model_path"] != plan["model_path"] or r["packages"] != plan["packages"] for r in training.values()):
        raise ValueError("Fixed adapters were trained with a different native runtime")
    return {"previous_negative_audit": str(path), "previous_negative_audit_sha256": streaming_hash(path),
            "previous_repeat_primary_passed": False, "previous_history_primary_passed": False, "training": training}


def build_registry(cases, runtime):
    physical, bindings = {}, []
    for case in cases:
        for variant in VARIANTS:
            for role in ROLES:
                query = next(q for q in case["variants"][variant]["queries"] if q["role"] == role)
                ledger, prompt = prompt_for(runtime, case, variant, query)
                ids, ledger_ids = runtime.ids(prompt), runtime.ids(ledger)
                answer_ids = runtime.continuation_ids(prompt, query["answer"])
                if ids[:len(ledger_ids)] != ledger_ids or len(answer_ids) != 3 or len(ids) > 4096:
                    raise ValueError("COPY prompt boundary, numeric answer or context budget changed")
                key = data_fingerprint(ids)
                candidate = {"id": key, "source_key": data_fingerprint(ledger_ids),
                    "prompt_ids_sha256": key, "prompt_tokens": len(ids), "ledger_tokens": len(ledger_ids),
                    "question_ids": ids[len(ledger_ids):], "question_tokens": len(ids) - len(ledger_ids),
                    "question": query["question"], "answer": query["answer"], "answer_ids": answer_ids,
                    "prelude_tokens": len(runtime.ids(runtime.prompt_start + INSTRUCTIONS))}
                if key in physical and physical[key]["tokens"] != candidate:
                    raise ValueError("Identical physical prompt has conflicting truth or token registration")
                physical.setdefault(key, {"tokens": candidate, "representative": {"case_id": case["id"], "variant": variant, "role": role}})
                bindings.append({"case_id": case["id"], "base_case_id": case["base_case_id"],
                    "seed": case["seed"], "distractor_updates": case["distractor_updates"],
                    "dependency_depth": case["dependency_depth"], "question_style": case["question_style"],
                    "variant": variant, "role": role, "physical_id": key})
    if len(physical) != 336 or len(bindings) != 432:
        raise ValueError("COPY physical/binding deduplication coverage changed")
    return {"physical": [{**entry["tokens"], "representative": entry["representative"]} for entry in physical.values()],
            "bindings": bindings, "smoke_case_ids": [cases[0]["id"], cases[-1]["id"]]}


def selected_inputs(registry, smoke=False):
    bindings = [b for b in registry["bindings"] if not smoke or b["case_id"] in registry["smoke_case_ids"]]
    ids = {b["physical_id"] for b in bindings}
    physical = [q for q in registry["physical"] if q["id"] in ids]
    return physical, bindings


def safety_samples(registry):
    bindings = [b for b in registry["bindings"] if b["case_id"] in registry["smoke_case_ids"] and b["role"] == ROLES[0]]
    query_ids = [b["physical_id"] for b in bindings]
    if len(query_ids) != 6 or len(set(query_ids)) != 6:
        raise ValueError("Prescribed six source/order safety samples changed")
    return query_ids


def registered_inputs(path=DESTINATION):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    plan = read_plan()
    cases, registry = (json.loads((path / name).read_text()) for name in ("cases.json", "registry.json"))
    validate_cases(cases, plan)
    if (manifest["status"] != "registered_before_inference" or manifest["training"] or manifest["model_inference"] or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or manifest["cases_sha256"] != data_fingerprint(cases) or
        manifest["registry_sha256"] != data_fingerprint(registry) or manifest["references"] != reference_gate(plan) or
        not manifest["complete_previous_native_audit_reexecuted"] or manifest["counts"] != plan["counts"] or
        manifest["model_path"] != plan["model_path"] or json.loads((path / "plan.json").read_text()) != plan or
        set(manifest["source_sha256"]) != set(DATA_SOURCES) or
        datetime.fromisoformat(manifest["registered_utc"]) <= datetime.fromisoformat(plan["registered_utc"])):
        raise ValueError("COPY transfer registration or fixed references changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived COPY transfer data source changed")
    tokenizer, checkpoint = model_artifacts(plan["model_path"])
    if manifest["tokenizer_sha256"] != tokenizer or manifest["checkpoint_sha256"] != checkpoint:
        raise ValueError("Actual native tokenizer/config/checkpoint changed")
    if build_registry(cases, tokenizer_runtime(plan["model_path"])) != registry:
        raise ValueError("Complete actual COPY token registry no longer reconstructs")
    return manifest, cases, registry
