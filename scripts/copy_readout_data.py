"""Frozen COPY inputs and program-derived, equal-budget historical KV masks."""

from bisect import bisect_left
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.kv_readout import evidence_token_keep
from sas_memory.oracle import evidence_slice
from sas_memory.tasks import INSTRUCTIONS, Operation, execute
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.copy_transfer_data import MODELS, ROLES, registered_inputs as source_inputs, safety_samples, selected_inputs
from scripts.run_controller_trial import prompt_for
from scripts.run_copy_dependency_transfer import RUNTIME_SOURCES as SOURCE_RUNTIME, actual_prompt

PROTOCOL = ROOT / "configs/copy_readout_sufficiency_plan.json"
PROTOCOL_SHA256 = "20c576579c9b076bc56d3f5dd66413ef955235033bf228afaa34fab0f9bb080b"
DESTINATION = Path("/media/990Pro/Code/SAS/datasets/memory_update/copy-readout-sufficiency-v1")
REGIMES = ("oracle", "nonslice")
DATA_SOURCES = tuple(sorted(set(SOURCE_RUNTIME) | {
    "configs/copy_readout_sufficiency_plan.json", "scripts/copy_readout_data.py", "scripts/prepare_copy_readout_sufficiency.py",
    "sas_memory/kv_readout.py", "sas_memory/oracle.py",
}))


def read_plan():
    plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or plan["training"] or plan["new_adapters"] or
        not plan["local_files_only"] or plan["model_key"] != "it" or plan["models"] != list(MODELS) or plan["regimes"] != list(REGIMES)):
        raise ValueError("Frozen readout diagnostic scope changed")
    return plan


def checked_source(plan, reexecute=False):
    source_plan_path = Path(plan["source_transfer_plan"])
    source_plan = json.loads(source_plan_path.read_text())
    source_path = Path(plan["source_dataset"])
    audit_path = Path(plan["source_audit"])
    audit = json.loads(audit_path.read_text())
    run = Path(plan["source_run"])
    run_manifest = json.loads((run / "manifest.json").read_text())
    registered, cases, registry = source_inputs(source_path)
    if (streaming_hash(source_plan_path) != plan["source_transfer_plan_sha256"] or
        streaming_hash(audit_path) != plan["source_audit_sha256"] or audit["status"] != "completed_and_audited" or
        audit["physical_readouts_audited"] != 2688 or audit["reported_bindings_audited"] != 3456 or
        audit["metrics"]["descriptive_shared_candidate_gate"]["passed"] or
        streaming_hash(run / "manifest.json") != plan["source_manifest_sha256"] or
        audit["manifest_sha256"] != plan["source_manifest_sha256"] or audit["source_run"] != str(run) or
        audit["weights_sha256"] != plan["native_loaded_weights_sha256"] or source_plan["packages"] != plan["packages"] or
        registered["model_path"] != plan["model_path"] or
        streaming_hash(source_path / "manifest.json") != plan["source_dataset_manifest_sha256"] or
        registered["registry_sha256"] != plan["source_registry_sha256"] or data_fingerprint(registry) != plan["source_registry_sha256"] or
        streaming_hash(audit_path.with_suffix(".source.py")) != audit["analysis_source_sha256"]):
        raise ValueError("Complete fixed COPY transfer reference changed")
    for name, digest in run_manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Actual/archived fixed COPY reference implementation changed")
    rows = {}
    for name in MODELS:
        path = run / (name + ".jsonl")
        if streaming_hash(path) != run_manifest["models"][name]["results_sha256"]:
            raise ValueError("Actual complete full-history reference observations changed")
        rows[name] = [json.loads(line) for line in path.read_text().splitlines()]
    if reexecute:
        from scripts.analyze_copy_dependency_transfer import audit_run
        if audit_run(run) != audit:
            raise ValueError("Complete2688-query transfer audit no longer reproduces")
    refs = {"source_audit": str(audit_path), "source_audit_sha256": streaming_hash(audit_path),
        "source_manifest_sha256": audit["manifest_sha256"], "source_registry_sha256": registered["registry_sha256"],
        "training": registered["references"]["training"], "prior_shared_candidate_passed": False,
        "prior_repeat_and_history_primaries_passed": False}
    return source_plan, cases, registry, rows, refs


def nonslice_keep(oracle, prelude):
    if (not 0 < prelude < len(oracle) or any(type(x) is not bool for x in oracle) or not all(oracle[:prelude])):
        raise ValueError("Exact boolean mask and retained prelude required")
    evidence = [i for i in range(prelude, len(oracle)) if oracle[i]]
    pool = [i for i in range(prelude, len(oracle)) if not oracle[i]]
    if not evidence or len(pool) < len(evidence):
        raise ValueError("Insufficient non-slice tokens for an equal-budget control")
    def distance(i):
        index = bisect_left(evidence, i)
        return min(abs(i - evidence[j]) for j in (index - 1, index) if 0 <= j < len(evidence))
    selected = set(sorted(pool, key=lambda i: (distance(i), i))[:len(evidence)])
    return [i < prelude or i in selected for i in range(len(oracle))]


def build_masks(cases, registry, runtime):
    case_map, masks = {c["id"]: c for c in cases}, {}
    for q in registry["physical"]:
        ref = q["representative"]
        case = case_map[ref["case_id"]]
        original = next(v for v in case["variants"][ref["variant"]]["queries"] if v["role"] == ref["role"])
        events = [Operation(**e) for e in case["prefix"] + case["variants"][ref["variant"]]["tail"]]
        selected = evidence_slice(events, original["entity"], original["version"])
        if str(execute(selected)[original["version"]][original["entity"]]) != q["answer"]:
            raise ValueError("Executable oracle evidence does not reproduce own-world gold")
        identities = {id(e) for e in selected}
        indices = [i for i, e in enumerate(events) if id(e) in identities]
        ledger = actual_prompt(runtime, case_map, q)
        prompt = ledger + runtime.query_suffix(q["question"])
        oracle = evidence_token_keep(runtime.tokenizer, prompt, ledger, runtime.prompt_start + INSTRUCTIONS,
                                     [e.text() for e in events], indices)
        control = nonslice_keep(oracle, q["prelude_tokens"])
        if len(oracle) != q["ledger_tokens"] or sum(oracle) != sum(control) or any(
            a and b for a, b in zip(oracle[q["prelude_tokens"]:], control[q["prelude_tokens"]:])):
            raise ValueError("Historical length, token budget or complement control changed")
        masks[q["id"]] = {"id": q["id"], "oracle_event_indices": indices, "event_count": len(events),
            "ledger_tokens": len(oracle), "prelude_tokens": q["prelude_tokens"],
            "retained_record_tokens": sum(oracle) - q["prelude_tokens"],
            "keep_bits": {r: "".join("1" if b else "0" for b in keep) for r, keep in zip(REGIMES, (oracle, control))}}
    return masks


def sample_ids(cases, registry):
    selected = set(safety_samples(registry))
    for case, variant in ((cases[0], "control"), (cases[-1], "treatment")):
        selected.update(b["physical_id"] for b in registry["bindings"] if b["case_id"] == case["id"] and
                        b["variant"] == variant and b["role"] in ROLES[1:])
    result = [q["id"] for q in registry["physical"] if q["id"] in selected]
    if len(result) != 10:
        raise ValueError("Prescribed all-role safety sample changed")
    return result


def registered_inputs(path=DESTINATION):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    path = Path(path)
    plan = read_plan()
    source_plan, cases, registry, rows, refs = checked_source(plan)
    manifest = json.loads((path / "manifest.json").read_text())
    masks = json.loads((path / "masks.json").read_text())
    if (manifest["status"] != "registered_before_inference" or manifest["training"] or manifest["model_inference"] or
        not manifest["complete_source2688_query_audit_reexecuted"] or manifest["counts"] != plan["counts"] or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or manifest["references"] != refs or
        manifest["mask_registry_sha256"] != data_fingerprint(masks) or manifest["sample_query_ids"] != sample_ids(cases, registry) or
        json.loads((path / "plan.json").read_text()) != plan or set(manifest["source_sha256"]) != set(DATA_SOURCES) or
        datetime.fromisoformat(manifest["registered_utc"]) <= datetime.fromisoformat(plan["registered_utc"])):
        raise ValueError("Frozen readout registration or complete references changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived readout data implementation changed")
    if build_masks(cases, registry, tokenizer_runtime(plan["model_path"])) != masks:
        raise ValueError("All actual original tokens or program-derived masks changed")
    return manifest, source_plan, cases, registry, masks, rows
