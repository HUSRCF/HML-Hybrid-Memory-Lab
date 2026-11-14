"""Whole-record evidence removal, with a nearby equal-token irrelevant control."""

from bisect import bisect_right
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.oracle import evidence_slice
from sas_memory.tasks import INSTRUCTIONS, Operation, execute
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.copy_readout_data import MODELS, ROLES, registered_inputs as prior_inputs, sample_ids, safety_samples, selected_inputs
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_copy_readout_sufficiency import RUNTIME_SOURCES as PRIOR_RUNTIME

PROTOCOL = ROOT / "configs/copy_evidence_necessity_plan.json"
PROTOCOL_SHA256 = "fcba9df3ded485c71bb030cb6fb5ce755f255f2c51f6d320006368c27fa80407"
DESTINATION = Path("/media/990Pro/Code/SAS/datasets/memory_update/copy-evidence-necessity-v1")
REGIMES = ("remove_support", "remove_matched_other")
DATA_SOURCES = tuple(sorted(set(PRIOR_RUNTIME) | {
    "configs/copy_evidence_necessity_plan.json", "scripts/copy_necessity_data.py",
    "scripts/prepare_copy_evidence_necessity.py",
}))


def read_plan():
    plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or plan["training"] or plan["new_adapters"] or
        not plan["local_files_only"] or plan["model_key"] != "it" or
        plan["models"] != list(MODELS) or plan["regimes"] != list(REGIMES)):
        raise ValueError("Frozen necessity diagnostic scope changed")
    return plan


def checked_previous(plan, reexecute=False):
    registered, source_plan, cases, registry, _, full_rows = prior_inputs()
    path = Path(plan["previous_readout_audit"])
    audit = json.loads(path.read_text())
    run = Path(audit["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    if (streaming_hash(path) != plan["previous_readout_audit_sha256"] or audit["status"] != "completed_and_audited" or
        audit["new_physical_readouts_audited"] != 5376 or audit["new_related_bindings_audited"] != 6912 or
        not audit["metrics"]["frozen_prefix_sufficiency"]["passed"] or not audit["metrics"]["address_specific_gain"]["passed"] or
        not audit["metrics"]["prior_shared_history_repeat_primaries_remain_failed"] or
        audit["references"] != registered["references"] or audit["weights_sha256"] != plan["native_loaded_weights_sha256"] or
        streaming_hash(run / "manifest.json") != audit["manifest_sha256"] or
        streaming_hash(path.with_suffix(".source.py")) != audit["analysis_source_sha256"] or
        any(v["physical_correct"] != 0 for v in audit["metrics"]["models_by_regime"]["nonslice"].values())):
        raise ValueError("Complete previous sufficiency evidence or negative decisions changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Previous actual/archived diagnostic source changed")
    for model in MODELS:
        for regime, digest in manifest["models"][model]["results_sha256"].items():
            if streaming_hash(run / f"{model}.{regime}.jsonl") != digest:
                raise ValueError("Previous actual complete readouts changed")
    if reexecute:
        from scripts.analyze_copy_readout_sufficiency import audit_run
        if audit_run(run) != audit:
            raise ValueError("Complete previous5376-query audit no longer reproduces")
    refs = {**registered["references"], "previous_readout_audit": str(path), "previous_readout_audit_sha256": streaming_hash(path),
            "previous_readout_manifest_sha256": audit["manifest_sha256"], "previous_sufficiency_passed": True,
            "previous_address_gate_passed": True, "previous_nonslice_physical_correct_all_models": 0}
    return source_plan, cases, registry, full_rows, refs


def event_token_spans(runtime, ledger, prompt, events, prelude_tokens):
    """Partition original record tokens into whole events, including punctuation."""
    prelude = runtime.prompt_start + INSTRUCTIONS
    texts = [e.text() for e in events]
    if ledger != prelude + "".join(texts) or not prompt.startswith(ledger):
        raise ValueError("Original events do not reconstruct ledger")
    encoded = runtime.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = runtime.ids(ledger)
    if encoded["input_ids"][:len(ids)] != ids or prelude_tokens != len(runtime.ids(prelude)):
        raise ValueError("Original token or prelude boundaries changed")
    starts, ends, cursor = [], [], len(prelude)
    for text in texts:
        starts.append(cursor)
        cursor += len(text)
        ends.append(cursor)
    assigned = [[] for _ in events]
    for token, (left, right) in enumerate(encoded["offset_mapping"][:len(ids)]):
        if token < prelude_tokens:
            if right > len(prelude):
                raise ValueError("Token crosses prelude/event boundary")
            continue
        index = bisect_right(starts, left) - 1
        if index < 0 or right <= left or left < starts[index] or right > ends[index]:
            raise ValueError("Token crosses whole-event boundary")
        assigned[index].append(token)
    spans = []
    cursor = prelude_tokens
    for tokens in assigned:
        if not tokens or tokens != list(range(cursor, cursor + len(tokens))):
            raise ValueError("Whole event token spans must partition complete history")
        spans.append([tokens[0], tokens[-1] + 1])
        cursor += len(tokens)
    if cursor != len(ids):
        raise ValueError("Record-token partition incomplete")
    return spans


def choose_records(events, selected_indices, spans):
    if not selected_indices or selected_indices != sorted(set(selected_indices)) or len(spans) != len(events):
        raise ValueError("Complete ordered evidence and event spans required")
    leaves = [i for i in selected_indices if events[i].kind == "SET"]
    if len(leaves) != 1 or leaves[0] != selected_indices[0]:
        raise ValueError("Expected exactly one earliest SET evidence leaf")
    leaf = leaves[0]
    width = spans[leaf][1] - spans[leaf][0]
    entities = set().union(*(events[i].reads | events[i].writes for i in selected_indices))
    pool = [i for i, e in enumerate(events) if i not in selected_indices and e.kind == "SET" and
            not (e.reads | e.writes) & entities and spans[i][1] - spans[i][0] == width]
    if width < 1 or not pool:
        raise ValueError("No equal-token semantically irrelevant whole SET control")
    other = min(pool, key=lambda i: (abs(sum(spans[i]) - sum(spans[leaf])), i))
    return leaf, other, sorted(entities)


def build_masks(cases, registry, runtime):
    case_map, masks = {c["id"]: c for c in cases}, {}
    for q in registry["physical"]:
        ref = q["representative"]
        case = case_map[ref["case_id"]]
        original = next(v for v in case["variants"][ref["variant"]]["queries"] if v["role"] == ref["role"])
        events = [Operation(**e) for e in case["prefix"] + case["variants"][ref["variant"]]["tail"]]
        selected = evidence_slice(events, original["entity"], original["version"])
        if str(execute(selected)[original["version"]][original["entity"]]) != q["answer"]:
            raise ValueError("Symbolic evidence does not reproduce own-world gold")
        identities = {id(e) for e in selected}
        indices = [i for i, e in enumerate(events) if id(e) in identities]
        ledger = actual_prompt(runtime, case_map, q)
        spans = event_token_spans(runtime, ledger, ledger + runtime.query_suffix(q["question"]), events, q["prelude_tokens"])
        leaf, other, entities = choose_records(events, indices, spans)
        try:
            value = str(execute([events[i] for i in indices if i != leaf])[original["version"]][original["entity"]])
        except KeyError:
            value = None
        if value == q["answer"]:
            raise ValueError("Selected SET leaf is not symbolically necessary")
        keeps = {regime: "".join("0" if spans[index][0] <= t < spans[index][1] else "1" for t in range(q["ledger_tokens"]))
                 for regime, index in zip(REGIMES, (leaf, other))}
        width = spans[leaf][1] - spans[leaf][0]
        if any(s.count("0") != width or s[:q["prelude_tokens"]] != "1" * q["prelude_tokens"] for s in keeps.values()):
            raise ValueError("Unequal removal budget or prelude changed")
        masks[q["id"]] = {"id": q["id"], "oracle_event_indices": indices, "evidence_entities": entities,
            "support_event_index": leaf, "other_event_index": other, "support_token_span": spans[leaf], "other_token_span": spans[other],
            "support_leaf_symbolically_necessary": True, "event_count": len(events), "ledger_tokens": q["ledger_tokens"],
            "prelude_tokens": q["prelude_tokens"], "removed_record_tokens": width,
            "retained_record_tokens": q["ledger_tokens"] - q["prelude_tokens"] - width,
            "twice_token_center_distance": abs(sum(spans[other]) - sum(spans[leaf])), "keep_bits": keeps}
    return masks


def registered_inputs(path=DESTINATION):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    path = Path(path)
    plan = read_plan()
    source_plan, cases, registry, rows, refs = checked_previous(plan)
    manifest = json.loads((path / "manifest.json").read_text())
    masks = json.loads((path / "masks.json").read_text())
    if (manifest["status"] != "registered_before_inference" or manifest["training"] or manifest["model_inference"] or
        not manifest["complete_previous5376_query_audit_reexecuted"] or manifest["counts"] != plan["counts"] or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or manifest["references"] != refs or
        manifest["mask_registry_sha256"] != data_fingerprint(masks) or manifest["sample_query_ids"] != sample_ids(cases, registry) or
        json.loads((path / "plan.json").read_text()) != plan or set(manifest["source_sha256"]) != set(DATA_SOURCES) or
        datetime.fromisoformat(manifest["registered_utc"]) <= datetime.fromisoformat(plan["registered_utc"])):
        raise ValueError("Frozen whole-record necessity registration changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived necessity data implementation changed")
    if build_masks(cases, registry, tokenizer_runtime(plan["model_path"])) != masks:
        raise ValueError("Whole original event tokens, semantic support or matched controls changed")
    return manifest, source_plan, cases, registry, masks, rows
