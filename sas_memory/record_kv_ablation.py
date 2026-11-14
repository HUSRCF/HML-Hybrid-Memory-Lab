"""Boundary-only masks: keep instructions, block historical ledger-record KV."""

import hashlib
import json
from pathlib import Path

from sas_memory.counterbalance_data import encode_counterbalance, file_digest, locked_training_sources
from sas_memory.counterbalance_tasks import ORDERS, validate_counterbalance_cases
from sas_memory.tasks import INSTRUCTIONS
from sas_memory.training_data import data_fingerprint


def record_blocked_mask(runtime, ledger, prompt):
    prelude = runtime.prompt_start + INSTRUCTIONS
    if not ledger.startswith(prelude) or ledger == prelude or not prompt.startswith(ledger):
        raise ValueError("Expected the complete prelude, records and question")
    prefix, history, whole = runtime.ids(prelude), runtime.ids(ledger), runtime.ids(prompt)
    if history[:len(prefix)] != prefix or whole[:len(history)] != history:
        raise ValueError("Tokenizer crosses the prelude or question boundary")
    if not 0 < len(prefix) < len(history) < len(whole):
        raise ValueError("Invalid prelude/record/question token intervals")
    keep = [i < len(prefix) for i in range(len(history))]
    return {"prelude_tokens": len(prefix), "ledger_tokens": len(history), "prompt_tokens": len(whole),
            "prompt_ids_sha256": hashlib.sha256(json.dumps(whole).encode()).hexdigest(),
            "keep_bits": "".join("1" if value else "0" for value in keep),
            "keep_sha256": hashlib.sha256(bytes(keep)).hexdigest(),
            "blocked_record_tokens": len(history) - len(prefix)}


def build_record_registry(runtime, cases):
    rows = []
    for case in cases:
        for order in ORDERS:
            ledger, prompt, original = encode_counterbalance(runtime, case, order)
            record = record_blocked_mask(runtime, ledger, prompt)
            if any(record[key] != original[key] for key in ("ledger_tokens", "prompt_tokens", "prompt_ids_sha256")):
                raise ValueError("Record blocking changed the registered full input")
            rows.append({"case_id": case["id"], "variant": order, **record})
    return rows


def reference_protocol(plan, root):
    reference = Path(plan["reference_run"])
    manifest = json.loads((reference / "manifest.json").read_text())
    original_plan = json.loads((reference / "plan.json").read_text())
    cases = json.loads((reference / "cases.json").read_text())
    if manifest["status"] != "completed" or manifest["smoke"] or manifest["training"]:
        raise ValueError("Reference evaluation is incomplete or not a frozen full run")
    if manifest["completed_modes"] != plan["models"] or manifest["modes"] != plan["models"]:
        raise ValueError("Reference checkpoints differ from the ablation plan")
    if data_fingerprint(cases) != plan["expected_cases_sha256"] or manifest["cases_sha256"] != plan["expected_cases_sha256"]:
        raise ValueError("Reference cases changed")
    if validate_counterbalance_cases(cases)["contexts"] != plan["contexts_per_model"]:
        raise ValueError("Wrong number of registered contexts")
    sources = locked_training_sources(original_plan, root, {case["seed"] for case in cases})
    if sources != manifest["checkpoint_sources"]:
        raise ValueError("Reference checkpoints or training sources changed")
    files = {model: {regime: file_digest(reference / f"{model}.{regime}.jsonl") for regime in ("raw_full", "oracle")}
             for model in plan["models"]}
    return reference, manifest, original_plan, cases, sources, files
