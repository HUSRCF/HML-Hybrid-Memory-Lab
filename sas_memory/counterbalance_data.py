"""Token-level registration and locked checkpoint provenance for COPY controls."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from sas_memory.counterbalance_tasks import MASKS, ORDERS, counterbalance_indices
from sas_memory.kv_readout import evidence_token_keep
from sas_memory.tasks import INSTRUCTIONS, Operation, render
from sas_memory.training_data import data_fingerprint


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokenizer_runtime(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    marker = "SAS_UNIQUE_CONTENT_MARKER"
    template = tokenizer.apply_chat_template([{"role": "user", "content": marker}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    start, end = template.split(marker)
    return SimpleNamespace(tokenizer=tokenizer, prompt_start=start,
                           ids=lambda text: tokenizer.encode(text, add_special_tokens=False),
                           query_suffix=lambda question: f"Question: {question}\nAnswer:" + end)


def encode_counterbalance(runtime, case, order):
    variant = case["variants"][order]
    query = variant["queries"][0]
    events = [Operation(**e) for e in case["prefix"] + variant["tail"]]
    prelude = runtime.prompt_start + INSTRUCTIONS
    ledger = prelude + render(case["prefix"] + variant["tail"])
    prompt = ledger + runtime.query_suffix(query["question"])
    ids, prefix_ids = runtime.ids(prompt), runtime.ids(ledger)
    full_answer = runtime.ids(prompt + query["answer"])
    if ids[:len(prefix_ids)] != prefix_ids or full_answer[:len(ids)] != ids:
        raise ValueError("Counterbalance tokenizer boundary changed")
    masks = {}
    for kind in MASKS:
        indices = counterbalance_indices(case, order, kind)
        keep = evidence_token_keep(runtime.tokenizer, prompt, ledger, prelude, [e.text() for e in events], indices)
        masks[kind] = {"selected_event_indices": indices, "keep_bits": "".join("1" if x else "0" for x in keep),
                       "kept_history_tokens": sum(keep), "keep_sha256": hashlib.sha256(bytes(keep)).hexdigest()}
    if masks["extra_source"]["kept_history_tokens"] != masks["extra_unrelated"]["kept_history_tokens"]:
        raise ValueError("Source and unrelated extras have unequal token budgets")
    row = {"case_id": case["id"], "variant": order, "ledger_tokens": len(prefix_ids), "prompt_tokens": len(ids),
           "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
           "answer_ids": full_answer[len(ids):], "masks": masks}
    return ledger, prompt, row


def build_counterbalance_registry(runtime, cases):
    registry = []
    for case in cases:
        pair = {order: encode_counterbalance(runtime, case, order)[2] for order in ORDERS}
        left, right = (pair[order] for order in ORDERS)
        if left["ledger_tokens"] != right["ledger_tokens"] or left["prompt_tokens"] != right["prompt_tokens"]:
            raise ValueError("Swapped SET positions changed token lengths")
        for kind, counterpart in (("extra_source", "extra_unrelated"), ("extra_unrelated", "extra_source")):
            # The two additional SETs occupy the exact same token intervals when
            # their semantic roles are swapped, not merely similar distances.
            def extra_positions(row, label):
                oracle, extended = row["masks"]["oracle"]["keep_bits"], row["masks"][label]["keep_bits"]
                if any(a == "1" and b == "0" for a, b in zip(oracle, extended)):
                    raise ValueError("Extra-record mask removed necessary evidence")
                return [i for i, (a, b) in enumerate(zip(oracle, extended)) if a == "0" and b == "1"]
            if extra_positions(left, kind) != extra_positions(right, counterpart):
                raise ValueError("Role-swapped extra SETs have different absolute token positions")
        registry.extend(pair.values())
    return registry


def locked_training_sources(plan, root, case_seeds):
    sources, common = {}, None
    for label, spec in plan["checkpoint_sources"].items():
        source = Path(spec["training_run"])
        manifest = json.loads((source / "manifest.json").read_text())
        mode, config = spec["mode"], manifest["config"]
        checkpoint = source / mode / "controller.safetensors"
        if manifest["status"] != "completed" or mode not in manifest["completed_modes"]:
            raise ValueError("Incomplete training checkpoint")
        if config["initialization_seed"] != spec["initialization_seed"] or config.get("token_policy") != "question" or config["model"] != "it":
            raise ValueError("Wrong frozen checkpoint protocol")
        if file_digest(checkpoint) != spec["checkpoint_sha256"]:
            raise ValueError("Locked checkpoint changed")
        data = json.loads((source / "data.json").read_text())
        if data_fingerprint(data) != manifest["data_sha256"]:
            raise ValueError("Training data changed")
        used = {entry["case"]["seed"] for entry in data["train"]}
        used.update(case["seed"] for split in ("validation", "test") for case in data[split])
        if set(case_seeds) & used:
            raise ValueError("Fresh cohort overlaps checkpoint training/development seeds")
        encoded = json.loads((source / "tokenized_train.json").read_text())
        tokens_digest = data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded])
        identity = (config["model"], config["group_index"], config["modes"][mode], config.get("adapt_gdn_keys"),
                    config["steps"], config["learning_rate"], config["gradient_accumulation"], manifest["data_sha256"], tokens_digest)
        if common is not None and identity != common:
            raise ValueError("Initialization checkpoints have unmatched training budgets")
        common = identity
        for relative in ("sas_memory/runtime.py", "sas_memory/controllers.py"):
            if file_digest(root / relative) != manifest["source_sha256"][relative]:
                raise ValueError("Runtime/controller differs from checkpoint training")
        sources[label] = {**spec, "config": config, "training_manifest_sha256": file_digest(source / "manifest.json"),
                          "training_data_sha256": manifest["data_sha256"], "training_tokens_sha256": tokens_digest}
    if list(sources) != plan["checkpoints"][1:] or plan["checkpoints"][0] != "native":
        raise ValueError("Checkpoint coverage differs from plan")
    return sources
