"""Minimal-ledger and direct-arithmetic competence controls for native ADD errors."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.oracle import evidence_slice
from sas_memory.tasks import INSTRUCTIONS, Operation, execute, render
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_paired_semantic_probe import audit_run as audit_native, identity, registered_dataset
from scripts.run_controller_trial import write_json
from scripts.run_oracle_kv_readout import measure


def controls_for(runtime, example):
    case, query = example["case"], example["query"]
    events = [Operation(**event) for event in case["prefix"] + case["variants"][example["variant"]]["tail"]]
    selected = evidence_slice(events, query["entity"], "current")
    if (not selected or selected[0].kind != "SET" or any(event.kind != "ADD" for event in selected[1:]) or
        len(selected) != (2 if example["variant"] == "control" else 3) or
        str(execute(selected)["current"][query["entity"]]) != query["answer"]):
        raise ValueError("Minimal ADD program does not preserve original truth")
    expression = " + ".join(str(event.value) for event in selected)
    if sum(event.value for event in selected) != int(query["answer"]):
        raise ValueError("Direct arithmetic differs from executable truth")
    options = {"minimal_dependency_ledger": (runtime.prompt_start + INSTRUCTIONS + render([asdict(event) for event in selected]), query["question"]),
               "direct_arithmetic": (runtime.prompt_start + "Compute the integer expression. Answer using only one integer, with no explanation.\n\n", f"What is {expression}?")}
    return {condition: {"ledger": ledger, "question": question, "answer": query["answer"],
                        "prompt_ids_sha256": data_fingerprint(runtime.ids(ledger + runtime.query_suffix(question))),
                        "ledger_tokens": len(runtime.ids(ledger)), "events": [asdict(event) for event in selected]}
            for condition, (ledger, question) in options.items()}


def prepare(plan):
    registered, data, _ = registered_dataset(Path(plan["data_source"]))
    native = audit_native(Path(plan["native_reference_run"]))
    selection = plan["selection"]
    members = [example for example in data["validation"] if example["case"]["family"] == selection["family"] and
               example["query"]["role"] == selection["role"] and example["variant"] in selection["variants"]]
    if len(members) != plan["examples_per_condition"]:
        raise ValueError("ADD competence subset changed")
    runtime = tokenizer_runtime(registered["model_path"])
    controls = [{"key": identity(example), "conditions": controls_for(runtime, example)} for example in members]
    return registered, members, controls, native


def audit_run(path):
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    registered, members, controls, native = prepare(plan)
    if (manifest["status"] != "completed" or manifest["training"] or not manifest["backbone_frozen_no_gradients"] or
        manifest["plan_sha256"] != data_fingerprint(plan) or manifest["native_reference"] != native or
        controls != json.loads((path / "controls.json").read_text()) or manifest["controls_sha256"] != data_fingerprint(controls) or
        datetime.fromisoformat(plan["registered_utc"]) >= datetime.fromisoformat(manifest["started_utc"])):
        raise ValueError("ADD competence registration/completion changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest:
            raise ValueError("Archived ADD control source changed")
    expected = {tuple(identity(example)): example["query"]["answer"] for example in members}
    native_path = Path(plan["native_reference_run"]) / "validation.native.jsonl"
    source = {tuple([row["case_id"], row["variant"], row["role"]]): row for row in
              (json.loads(line) for line in native_path.read_text().splitlines())}
    metrics = {"full_ledger_reference": {"correct": sum(source[key]["greedy_exact"] for key in expected), "examples": len(expected)}}
    for condition in plan["conditions"]:
        rows = load_rows(path / f"{condition}.jsonl", expected)
        for row, control in zip(rows, controls):
            if ([row["case_id"], row["variant"], row["role"]] != control["key"] or row["condition"] != condition or row["registration"] != control["conditions"][condition]):
                raise ValueError("ADD competence input identity changed")
        metrics[condition] = {"correct": sum(row["greedy_exact"] for row in rows), "examples": len(rows),
                              "answer_nll": sum(row["answer_nll"] for row in rows) / len(rows)}
    return {"status": "completed_and_audited", "source_run": str(path), "manifest_sha256": file_hash(path / "manifest.json"),
            "metrics": metrics, "records_audited": 2 * len(expected), "interpretation": plan["interpretation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/semantic_add_competence_probe.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["training"] or plan["model"] != "it" or plan["conditions"] != ["minimal_dependency_ledger", "direct_arithmetic"]:
        raise ValueError("Unregistered ADD competence protocol")
    registered, members, controls, native = prepare(plan)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "controls.json", controls)
    sources = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(),
        ROOT / "scripts/analyze_paired_semantic_probe.py", ROOT / "scripts/prepare_paired_semantic_updates.py",
        ROOT / "scripts/run_controller_trial.py", ROOT / "scripts/run_oracle_kv_readout.py",
        ROOT / "scripts/analyze_controller_trial.py", ROOT / "scripts/analyze_blocked_reader_adaptation.py"]
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "training": False,
                "plan_sha256": data_fingerprint(plan), "controls_sha256": data_fingerprint(controls), "native_reference": native,
                "source_sha256": {str(source.relative_to(ROOT)): file_hash(source) for source in sources}}
    for relative in manifest["source_sha256"]:
        destination = args.output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    write_json(args.output / "manifest.json", manifest)
    import torch
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    try:
        runtime = TextRuntime(registered["model_path"], "it")
        for condition in plan["conditions"]:
            with (args.output / f"{condition}.jsonl").open("w", buffering=1) as output:
                for example, control in zip(members, controls):
                    actual = controls_for(runtime, example)[condition]
                    if actual != control["conditions"][condition]:
                        raise ValueError("Actual ADD competence prompt changed")
                    result = measure(runtime, actual["ledger"], actual)
                    output.write(json.dumps({"case_id": example["case"]["id"], "variant": example["variant"],
                        "role": example["query"]["role"], "condition": condition, "registration": actual, **result}) + "\n")
        manifest["backbone_frozen_no_gradients"] = all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.model.parameters())
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
