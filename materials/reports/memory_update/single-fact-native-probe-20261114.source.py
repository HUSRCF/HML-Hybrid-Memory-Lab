"""Audit same-question single-fact controls and complete counterfactual families."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import file_digest
from sas_memory.record_kv_ablation import record_blocked_mask
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.analyze_controller_trial import load_rows
from scripts.prepare_single_fact_control import validate_data
from scripts.run_controller_trial import encode_example, prompt_for, write_json


def registered_dataset(path):
    manifest = json.loads((path / "manifest.json").read_text())
    data = json.loads((path / "data.json").read_text())
    registry = json.loads((path / "registry.json").read_text())
    encoded = json.loads((path / "tokenized_train.json").read_text())
    if manifest["status"] != "registered_before_inference" or manifest["training"] or manifest["model_inference"]:
        raise ValueError("Expected a pre-inference dataset registration")
    if validate_data(data, 32, 16) != manifest["counts"]:
        raise ValueError("Executable truth/family coverage changed")
    if data_fingerprint(data) != manifest["data_sha256"] or data_fingerprint(registry) != manifest["registry_sha256"]:
        raise ValueError("Registered examples or masks changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_digest(path / "sources" / relative) != digest:
            raise ValueError("Dataset source snapshot changed")
    for relative in ("sas_memory/runtime.py", "sas_memory/tasks.py", "sas_memory/kv_readout.py", "sas_memory/record_kv_ablation.py", "scripts/run_controller_trial.py"):
        if file_digest(ROOT / relative) != manifest["source_sha256"][relative]:
            raise ValueError("Actual input/mask runtime differs from registration")
    for name, digest in manifest["tokenizer_sha256"].items():
        if file_digest(Path(manifest["model_path"]) / name) != digest:
            raise ValueError("Tokenizer or model configuration changed")
    runtime = tokenizer_runtime(manifest["model_path"])
    for split in ("train", "validation"):
        if len(registry[split]) != len(data[split]):
            raise ValueError("Registered token rows missing")
        families = defaultdict(list)
        for example, row in zip(data[split], registry[split]):
            case, query = example["case"], example["query"]
            ledger, prompt = prompt_for(runtime, case, example["variant"], query)
            mask = record_blocked_mask(runtime, ledger, prompt)
            if any(row[key] != value for key, value in mask.items()):
                raise ValueError("Registered prompt/boundary mask changed")
            if (row["case_id"] != case["id"] or row["base_case_id"] != case["base_case_id"] or row["value_version"] != case["value_version"]):
                raise ValueError("Registered family identity changed")
            tokens = runtime.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            positions = row["value_token_positions"]
            value = query["answer"]
            if ledger.count(value) != 1:
                raise ValueError("Fact value occurrence not unique")
            start = ledger.index(value)
            if (len(positions) != 3 or positions != list(range(positions[0], positions[0] + 3)) or
                any(not mask["prelude_tokens"] <= index < mask["ledger_tokens"] for index in positions) or
                [tuple(tokens["offset_mapping"][index]) for index in positions] != [(start + i, start + i + 1) for i in range(3)]):
                raise ValueError("Registered differing tokens are not the three fact-value digits")
            normalized = tuple(-1 if index in positions else token for index, token in enumerate(tokens["input_ids"]))
            question = tuple(tokens["input_ids"][mask["ledger_tokens"]:])
            if data_fingerprint(list(normalized)) != row["normalized_prompt_sha256"] or data_fingerprint(list(question)) != row["question_ids_sha256"]:
                raise ValueError("Normalized prompt/question fingerprints changed")
            families[case["base_case_id"]].append((normalized, question, mask["ledger_tokens"], mask["prompt_tokens"], tuple(positions), mask["keep_sha256"]))
        if any(len(siblings) != 4 or len(set(siblings)) != 1 for siblings in families.values()):
            raise ValueError("Counterfactual siblings differ outside the fact value")
    reproduced = [encode_example(runtime, example, "records_blocked") for example in data["train"]]
    if reproduced != encoded or data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded]) != manifest["tokenized_input_answer_sha256"]:
        raise ValueError("Registered training tokens/labels changed")
    return manifest, data, registry


def family_metrics(rows):
    if not rows:
        raise ValueError("Empty counterfactual cohort")
    groups = defaultdict(list)
    for row in rows:
        groups[row["base_case_id"]].append(row)
    histogram = Counter()
    identical = 0
    for siblings in groups.values():
        if (len(siblings) != 4 or {row["value_version"] for row in siblings} != {0, 1, 2, 3} or
            len({row["answer"] for row in siblings}) != 4):
            raise ValueError("Incomplete or non-counterfactual family")
        histogram[sum(row["greedy_exact"] for row in siblings)] += 1
        identical += len({row["generated"].strip() for row in siblings}) == 1
    return {"examples": len(rows), "correct": sum(row["greedy_exact"] for row in rows),
            "accuracy": mean(row["greedy_exact"] for row in rows), "answer_nll": mean(row["answer_nll"] for row in rows),
            "families": len(groups), "all_four_correct": histogram[4],
            "more_than_one_correct": sum(count for correct, count in histogram.items() if correct > 1),
            "correct_per_family_histogram": {str(key): histogram[key] for key in range(5)},
            "identical_predictions_in_family": identical,
            "prediction_counts": dict(Counter(row["generated"].strip() for row in rows))}


def audit_run(path, allow_smoke=False):
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    dataset = Path(plan["data_source"])
    registered, data, registry = registered_dataset(dataset)
    if (manifest["status"] != "completed" or manifest["training"] or manifest["completed_splits"] != plan["splits"] or
        manifest["plan_sha256"] != data_fingerprint(plan) or manifest["data_manifest_sha256"] != file_digest(dataset / "manifest.json")):
        raise ValueError("Incomplete probe or changed registration")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Smoke is operational evidence, not full-cohort evidence")
    if (registered["data_sha256"] != plan["data_sha256"] or registered["registry_sha256"] != plan["registry_sha256"] or
        manifest["model_path"] != registered["model_path"]):
        raise ValueError("Model/examples differ from registered plan")
    for stamp in (registered["registered_utc"], plan["registered_utc"]):
        if datetime.fromisoformat(stamp) >= datetime.fromisoformat(manifest["started_utc"]):
            raise ValueError("Registration was not fixed before inference")
    for relative, digest in manifest["source_sha256"].items():
        if file_digest(path / "sources" / relative) != digest:
            raise ValueError("Probe source snapshot changed")
    for relative in ("sas_memory/runtime.py", "sas_memory/kv_readout.py", "sas_memory/record_kv_ablation.py"):
        if file_digest(ROOT / relative) != manifest["source_sha256"][relative]:
            raise ValueError("Actual runtime differs from the probe")
    if not manifest["frozen_parameters_and_no_gradients"] or not manifest["hooks_restored"]:
        raise ValueError("Model freeze or hook restoration failed")
    models, total = {}, 0
    for split in plan["splits"]:
        members = data[split]
        if manifest["smoke"]:
            first_family = min(example["case"]["base_case_id"] for example in members)
            members = [example for example in members if example["case"]["base_case_id"] == first_family]
        if [example["case"]["id"] for example in members] != manifest["evaluated_case_ids"][split]:
            raise ValueError("Evaluated cases were changed")
        lookup = {example["case"]["id"]: example for example in members}
        expected = {(case_id, example["variant"], example["query"]["role"]): example["query"]["answer"] for case_id, example in lookup.items()}
        registered_rows = {row["case_id"]: row for row in registry[split]}
        model = {}
        for condition in plan["conditions"]:
            rows = load_rows(path / f"{split}.{condition}.jsonl", expected)
            for row in rows:
                example = lookup[row["case_id"]]
                if (row["condition"] != condition or row["registration"] != registered_rows[row["case_id"]] or
                    any(row[key] != example["case"][key] for key in ("base_case_id", "value_version", "family", "seed"))):
                    raise ValueError("Result condition/input/family metadata changed")
                if condition == "records_blocked" and (row["readout_calls"]["prefill"] != 1 or row["readout_calls"]["cached"] < 1):
                    raise ValueError("Cached answers were not masked")
                if condition == "full_history" and row["readout_calls"] is not None:
                    raise ValueError("Native baseline unexpectedly used an intervention")
            model[condition] = family_metrics(rows)
            total += len(rows)
        sample_ids = [min(lookup), max(lookup)]
        if [check["case_id"] for check in manifest["safety_checks"][split]] != sample_ids:
            raise ValueError("Missing prescribed sample checks")
        for check in manifest["safety_checks"][split]:
            for key in ("all_history_full_cache_and_logits_exact", "all_history_scoring_and_greedy_exact", "pre_question_kv_recurrent_conv_and_logits_exact", "historical_kv_exact_in_full_prefill", "native_replay_after_mask_exact"):
                if check[key] is not True:
                    raise ValueError("Native/mask/prefix identity failed")
            if check["historical_kv_fields_checked"] != 12 or check["unrestricted_replay_max_nll_error"] != 0:
                raise ValueError("Missing KV fields or native replay mismatch")
            if set(check["blocked_weights"]) != {str(i) for i in (3, 7, 11, 15, 19, 23)}:
                raise ValueError("Missing full-attention layers")
            for layer, stat in check["blocked_weights"].items():
                if (stat["prefill_calls"] != 1 or stat["cached_calls"] < 1 or stat["max_record_probability"] != 0 or
                    stat["max_record_mass"] != 0 or check["full_weights"][layer]["max_record_mass"] <= 0):
                    raise ValueError("Blocked actual probability or nonvacuous control failed")
        models[split] = model
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
            "manifest_sha256": file_digest(path / "manifest.json"), "records_audited": total, "splits": models,
            "interpretation": plan["interpretation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report = audit_run(args.run, args.allow_smoke)
    report["analysis_source_sha256"] = file_digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({split: {condition: {key: metrics[key] for key in ("correct", "examples", "all_four_correct", "families", "more_than_one_correct")} for condition, metrics in conditions.items()} for split, conditions in report["splits"].items()}))


if __name__ == "__main__":
    main()
