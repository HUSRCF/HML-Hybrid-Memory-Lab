"""Compare short matched programs with a source versus unrelated-account update."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.generalization_tasks import copy_update_challenges, validate_generalization_cases
from sas_memory.training_data import data_fingerprint
from sas_memory.tasks import INSTRUCTIONS, render
from scripts.run_controller_confirmation import verify_native_prefix
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/copy_update_controls.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    reference = Path(config["reference_run"])
    original = json.loads((reference / "manifest.json").read_text())
    if original["status"] != "completed" or original["completed_oracle_modes"] != config["modes"]:
        raise ValueError("Reference evidence evaluation is incomplete")
    cases = json.loads((reference / "cases.json").read_text())
    validate_generalization_cases(cases)
    if data_fingerprint(cases) != config["expected_cases_sha256"]:
        raise ValueError("Control cohort changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "cases.json", cases)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "cases_sha256": data_fingerprint(cases), "reference_manifest_sha256": hashlib.sha256((reference / "manifest.json").read_bytes()).hexdigest(),
                "reference_oracle_sha256": {}, "checkpoint_sha256": {}, "completed_modes": [],
                "minimal_replay": {}, "pre_question_cache_checks": {}, "matched_prompt_lengths": {},
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((ROOT / "sas_memory").glob("*.py"))
                                  + [ROOT / "scripts/run_controller_confirmation.py", ROOT / "scripts/run_controller_trial.py", Path(__file__).resolve()]},
                "training": False, "interpretation": "Exploratory semantic-update controls on every control-order case. Filtering changes context; source/unrelated programs share their prefix and differ only in the final SET target."}
    for relative in manifest["source_sha256"]:
        snapshot = args.output / "sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / relative).read_bytes())
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    protocol = original["training_config"]
    model_path = json.loads((ROOT / "configs/models.json").read_text())[protocol["model"]]
    try:
        runtime = TextRuntime(model_path, protocol["model"])
        manifest.update(model_path=model_path, python=sys.executable, gpu=torch.cuda.get_device_name(0),
                        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")})
        for label in config["modes"]:
            controller, policy = None, "all"
            if label != "native":
                spec = original["checkpoint_sources"][label]
                protocol = spec["config"]
                checkpoint = Path(spec["training_run"]) / spec["mode"] / "controller.safetensors"
                digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                if digest != original["checkpoint_sha256"][label]:
                    raise ValueError("Reference controller changed")
                manifest["checkpoint_sha256"][label] = digest
                controller = GroupController(runtime.model, protocol["group_index"], protocol["modes"][spec["mode"]],
                                             spec["mode"], adapt_gdn_keys=protocol.get("adapt_gdn_keys", False))
                controller.load_state_dict(load_file(str(checkpoint), device=str(runtime.model.device)), strict=True)
                policy = protocol.get("token_policy", "all")
            reference_path = reference / f"{label}.oracle.jsonl"
            manifest["reference_oracle_sha256"][label] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            reference_rows = {(row["case_id"], row["variant"]): row for row in
                              (json.loads(line) for line in reference_path.read_text().splitlines())}
            matched = 0
            try:
                with (args.output / f"{label}.jsonl").open("w", buffering=1) as output:
                    for index, case in enumerate(cases):
                        query = next(q for q in case["variants"]["control"]["queries"] if q["role"] == "target_current")
                        programs = copy_update_challenges(case)
                        ledgers = {name: runtime.prompt_start + INSTRUCTIONS + render([asdict(event) for event in events])
                                   for name, events in programs.items()}
                        lengths = {name: len(runtime.ids(ledger + runtime.query_suffix(query["question"]))) for name, ledger in ledgers.items()}
                        if lengths["source_update"] != lengths["unrelated_update"]:
                            raise AssertionError("Matched SET-target programs have different token lengths")
                        matched += 1

                        def measure(name):
                            ledger = ledgers[name]
                            if controller is not None:
                                controller.set_token_policy(policy, len(runtime.ids(ledger)))
                            result = runtime.probe(ledger, None, query["question"], query["answer"], [query["answer"]],
                                                   generate=True, prefill="full", score_batch=False)
                            for field in ("candidate_prediction", "candidate_correct", "candidate_scores"):
                                result.pop(field)
                            return result

                        if index == 0:
                            actual, expected = measure("minimal"), reference_rows[(case["id"], "control")]
                            error = abs(actual["answer_nll"] - expected["answer_nll"])
                            if error > 1e-5 or actual["generated"] != expected["generated"]:
                                raise AssertionError("Minimal-evidence checkpoint replay changed")
                            manifest["minimal_replay"][label] = {"case_id": case["id"], "max_nll_error": error, "greedy_exact_match": True}
                        for name in config["regimes"]:
                            if index == 0 and controller is not None and policy == "question":
                                manifest["pre_question_cache_checks"].setdefault(label, {})[name] = {
                                    "case_id": case["id"], **verify_native_prefix(runtime, controller, ledgers[name])}
                            row = {"case_id": case["id"], "source_variant": "control", "variant": name, "role": "target_current",
                                   "events": [asdict(event) for event in programs[name]], "prompt_tokens": lengths[name], **measure(name)}
                            output.write(json.dumps(row) + "\n")
                manifest["matched_prompt_lengths"][label] = matched
                manifest["completed_modes"].append(label)
                write_json(manifest_path, manifest)
                print(json.dumps({"event": "update_controls_complete", "mode": label, "rows": len(cases) * len(config["regimes"])}), flush=True)
            finally:
                if controller is not None:
                    controller.close()
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
