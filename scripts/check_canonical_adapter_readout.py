"""Actual-model arithmetic check before the newly registered scientific smoke."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from sas_memory.controllers import GroupController
from sas_memory.runtime import TextRuntime
from sas_memory.write_locality_data import streaming_hash
from scripts.copy_necessity_data import read_plan, registered_inputs, safety_samples
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_adapter_transfer import verify_question_prefix
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint
from scripts.stable_adapter_linears import StableAdapterLinears

ADDENDUM = ROOT / "configs/copy_evidence_necessity_precision_addendum.json"
ADDENDUM_SHA256 = "d7911f61eb63091d49e69f95159f1d28bff37284eb0dc007a987490249b433bc"
OUTPUT = Path("/media/990Pro/Code/SAS/artifacts/memory_update/canonical-adapter-readout-check-20261114-01.json")


def main():
    if OUTPUT.exists() or streaming_hash(ADDENDUM) != ADDENDUM_SHA256:
        raise ValueError("Preserve registered precision scope and new diagnostic output")
    torch.set_num_threads(4)
    registered, source_plan, cases, registry, masks, full_rows = registered_inputs()
    keys = safety_samples(registry)
    queries = [q for q in registry["physical"] if q["id"] in keys]
    case_map = {c["id"]: c for c in cases}
    refs = {m: {r["id"]: r for r in rows} for m, rows in full_rows.items()}
    names = ("attention_only", "local_question", "gdn_question_only")
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Arithmetic diagnostic only; not the required432-query operational smoke or scientific evidence.",
        "precision_addendum_sha256": ADDENDUM_SHA256, "profile": StableAdapterLinears.PROFILE,
        "training": False, "query_ids": [q["id"] for q in queries], "orders": [list(names), list(reversed(names))],
        "records": [], "prefix_checks": [], "native_replays": [], "adapter_checks": [],
        "source_sha256": {n: streaming_hash(ROOT / n) for n in (
            "scripts/check_canonical_adapter_readout.py", "scripts/stable_adapter_linears.py",
            "sas_memory/controllers.py", "sas_memory/runtime.py")}}
    try:
        runtime = TextRuntime(read_plan()["model_path"], "it")
        native_hooks = hook_state(runtime.model)
        report["weights_before"] = weights_fingerprint(runtime.model)
        first = {}
        for order, sequence in enumerate((names, tuple(reversed(names)))):
            for name in sequence:
                spec = source_plan["models"][name]
                controller = GroupController(runtime.model, spec["group_index"], spec["rank"], spec["mode"],
                                             adapt_gdn_keys=spec["adapt_gdn_keys"])
                arithmetic = None
                try:
                    controller.load_state_dict(load_file(spec["checkpoint"], device=str(runtime.model.device)), strict=True)
                    controller.eval().requires_grad_(False)
                    before = state_fingerprint(controller)
                    arithmetic = StableAdapterLinears(controller)
                    with torch.inference_mode():
                        for q in queries:
                            ledger = actual_prompt(runtime, case_map, q)
                            if spec["token_policy"] == "question":
                                check = verify_question_prefix(runtime, controller, ledger, q)
                                report["prefix_checks"].append({"order": order, "model": name, **check})
                            controller.set_token_policy(spec["token_policy"], q["ledger_tokens"])
                            for repeat in range(3):
                                actual = measure(runtime, ledger, q)
                                key = (name, q["id"])
                                if key in first and actual != first[key]:
                                    raise ValueError(f"Canonical repeated/cross-order readout changed: {name},{q['id']},{actual},{first[key]}")
                                first.setdefault(key, actual)
                                report["records"].append({"order": order, "repeat": repeat, "model": name, "id": q["id"],
                                    "actual": actual, "legacy_fp32": readout(refs[name][q["id"]])})
                    after = state_fingerprint(controller)
                    if before != after or any(v < 1 for v in arithmetic.calls.values()):
                        raise ValueError("Canonical controller weights changed or an affine hook was not exercised")
                    report["adapter_checks"].append({"order": order, "model": name, "before_sha256": before,
                        "after_sha256": after, "arithmetic": arithmetic.description(), "actual_calls": arithmetic.calls})
                finally:
                    if arithmetic is not None:
                        arithmetic.close()
                    controller.close()
                if hook_state(runtime.model) != native_hooks:
                    raise ValueError("Canonical adapter cleanup failed to restore native model hooks")
                q = queries[1]
                with torch.inference_mode():
                    actual = measure(runtime, actual_prompt(runtime, case_map, q), q)
                if actual != readout(refs["native"][q["id"]]):
                    raise ValueError("Native readout changed after canonical-hook removal")
                report["native_replays"].append({"order": order, "after_model": name, "id": q["id"], "exact": True})
                print(json.dumps({"event": "canonical_arithmetic_model_checked", "order": order, "model": name,
                    "queries": len(queries), "three_repeats_exact": True}), flush=True)
        report["weights_after"] = weights_fingerprint(runtime.model)
        if report["weights_after"] != report["weights_before"]:
            raise ValueError("Canonical-hook diagnostic changed native checkpoint tensors or gradients")
        prefixes = {}
        for p in report["prefix_checks"]:
            key = (p["model"], p["source_key"])
            if key in prefixes and prefixes[key] != p["prefix_sha256"]:
                raise ValueError("Question-only prefix hash differs across arm orders")
            prefixes[key] = p["prefix_sha256"]
        if len(report["records"]) != 108 or len(report["prefix_checks"]) != 24:
            raise ValueError("Incomplete actual-model canonical arithmetic check")
        report.update(status="completed_arithmetic_diagnostic_only", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(OUTPUT, report)
        OUTPUT.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": report["status"], "exact_readouts": len(report["records"]),
        "exact_native_prefix_checks": len(report["prefix_checks"]), "native_replays": len(report["native_replays"])}), flush=True)


if __name__ == "__main__":
    main()
