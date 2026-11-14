"""Diagnostic-only arm-sequence replay; never eligible as a scientific smoke gate."""

import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from sas_memory.controllers import GroupController
from sas_memory.runtime import TextRuntime
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_load_screen import weights_fingerprint, hook_state
import scripts.run_copy_evidence_necessity as runner

OUTPUT = Path("/media/990Pro/Code/SAS/artifacts/memory_update/copy-arm-replay-diagnostic-20261114-01.json")
RECORDS = []
ORIGINAL = runner.prove_and_measure


def controller_for(runtime):
    candidates = [h.__self__ for h in runtime.model._forward_pre_hooks.values()
                  if isinstance(getattr(h, "__self__", None), GroupController)]
    if len(candidates) != 1:
        raise ValueError("Expected one active diagnostic controller")
    return candidates[0]


def packet(runtime, controller, ledger, query, expected, stage, count=3):
    for i in range(count):
        actual = measure(runtime, ledger, query)
        row = {"stage": stage, "repeat": i, "actual": actual,
               "differences": {k: {"expected": expected[k], "actual": v} for k, v in actual.items() if expected[k] != v}}
        RECORDS.append(row)
        print(json.dumps(row), flush=True)


def trace(runtime, controller, ledger, query, stage):
    rows, handles = [], []
    def observe(name):
        def after(module, args, result):
            tensor = result[0] if isinstance(result, tuple) else result
            if not isinstance(tensor, torch.Tensor):
                return
            value = tensor.detach().contiguous().cpu()
            digest = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
            rows.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest,
                         "last_values": value.flatten()[-16:].float().tolist()})
        return after
    for name, module in runtime.model.named_modules():
        if (name.startswith("model.layers.") and name.count(".") == 2 or
            name in ("model.layers.15.self_attn.q_proj", "model.layers.15.self_attn.k_proj")):
            handles.append(module.register_forward_hook(observe(name)))
    for name, module in controller.named_modules():
        if name in ("encoders.attention", "outputs.attention_q_proj", "outputs.attention_k_proj"):
            handles.append(module.register_forward_hook(observe("adapter." + name)))
    try:
        actual = measure(runtime, ledger, query)
    finally:
        for h in handles:
            h.remove()
    RECORDS.append({"stage": stage, "actual": actual, "activation_trace": rows})
    print(json.dumps({"stage": stage, "actual": actual, "trace_entries": len(rows)}), flush=True)


def diagnose(runtime, ledger, query, masks, expected_row):
    try:
        return ORIGINAL(runtime, ledger, query, masks, expected_row)
    except ValueError as error:
        if "Actual unmasked reference readout changed" not in str(error):
            raise
        controller = controller_for(runtime)
        expected = readout(expected_row)
        RECORDS.append({"stage": "failure", "error": str(error), "query": query, "expected": expected,
            "controller": controller.description(), "adapter_state_sha256": state_fingerprint(controller),
            "model_weights": weights_fingerprint(runtime.model), "hooks": hook_state(runtime.model),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic})
        packet(runtime, controller, ledger, query, expected, "immediate")
        trace(runtime, controller, ledger, query, "warm_trace")
        torch.cuda.synchronize()
        packet(runtime, controller, ledger, query, expected, "synchronized")
        gc.collect()
        torch.cuda.empty_cache()
        packet(runtime, controller, ledger, query, expected, "empty_cache")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        packet(runtime, controller, ledger, query, expected, "deterministic")
        from scripts.copy_necessity_data import read_plan
        from scripts.copy_readout_data import read_plan as source_plan
        plan = read_plan()
        spec = json.loads(Path(plan["source_transfer_plan"]).read_text())["models"]["attention_only"]
        controller.close()
        fresh_controller = GroupController(runtime.model, spec["group_index"], spec["rank"], spec["mode"],
                                           adapt_gdn_keys=spec["adapt_gdn_keys"])
        fresh_controller.load_state_dict(load_file(spec["checkpoint"], device=str(runtime.model.device)), strict=True)
        fresh_controller.eval().requires_grad_(False)
        fresh_controller.set_token_policy(spec["token_policy"], query["ledger_tokens"])
        try:
            packet(runtime, fresh_controller, ledger, query, expected, "fresh_controller")
            trace(runtime, fresh_controller, ledger, query, "fresh_controller_trace")
        finally:
            fresh_controller.close()
        fresh = TextRuntime(plan["model_path"], "it")
        fresh_controller = GroupController(fresh.model, spec["group_index"], spec["rank"], spec["mode"],
                                           adapt_gdn_keys=spec["adapt_gdn_keys"])
        fresh_controller.load_state_dict(load_file(spec["checkpoint"], device=str(fresh.model.device)), strict=True)
        fresh_controller.eval().requires_grad_(False)
        fresh_controller.set_token_policy(spec["token_policy"], query["ledger_tokens"])
        try:
            packet(fresh, fresh_controller, ledger, query, expected, "fresh_model")
            trace(fresh, fresh_controller, ledger, query, "fresh_model_trace")
        finally:
            fresh_controller.close()
        raise


def main():
    if OUTPUT.exists():
        raise ValueError("Preserve diagnostic artifact")
    runner.prove_and_measure = diagnose
    runner.RUNTIME_SOURCES = tuple(sorted(set(runner.RUNTIME_SOURCES) | {str(Path(__file__).relative_to(ROOT))}))
    try:
        runner.main()
    finally:
        write_json(OUTPUT, {"status": "diagnostic_only_never_a_scientific_smoke_gate", "records": RECORDS,
                           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        OUTPUT.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
