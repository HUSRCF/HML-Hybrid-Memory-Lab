"""Capture a small FP32 projection fixture; no full-model weight copy."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file, save_file
from sas_memory.controllers import GroupController
from sas_memory.runtime import TextRuntime
from scripts.copy_necessity_data import read_plan
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_oracle_kv_readout import measure


def main():
    output = Path("/media/990Pro/Code/SAS/artifacts/memory_update/attention-projection-fixture-20261114-01")
    if output.with_suffix(".json").exists() or output.with_suffix(".safetensors").exists():
        raise ValueError("Preserve captured numeric fixture")
    torch.set_num_threads(4)
    plan = read_plan()
    source_plan = json.loads(Path(plan["source_transfer_plan"]).read_text())
    source = Path(plan["source_dataset"])
    registry = json.loads((source / "registry.json").read_text())
    cases = {c["id"]: c for c in json.loads((source / "cases.json").read_text())}
    query = next(q for q in registry["physical"] if q["id"] == "571aa66bb133a769f6f54a82e54ffe78da046507b55347a5946b46be66d593bf")
    expected = next(json.loads(line) for line in (Path(plan["source_run"]) / "attention_only.jsonl").read_text().splitlines()
                    if json.loads(line)["id"] == query["id"])
    runtime = TextRuntime(plan["model_path"], "it")
    spec = source_plan["models"]["attention_only"]
    controller = GroupController(runtime.model, spec["group_index"], spec["rank"], spec["mode"], adapt_gdn_keys=spec["adapt_gdn_keys"])
    controller.load_state_dict(load_file(spec["checkpoint"], device=str(runtime.model.device)), strict=True)
    controller.eval().requires_grad_(False)
    controller.set_token_policy(spec["token_policy"], query["ledger_tokens"])
    tensors, handles = {}, []
    def capture(name):
        def after(module, args, result):
            if name not in tensors:
                tensors[name + "_input"] = args[0].detach().cpu().contiguous()
                tensors[name] = result.detach().cpu().contiguous()
        return after
    for kind in ("q", "k"):
        handles.append(controller.outputs[f"attention_{kind}_proj"].register_forward_hook(capture(kind)))
    try:
        with torch.inference_mode():
            result = measure(runtime, actual_prompt(runtime, cases, query), query)
        if result != readout(expected):
            raise ValueError(f"Cold fixture readout differs: {result}")
        if tensors["q"].shape != (1, 421, 2048) or not torch.equal(tensors["q_input"], tensors["k_input"]):
            raise ValueError("Unexpected whole-prefill projection fixture")
        save_file(tensors, output.with_suffix(".safetensors"))
        write_json(output.with_suffix(".json"), {"status": "cold_exact_reference_numeric_fixture", "query_id": query["id"],
            "result": result, "checkpoint": spec["checkpoint"], "checkpoint_sha256": spec["checkpoint_sha256"],
            "adapter_state_sha256": state_fingerprint(controller), "blas_backend": str(torch.backends.cuda.preferred_blas_library()),
            "fixture_sha256": hashlib.sha256(output.with_suffix(".safetensors").read_bytes()).hexdigest(),
            "tensors": {k: {"shape": list(v.shape), "sha256": hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()} for k, v in tensors.items()},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
        print(json.dumps({"status": "captured", "nll": result["answer_nll"], "fixture_bytes": output.with_suffix(".safetensors").stat().st_size}), flush=True)
    finally:
        for h in handles:
            h.remove()
        controller.close()


if __name__ == "__main__":
    main()
