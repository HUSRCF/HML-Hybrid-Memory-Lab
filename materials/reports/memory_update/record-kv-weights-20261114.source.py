"""Independently observe eager attention probabilities on blocked record columns."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import encode_counterbalance, file_digest
from sas_memory.counterbalance_tasks import ORDERS
from sas_memory.kv_readout import QuestionKVReadout
from scripts.run_controller_trial import write_json
from scripts.run_oracle_kv_readout import measure


class ObservedRecordWeights:
    def __init__(self, model, prelude_tokens, ledger_tokens):
        self.prelude, self.boundary = prelude_tokens, ledger_tokens
        self.current = None
        self.stats = {}
        self.handles = [model.register_forward_pre_hook(self.before, with_kwargs=True)]
        for index, layer in enumerate(model.model.layers):
            if model.config.layer_types[index] == "full_attention":
                self.stats[index] = {"prefill_calls": 0, "cached_calls": 0, "query_rows": 0,
                                     "max_record_probability": 0., "max_record_mass": 0.}
                self.handles.append(layer.self_attn.register_forward_hook(self.after(index)))
        if len(self.stats) != 6:
            raise ValueError("Expected six eager full-attention layers")

    def before(self, module, args, kwargs):
        if args:
            raise ValueError("Keyword input_ids required")
        cache = kwargs.get("past_key_values")
        self.current = (cache.get_seq_length() if cache is not None else 0, kwargs["input_ids"].shape[1])

    def after(self, index):
        def observe(module, args, result):
            import torch
            weights = result[1]
            past, length = self.current
            if not isinstance(weights, torch.Tensor) or weights.ndim != 4 or weights.shape[-2:] != (length, past + length):
                raise ValueError("Unexpected eager attention weight layout")
            start = max(0, self.boundary - past)
            if start >= length:
                return
            selected = weights[..., start:, self.prelude:self.boundary]
            if selected.numel() == 0:
                raise ValueError("No actual query-to-record attention entries inspected")
            stat = self.stats[index]
            stat["prefill_calls" if past == 0 else "cached_calls"] += 1
            stat["query_rows"] += length - start
            stat["max_record_probability"] = max(stat["max_record_probability"], selected.abs().max().item())
            stat["max_record_mass"] = max(stat["max_record_mass"], selected.float().sum(-1).abs().max().item())
        return observe

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["smoke"]:
        raise ValueError("A completed full ablation is required")
    plan = json.loads((args.run / "plan.json").read_text())
    cases = json.loads((args.run / "cases.json").read_text())
    registry = {(row["case_id"], row["variant"]): row for row in json.loads((args.run / "registry.json").read_text())}
    reference = Path(plan["reference_run"])
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    runtime = TextRuntime(manifest["model_path"], "it")
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(args.run),
              "manifest_sha256": file_digest(args.run / "manifest.json"), "samples": [],
              "source_sha256": file_digest(Path(__file__).resolve()),
              "interpretation": "Direct eager softmax-weight inspection on first/last cases, both late orders, native and seed1701. Ledger columns must have exactly zero probability/mass on every observed question/scoring/generation forward, while unmasked controls are nonzero. This checks the access intervention, not isolated recurrent-state content."}
    try:
        for label in ("native", "question_only_seed1701"):
            controller = None
            if label != "native":
                spec = manifest["checkpoint_sources"][label]
                protocol = spec["config"]
                checkpoint = Path(spec["training_run"]) / spec["mode"] / "controller.safetensors"
                if file_digest(checkpoint) != spec["checkpoint_sha256"]:
                    raise ValueError("Checkpoint changed")
                controller = GroupController(runtime.model, protocol["group_index"], protocol["modes"][spec["mode"]],
                                             spec["mode"], adapt_gdn_keys=protocol.get("adapt_gdn_keys", False))
                controller.load_state_dict(load_file(str(checkpoint), device=str(runtime.model.device)), strict=True)
            reference_rows = {(row["case_id"], row["variant"]): row for row in (json.loads(line) for line in (reference / f"{label}.raw_full.jsonl").read_text().splitlines())}
            ablated_rows = {(row["case_id"], row["variant"]): row for row in (json.loads(line) for line in (args.run / f"{label}.records_blocked.jsonl").read_text().splitlines())}
            try:
                for case in (cases[0], cases[-1]):
                    for order in ORDERS:
                        ledger, _, _ = encode_counterbalance(runtime, case, order)
                        row = registry[(case["id"], order)]
                        if controller is not None:
                            controller.set_token_policy("question", row["ledger_tokens"])
                        query = case["variants"][order]["queries"][0]
                        with ObservedRecordWeights(runtime.model, row["prelude_tokens"], row["ledger_tokens"]) as full:
                            actual = measure(runtime, ledger, query)
                        expected = reference_rows[(case["id"], order)]
                        if abs(actual["answer_nll"] - expected["answer_nll"]) > 1e-5 or actual["generated"] != expected["generated"]:
                            raise ValueError("Probability observation changed full-history behavior")
                        with QuestionKVReadout(runtime.model, [bit == "1" for bit in row["keep_bits"]]):
                            with ObservedRecordWeights(runtime.model, row["prelude_tokens"], row["ledger_tokens"]) as blocked:
                                actual = measure(runtime, ledger, query)
                        expected = ablated_rows[(case["id"], order)]
                        if abs(actual["answer_nll"] - expected["answer_nll"]) > 1e-5 or actual["generated"] != expected["generated"]:
                            raise ValueError("Probability observation changed blocked behavior")
                        for index in blocked.stats:
                            stat, control = blocked.stats[index], full.stats[index]
                            if stat["prefill_calls"] != 1 or stat["cached_calls"] < 1 or stat["max_record_probability"] != 0. or stat["max_record_mass"] != 0.:
                                raise ValueError("Nonzero blocked attention or missing observed forwards")
                            if control["max_record_probability"] <= 0. or control["max_record_mass"] <= 0.:
                                raise ValueError("Unmasked attention control is vacuous")
                        report["samples"].append({"model": label, "case_id": case["id"], "variant": order,
                                                  "full_history": full.stats, "records_blocked": blocked.stats,
                                                  "full_and_blocked_reference_replays_exact": True})
            finally:
                if controller is not None:
                    controller.close()
        report.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps({"status": report["status"], "samples": len(report["samples"]), "all_blocked_probabilities_and_masses": 0.0}))


if __name__ == "__main__":
    main()
