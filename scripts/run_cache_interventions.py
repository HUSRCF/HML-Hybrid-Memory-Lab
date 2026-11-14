"""Locate pathway effects with matched-length, explicitly counterfactual cache transplants."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", choices=("it", "base"), default="it")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", default="overwrite")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--groups", default="all,0,1,2,3,4,5")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    cases = [case for case in cases if case["family"] == args.family][:args.limit]
    if not cases:
        raise ValueError("No matching cases")
    args.output.mkdir(parents=True, exist_ok=False)
    from sas_memory.tasks import INSTRUCTIONS, render
    from sas_memory.runtime import TextRuntime, assert_cache_equal, fork_cache
    from sas_memory.interventions import transplant
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    path = json.loads((ROOT / "configs/models.json").read_text())[args.model]
    manifest = {"status": "running", "model": args.model, "model_path": path,
                "started_utc": datetime.now(timezone.utc).isoformat(), "case_ids": [c["id"] for c in cases],
                "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve()]},
                "interpretation": "Cache transplants are unnatural counterfactuals, not task accuracy or a trained method."}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    try:
        runtime = TextRuntime(path, args.model)
        layer_types = list(runtime.model.config.layer_types)
        group_sets = [(label, list(range(len(layer_types) // 4)) if label == "all" else [int(label)])
                      for label in args.groups.split(",")]
        records = []
        verification = {"self_transplant_equal": True, "all_path_matches_donor_max_nll_error": 0.0}
        with (args.output / "results.jsonl").open("w", buffering=1) as output:
            for case_index, case in enumerate(cases):
                ledgers, caches = {}, {}
                for variant in ("control", "treatment"):
                    ledgers[variant] = runtime.prompt_start + INSTRUCTIONS + render(case["prefix"] + case["variants"][variant]["tail"])
                    caches[variant], _, _ = runtime.advance(ledgers[variant])
                if len(runtime.ids(ledgers["control"])) != len(runtime.ids(ledgers["treatment"])):
                    raise ValueError(f"Position mismatch: {case['id']}")
                saved = {key: fork_cache(cache) for key, cache in caches.items()}
                native = {}

                def measure(recipient, donor, cache, condition, group=None, path=None, prefill="cached"):
                    measured = []
                    for query_index, query in enumerate(case["variants"][recipient]["queries"]):
                        donor_answer = case["variants"][donor]["queries"][query_index]["answer"]
                        result = runtime.probe(ledgers[recipient], cache, query["question"], query["answer"],
                                               case["candidates"], generate=False, prefill=prefill)
                        scores = result["candidate_scores"]
                        row = {"case_id": case["id"], "distractor_updates": case["distractor_updates"],
                               "recipient": recipient, "donor": donor, "condition": condition,
                               "group": group, "path": path, "role": query["role"],
                               "recipient_answer": query["answer"], "donor_answer": donor_answer,
                               "recipient_answer_nll": scores[query["answer"]]["nll_mean"],
                               "donor_answer_nll": scores[donor_answer]["nll_mean"],
                               "donor_vs_recipient_log_odds": scores[query["answer"]]["nll_sum"] - scores[donor_answer]["nll_sum"],
                               "candidate_prediction": result["candidate_prediction"], "candidate_scores": scores}
                        records.append(row)
                        measured.append(row)
                        output.write(json.dumps(row) + "\n")
                    return measured

                for variant in caches:
                    native[variant] = measure(variant, variant, caches[variant], "natural_cached")
                    measure(variant, variant, caches[variant], "natural_full", prefill="full")
                    self_cache = transplant(caches[variant], caches[variant], layer_types,
                                            list(range(len(layer_types) // 4)), "both")
                    assert_cache_equal(self_cache, caches[variant])
                for recipient, donor in (("control", "treatment"), ("treatment", "control")):
                    for label, groups in group_sets:
                        for path_kind in ("recurrent", "kv", "both"):
                            mixed = transplant(caches[recipient], caches[donor], layer_types, groups, path_kind)
                            measured = measure(recipient, donor, mixed, "transplant", label, path_kind)
                            if label == "all" and path_kind == "both":
                                for row, donor_row in zip(measured, native[donor]):
                                    error = max(abs(row["candidate_scores"][c]["nll_mean"] - donor_row["candidate_scores"][c]["nll_mean"])
                                                for c in case["candidates"])
                                    verification["all_path_matches_donor_max_nll_error"] = max(error, verification["all_path_matches_donor_max_nll_error"])
                                    if error > 1e-5:
                                        raise AssertionError(f"All-path transplant did not reproduce donor: {error}")
                for variant in caches:
                    assert_cache_equal(caches[variant], saved[variant])
                print(json.dumps({"event": "case_done", "case": case["id"], "index": case_index + 1,
                                  "total": len(cases), "rows": len(records),
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
                (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        manifest.update(status="completed", rows=len(records), elapsed_seconds=time.monotonic() - started,
                        completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
