"""Require exact252-readout fresh-cohort smoke and complete-prefix order agreement."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.fresh_copy_data import MASKED_REGIMES, MODELS, model_order, read_plan
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import write_json


def compare_observations(forward, reverse, first_rows, second_rows):
    """No tolerance, accuracy filtering, omitted rows or cross-profile references."""
    if (forward["model_order"] != model_order() or reverse["model_order"] != model_order(True) or
        forward["selected_physical_ids"] != reverse["selected_physical_ids"] or
        len(forward["selected_physical_ids"]) != 18 or forward["native_anchors"] != reverse["native_anchors"]):
        raise ValueError("Opposite complete model orders, query coverage and native anchors required")
    count = prefix_count = question_count = 0
    for name in MODELS:
        for regime in read_plan()["regimes_by_model"][name]:
            a, b = first_rows[name][regime], second_rows[name][regime]
            if [r["id"] for r in a] != forward["selected_physical_ids"] or [r["id"] for r in b] != forward["selected_physical_ids"]:
                raise ValueError("Cross-order readout coverage/order incomplete")
            for x, y in zip(a, b):
                if readout(x) != readout(y):
                    raise ValueError(f"Canonical cross-order readout differs: {name},{regime},{x['id']}")
                count += 1
        a, b = forward["models"][name], reverse["models"][name]
        if [p["id"] for p in a["proofs"]] != [p["id"] for p in b["proofs"]]:
            raise ValueError("Cross-order sampled complete-prefix coverage changed")
        for x, y in zip(a["proofs"], b["proofs"]):
            if any(x["regimes"][r][field] != y["regimes"][r][field]
                   for r in (MASKED_REGIMES if name in ("local_question", "gdn_question_only") else ()) for field in ("prefix_sha256", "masked_prefix_sha256")):
                raise ValueError(f"Canonical cross-order complete-prefix hash differs: {name},{x['id']}")
            prefix_count += int(name in ("local_question", "gdn_question_only"))
        if a["prefix_checks"] != b["prefix_checks"]:
            raise ValueError("Canonical question-only native-prefix hash differs across orders")
        question_count += len(a["prefix_checks"])
    if (count, prefix_count, question_count) != (252, 20, 12):
        raise ValueError("All252readouts,20complete-prefix samples and12question-only checks required")
    return {"exact_readouts": count, "exact_complete_prefix_samples": prefix_count,
            "exact_masked_prefix_hash_pairs": 3 * prefix_count, "exact_question_only_prefix_checks": question_count,
            "exact_precontroller_native_anchors": len(forward["native_anchors"])}


def audit_equivalence(forward_path, reverse_path):
    from scripts.analyze_fresh_copy_confirmation import audit_run
    reports, manifests, rows = [], [], []
    for path in (Path(forward_path), Path(reverse_path)):
        report = json.loads(path.read_text())
        if (report["status"] != "audited_fresh_copy_smoke" or report["new_physical_readouts_audited"] != 252 or
            streaming_hash(path.with_suffix(".source.py")) != report["analysis_source_sha256"] or
            audit_run(Path(report["source_run"]), True) != report):
            raise ValueError("Complete independent252-readout smoke audits required")
        run = Path(report["source_run"])
        manifest = json.loads((run / "manifest.json").read_text())
        reports.append(report)
        manifests.append(manifest)
        rows.append({m: {r: [json.loads(line) for line in (run / f"{m}.{r}.jsonl").read_text().splitlines()]
                         for r in read_plan()["regimes_by_model"][m]} for m in MODELS})
    first, second = manifests
    if (any(first[field] != second[field] for field in (
        "source_sha256", "precision_addendum_sha256", "arithmetic_profile", "dataset_manifest_sha256",
        "mask_registry_sha256", "references", "generation", "generation_config_sha256", "weights_before", "independent_novelty_recheck", "prior_inputs_sha256")) or
        first["reverse_model_order"] or not second["reverse_model_order"]):
        raise ValueError("Canonical smokes changed sources, checkpoint, data, masks or arithmetic")
    exact = compare_observations(first, second, rows[0], rows[1])
    return {"status": "audited_fresh_copy_smoke_order_equivalence",
        "forward_audit": str(forward_path), "forward_audit_sha256": streaming_hash(forward_path),
        "reverse_audit": str(reverse_path), "reverse_audit_sha256": streaming_hash(reverse_path),
        "forward_manifest_sha256": reports[0]["manifest_sha256"], "reverse_manifest_sha256": reports[1]["manifest_sha256"],
        "dataset_manifest_sha256": first["dataset_manifest_sha256"], "mask_registry_sha256": first["mask_registry_sha256"],
        "precision_addendum_sha256": first["precision_addendum_sha256"], "arithmetic_profile": first["arithmetic_profile"],
        "runtime_source_sha256": first["source_sha256"], "smoke_completed_utc": [m["completed_utc"] for m in manifests],
        **exact, "primary_evaluated": False, "accuracy_gate": False,
        "analysis_source_sha256": streaming_hash(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forward_audit", type=Path)
    parser.add_argument("reverse_audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior smoke equivalence reports")
    report = audit_equivalence(args.forward_audit, args.reverse_audit)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in report.items() if k != "runtime_source_sha256"}), flush=True)


if __name__ == "__main__":
    main()
