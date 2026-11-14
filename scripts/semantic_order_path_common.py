"""Fixed-cohort inputs and complete-cache controls for the native path probe."""

import copy
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.training_data import data_fingerprint
from scripts.analyze_initial_order_control import registered_inputs
from scripts.analyze_order_augmented_semantics import audit_run as audit_augmentation
from scripts.analyze_paired_semantic_probe import identity
from scripts.compare_paired_semantic_controllers import functional_units, paired_counts


CONDITIONS = {
    "canonical_natural": {"recurrent_and_convolution": "canonical", "kv": "canonical"},
    "permuted_natural": {"recurrent_and_convolution": "permuted", "kv": "permuted"},
    "canonical_state_permuted_kv": {"recurrent_and_convolution": "canonical", "kv": "permuted"},
    "permuted_state_canonical_kv": {"recurrent_and_convolution": "permuted", "kv": "canonical"},
}


def registered_probe_inputs(plan):
    if (plan["conditions"] != CONDITIONS or list(plan["conditions"]) != list(CONDITIONS) or
        plan["training"] or plan["adapters"] or plan["backbone"] != "it" or plan["reader_policy"] != "full" or
        plan["groups"] != list(range(6)) or plan["examples_per_condition"] != 432 or plan["query_readouts"] != 1728):
        raise ValueError("Native path probe changed its registered scope")
    order_path = Path(plan["order_control_plan"])
    if file_hash(order_path) != plan["order_control_plan_sha256"]:
        raise ValueError("Registered order protocol changed")
    order_plan = json.loads(order_path.read_text())
    permuted, base, data, cases, registry = registered_inputs(order_plan)
    checks = {
        "canonical_manifest_sha256": file_hash(Path(plan["canonical_data_source"]) / "manifest.json"),
        "permuted_manifest_sha256": file_hash(Path(plan["permuted_data_source"]) / "manifest.json"),
        "base_data_sha256": base["data_sha256"], "cases_sha256": permuted["cases_sha256"],
        "registry_sha256": permuted["registry_sha256"],
    }
    if any(plan[key] != value for key, value in checks.items()):
        raise ValueError("Native path probe data registration changed")
    if (Path(order_plan["data_source"]) != Path(plan["permuted_data_source"]) or
        Path(permuted["base_data"]) != Path(plan["canonical_data_source"]) or
        order_plan["native_reference_run"] != plan["full_reference_native_run"] or
        len(data["validation"]) != plan["examples_per_condition"]):
        raise ValueError("Path probe no longer matches the natural development cohorts")
    canonical_registry = json.loads((Path(plan["canonical_data_source"]) / "registry.json").read_text())["validation"]
    indexed = {"canonical": {(r["case_id"], r["variant"], r["role"]): r for r in canonical_registry},
               "permuted": {(r["case_id"], r["variant"], r["role"]): r for r in registry}}
    if indexed["canonical"].keys() != indexed["permuted"].keys():
        raise ValueError("Natural cache origins use different queries")
    for key in indexed["canonical"]:
        left, right = [indexed[origin][key] for origin in ("canonical", "permuted")]
        for field in ("ledger_tokens", "prompt_tokens", "question_ids_sha256"):
            if left[field] != right[field]:
                raise ValueError("Natural cache origins have unmatched query positions or answers")
    if len({(e["case"]["id"], e["variant"]) for e in data["validation"]}) != plan["unique_prefixes_per_order"]:
        raise ValueError("Registered number of natural prefix caches changed")
    return base, data, cases, indexed


def reference_gate(plan):
    reference = Path(plan["augmentation_reference_run"])
    if file_hash(reference / "plan.json") != plan["augmentation_plan_sha256"]:
        raise ValueError("Ordinary-augmentation protocol changed")
    augmentation_plan = json.loads((reference / "plan.json").read_text())
    if (augmentation_plan["native_reference_run"] != plan["full_reference_native_run"] or
        augmentation_plan["order_reference_run"] != plan["full_reference_order_run"]):
        raise ValueError("Path probe changed its full-prefill reference cohorts")
    audit = audit_augmentation(reference)
    if audit["status"] != "completed_and_audited" or audit["validation_records_audited"] != 3456:
        raise ValueError("The complete ordinary-augmentation audit must pass before native path inference")
    return audit


def selected_examples(data, smoke):
    if not smoke:
        return data["validation"]
    first = data["validation"][0]["case"]["base_case_id"]
    return [e for e in data["validation"] if e["case"]["base_case_id"] == first]


def cache_metadata(cache):
    def plain(value):
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        return copy.deepcopy(value)
    fields = ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size", "is_initialized")
    return {"sequence_length": int(cache.get_seq_length()),
            "layers": [{field: plain(getattr(layer, field)) for field in fields if hasattr(layer, field)} for layer in cache.layers]}


def assert_complete_cache_equal(left, right):
    from sas_memory.runtime import assert_cache_equal
    assert_cache_equal(left, right)
    if cache_metadata(left) != cache_metadata(right):
        raise AssertionError("Probe changed source cache bookkeeping")


def cache_fingerprint(cache):
    import torch
    from sas_memory.runtime import cache_tensors
    digest = hashlib.sha256(json.dumps(cache_metadata(cache), sort_keys=True).encode())
    for key, tensor in cache_tensors(cache):
        value = tensor.detach().contiguous().cpu()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_origins(cache, originals, origins):
    import torch
    from sas_memory.runtime import cache_tensors
    members = {name: dict(cache_tensors(value)) for name, value in originals.items()}
    actual = dict(cache_tensors(cache))
    if any(actual.keys() != source.keys() for source in members.values()):
        raise AssertionError("Counterfactual cache structure differs from its sources")
    for key, tensor in actual.items():
        origin = origins["recurrent_and_convolution" if key[1] in ("conv_states", "recurrent_states") else "kv"]
        if not torch.equal(tensor, members[origin][key]):
            raise AssertionError("Counterfactual tensor came from the wrong source")
    metadata = cache_metadata(cache)
    for index, layer in enumerate(metadata["layers"]):
        origin = origins["kv" if index % 4 == 3 else "recurrent_and_convolution"]
        expected = cache_metadata(originals[origin])
        if layer != expected["layers"][index] or metadata["sequence_length"] != expected["sequence_length"]:
            raise AssertionError("Counterfactual bookkeeping came from the wrong source")


def functional_report(rows_by_condition):
    units = {name: functional_units(rows) for name, rows in rows_by_condition.items()}
    counts = {name: {metric: {"correct": sum(values.values()), "units": len(values)} for metric, values in metrics.items()}
              for name, metrics in units.items()}
    comparisons = {name: {metric: paired_counts(units["canonical_natural"][metric], metrics[metric]) for metric in metrics}
                   for name, metrics in units.items() if name != "canonical_natural"}
    versus_permuted = {name: {metric: paired_counts(units["permuted_natural"][metric], metrics[metric]) for metric in metrics}
                       for name, metrics in units.items() if name != "permuted_natural"}
    updates = {name: {family: {"correct": sum(row["greedy_exact"] for row in rows if row["family"] == family and row["role"] == "target_current" and row["variant"] != "before"),
                              "examples": sum(row["family"] == family and row["role"] == "target_current" and row["variant"] != "before" for row in rows)}
                      for family in sorted({row["family"] for row in rows})} for name, rows in rows_by_condition.items()}
    return {"counts": counts, "versus_canonical_natural": comparisons, "versus_permuted_natural": versus_permuted,
            "target_updates_by_family": updates}


def nll_contrasts(rows_by_condition):
    indexed = {name: {(r["case_id"], r["variant"], r["role"]): r for r in rows} for name, rows in rows_by_condition.items()}
    keys = list(indexed["canonical_natural"])
    if set(indexed) != set(CONDITIONS) or any(set(rows) != set(keys) for rows in indexed.values()):
        raise ValueError("NLL contrasts require the complete paired four-condition cohort")
    contrasts = []
    for key in keys:
        values = {name: rows[key]["answer_nll"] for name, rows in indexed.items()}
        cc, pp, cp, pc = [values[name] for name in CONDITIONS]
        contrasts.append({"key": list(key), "gold_nll": values,
            "recurrent_permuted_minus_canonical_at_canonical_kv": pc - cc,
            "recurrent_permuted_minus_canonical_at_permuted_kv": pp - cp,
            "kv_permuted_minus_canonical_at_canonical_state": cp - cc,
            "kv_permuted_minus_canonical_at_permuted_state": pp - pc,
            "interaction_difference_in_differences": pp - cp - pc + cc})
    fields = [field for field in contrasts[0] if field not in ("key", "gold_nll")]
    return {"definition": "Gold answer NLL; positive contrasts mean higher loss, not more useful memory. Mixed caches are counterfactual and can be out of distribution.",
            "means": {field: mean(row[field] for row in contrasts) for field in fields}, "paired_examples": contrasts}
