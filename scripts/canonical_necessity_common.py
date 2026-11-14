"""Registered same-profile necessity execution, separate from sealed FP32 runs."""

from datetime import datetime
import json
from pathlib import Path

from sas_memory.write_locality_data import streaming_hash
from scripts.copy_necessity_data import DATA_SOURCES, MODELS, REGIMES as MASKED_REGIMES, read_plan
from scripts.check_canonical_adapter_readout import ADDENDUM, ADDENDUM_SHA256

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("full", *MASKED_REGIMES)
ARITHMETIC_AUDIT = Path("/media/990Pro/Code/SAS/artifacts/memory_update/canonical-adapter-readout-check-audit-20261114-01.json")
ARITHMETIC_AUDIT_SHA256 = "6be49e07bcafdeca3ce24b1ee4899b2fa35734e84d44bd539a1cfa032a02d6ab"
RUNTIME_SOURCES = tuple(sorted(set(DATA_SOURCES) | {
    "configs/copy_evidence_necessity_precision_addendum.json",
    "scripts/stable_adapter_linears.py", "scripts/check_canonical_adapter_readout.py",
    "scripts/analyze_canonical_adapter_check.py", "scripts/copy_necessity_metrics.py",
    "scripts/run_copy_evidence_necessity.py", "scripts/analyze_copy_evidence_necessity.py",
    "scripts/canonical_necessity_common.py", "scripts/run_canonical_copy_necessity.py",
    "scripts/analyze_canonical_copy_necessity.py", "scripts/compare_canonical_necessity_smokes.py",
}))


def sources():
    return set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}


def checked_profile(registered, dataset, reexecute=True):
    from scripts.analyze_canonical_adapter_check import audit
    plan = read_plan()
    addendum = json.loads(ADDENDUM.read_text())
    report = json.loads(ARITHMETIC_AUDIT.read_text())
    if (streaming_hash(ADDENDUM) != ADDENDUM_SHA256 or addendum["training"] or addendum["new_adapters"] or
        addendum["base_protocol_sha256"] != streaming_hash(Path(addendum["base_protocol"])) or
        addendum["regimes"] != list(REGIMES) or
        streaming_hash(ROOT / addendum["profile_source"]) != addendum["profile_source_sha256"] or
        addendum["data"]["manifest_sha256"] != streaming_hash(Path(dataset) / "manifest.json") or
        addendum["data"]["mask_registry_sha256"] != registered["mask_registry_sha256"] or
        streaming_hash(ARITHMETIC_AUDIT) != ARITHMETIC_AUDIT_SHA256 or
        report["status"] != "audited_arithmetic_diagnostic_not_scientific_smoke" or
        report["precision_addendum_sha256"] != ADDENDUM_SHA256 or
        report["exact_readouts"] != 108 or report["prefix_identity_checks"] != 24 or report["native_replays"] != 6 or
        streaming_hash(ARITHMETIC_AUDIT.with_suffix(".source.py")) != report["analysis_source_sha256"] or
        not datetime.fromisoformat(plan["registered_utc"]) < datetime.fromisoformat(addendum["registered_utc"])):
        raise ValueError("Registered canonical profile or completed arithmetic diagnostic changed")
    if reexecute and audit() != report:
        raise ValueError("Independent canonical arithmetic diagnostic no longer reproduces")
    return addendum, {"path": str(ARITHMETIC_AUDIT), "sha256": ARITHMETIC_AUDIT_SHA256}


def model_order(reverse=False):
    return list(reversed(MODELS)) if reverse else list(MODELS)


def expected_affines(description):
    return [f"{group}.{name}" for group, field in (("encoders", "encoder_shapes"), ("outputs", "output_shapes"))
            for name in description[field]]


def check_affine_scope(controller, native_model):
    """Reject accidentally wrapping a native module before installing any hooks."""
    from torch import nn
    native_ids = {id(m) for m in native_model.modules()}
    linears = [(n, m) for n, m in controller.named_modules() if isinstance(m, nn.Linear)]
    if (not linears or [n for n, _ in linears] != expected_affines(controller.description()) or
        any(id(m) in native_ids for _, m in linears)):
        raise ValueError("Canonical affine scope overlaps native model modules or omits controller modules")
    return {"native_module_overlap": False, "native_modules_wrapped": 0,
            "linear_modules": [n for n, _ in linears],
            "all_adapter_linear_weights_fp32": all(str(m.weight.dtype) == "torch.float32" for _, m in linears)}


def audit_affines(state, profile):
    names = expected_affines(state["description"])
    scope, arithmetic, calls = state["affine_scope"], state["arithmetic"], state["actual_affine_calls"]
    if (scope != {"native_module_overlap": False, "native_modules_wrapped": 0,
                  "linear_modules": names, "all_adapter_linear_weights_fp32": True} or
        arithmetic != {"profile": profile, "linear_modules": names, "parameter_storage_and_dtype_changed": False,
                       "backbone_arithmetic_changed": False} or
        list(calls) != names or any(type(v) is not int or v < 1 for v in calls.values())):
        raise ValueError("Actual canonical affine coverage or native-module separation failed")


def canonical_metrics(masked, full, cases, registry, source_plan, plan, profile):
    from scripts.copy_necessity_metrics import summarize
    result = summarize(masked, full, cases, registry, source_plan, plan)
    result.update(arithmetic_profile=profile, primary_reference="fresh_full_same_canonical_arithmetic_profile",
                  cross_profile_rows_in_primary=0,
                  prior_sufficiency_status="Recorded historical FP32-profile result only; not reevaluated under this profile.")
    return result


def checked_smoke_equivalence(path, dataset_digest):
    from scripts.compare_canonical_necessity_smokes import audit_equivalence
    path = Path(path)
    report = json.loads(path.read_text())
    expected = audit_equivalence(Path(report["forward_audit"]), Path(report["reverse_audit"]))
    if (report != expected or report["dataset_manifest_sha256"] != dataset_digest or
        streaming_hash(path.with_suffix(".source.py")) != report["analysis_source_sha256"]):
        raise ValueError("Both independently audited432-readout smokes and exact order equivalence required")
    return {"path": str(path), "sha256": streaming_hash(path)}
