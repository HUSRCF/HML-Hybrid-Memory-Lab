"""Independent same-profile full/oracle replication and complete-memory audit."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_copy_evidence_necessity import audit_rows
from scripts.canonical_necessity_common import ADDENDUM_SHA256, audit_affines
from scripts.trained_timing_evidence_access_data import (MODELS, PROTOCOL, MASKED_REGIMES, REGIMES,
    read_plan, registered_inputs, checked_smoke_equivalence, model_order, sources, require_exact_full)
from scripts.copy_readout_data import sample_ids, safety_samples, selected_inputs
from scripts.trained_timing_evidence_access_metrics import summarize
from scripts.copy_readout_runtime import readout, check_weight_stats
from scripts.analyze_unseen_shared_timing import audit_prefix
from scripts.trained_timing_evidence_access_data import unique_prefix_queries
from scripts.run_controller_trial import write_json


def audit_readout(actual, query):
    match = re.fullmatch(r"\s*(-?\d+)\s*", actual["generated"])
    if (actual["answer"] != query["answer"] or actual["query_tokens"] != query["question_tokens"] or
        actual["greedy_exact"] != bool(match and match.group(1) == query["answer"]) or
        not math.isfinite(actual["answer_nll"]) or actual["answer_nll"] < 0):
        raise ValueError("Own-world unrestricted grading, token boundary or finite gold NLL changed")


def audit_full_rows(rows, physical, model, policy):
    if [r["id"] for r in rows] != [q["id"] for q in physical]:
        raise ValueError("Fresh full-history reference coverage/order changed")
    for row, q in zip(rows, physical):
        audit_readout(readout(row), q)
        if (row["model"] != model or row["regime"] != "full" or row["token_policy"] != policy or
            row["prefill"] != "full" or row["registration_sha256"] != data_fingerprint(q) or
            row["mask_registration_sha256"] is not None or row["keep_bits_sha256"] is not None or row["hook_calls"] is not None):
            raise ValueError("Fresh full-history reference unexpectedly masked or mislabeled")


def audit_proofs(proofs, sample_keys, full_reference, indexed_rows):
    if [p["id"] for p in proofs] != sample_keys:
        raise ValueError("Complete prescribed all-role preservation samples required")
    for p in proofs:
        if (not p["unmasked_reference_exact"] or p["unmasked_readout"] != readout(full_reference[p["id"]]) or
            not p["all_history_cache_logits_readout_exact"] or p["all_history_hook_calls"]["prefill"] != 2 or
            p["all_history_hook_calls"]["cached"] < 1 or not p["hooks_restored"] or
            list(p["regimes"]) != list(MASKED_REGIMES) or list(p["unmasked_weights"]) != list(MASKED_REGIMES)):
            raise ValueError("Actual full/all-history controls or hook restoration failed")
        hashes = set()
        for r in MASKED_REGIMES:
            check_weight_stats(p["unmasked_weights"][r], False)
            s = p["regimes"][r]
            if (not s["complete_prefix_memory_exact"] or not s["prefix_logits_exact"] or s["historical_kv_fields_exact"] != 12 or
                s["prefix_sha256"] != s["masked_prefix_sha256"] or not re.fullmatch(r"[0-9a-f]{64}", s["prefix_sha256"]) or
                s["proof_hook_calls"] != {"prefill": 2, "cached": 0} or
                s["readout"] != readout(indexed_rows[r][p["id"]]) or s["measure_hook_calls"] != indexed_rows[r][p["id"]]["hook_calls"]):
                raise ValueError("Complete recurrent/conv/KV/bookkeeping preservation or observed readout changed")
            check_weight_stats(s["attention_stats"], True)
            hashes.add(s["prefix_sha256"])
        if len(hashes) != 1:
            raise ValueError("The two interventions used different original prefix states")


def audit_run(path, allow_smoke=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed" or (manifest["smoke"] and not allow_smoke):
        raise ValueError("Completed canonical full evaluation required; smoke is operational only")
    plan = read_plan()
    dataset = Path(manifest["dataset"])
    registered, source_plan, cases, registry, masks, references = registered_inputs(dataset)
    profile = json.loads(Path(plan["precision_addendum"]).read_text())
    source_manifest = json.loads((Path(plan["source_run"]) / "manifest.json").read_text())
    reference_rows = {m: {r["id"]: r for r in rows} for m, rows in references.items()}
    if (manifest['actual_eos_token_id']!=registered['references']['actual_eos_token_id'] or
        streaming_hash(path/'installed_modeling_source.py')!=registered['references']['installed_modeling_source_sha256']):
        raise ValueError('Actual stopping configuration or installed native source changed')
    physical, bindings = selected_inputs(registry, manifest["smoke"])
    samples, prefix_samples = sample_ids(cases, registry), safety_samples(registry)
    queries = {q["id"]: q for q in physical}
    order = model_order(manifest["reverse_model_order"])
    weights = manifest["weights_after"]
    if (manifest["training"] or manifest["new_adapters"] or
        (manifest["reverse_model_order"] and not manifest["smoke"]) or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or
        manifest["precision_addendum_sha256"] != ADDENDUM_SHA256 or manifest["arithmetic_profile"] != profile["profile"] or
        manifest["primary_reference"] != "fresh_full_same_canonical_arithmetic_profile" or manifest["cross_profile_rows_in_primary"] != 0 or
        manifest["dataset_manifest_sha256"] != streaming_hash(dataset / "manifest.json") or
        manifest["mask_registry_sha256"] != registered["mask_registry_sha256"] or manifest["references"] != registered["references"] or
        json.loads((path / "plan.json").read_text()) != plan or json.loads((path / "precision_addendum.json").read_text()) != profile or
        manifest["model_path"] != plan["model_path"] or manifest["packages"] != plan["packages"] or
        manifest["selected_physical_ids"] != list(queries) or manifest["bindings_per_model_regime"] != len(bindings) or
        manifest["sample_query_ids"] != samples or manifest["native_prefix_sample_ids"] != prefix_samples or
        manifest["generation"] != plan["generation"] or manifest["model_order"] != order or
        manifest["completed_models"] != order or list(manifest["models"]) != order or
        set(manifest["source_sha256"]) != sources() or manifest["weights_before"] != weights or
        weights["sha256"] != plan["native_loaded_weights_sha256"] or
        not weights["all_parameters_frozen"] or not weights["no_parameter_gradients"] or not manifest["hooks_restored"] or
        not (datetime.fromisoformat(plan["registered_utc"]) < datetime.fromisoformat(registered["registered_utc"]) <
             datetime.fromisoformat(manifest["started_utc"]) < datetime.fromisoformat(manifest["completed_utc"])) or
        not datetime.fromisoformat(profile["registered_utc"]) < datetime.fromisoformat(manifest["started_utc"])):
        raise ValueError("Registered canonical scope, runtime, source coverage, chronology or frozen native tensors changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived canonical runtime source changed")
    if manifest["generation_config_sha256"] != {p.name: streaming_hash(p) for p in Path(plan["model_path"]).glob("generation_config.json")}:
        raise ValueError("Actual native generation config changed")
    if manifest["smoke"]:
        if manifest["smoke_equivalence_gate"] is not None:
            raise ValueError("Operational smoke has an unexpected primary gate")
    else:
        gate = manifest["smoke_equivalence_gate"]
        if checked_smoke_equivalence(Path(gate["path"]), manifest["dataset_manifest_sha256"]) != gate:
            raise ValueError("Pre-primary independent smoke equivalence gate changed")
        eq = json.loads(Path(gate["path"]).read_text())
        if (eq["runtime_source_sha256"] != manifest["source_sha256"] or
            any(datetime.fromisoformat(t) >= datetime.fromisoformat(manifest["started_utc"]) for t in eq["smoke_completed_utc"])):
            raise ValueError("Primary run preceded operational smoke or changed its execution source")
    if [r["id"] for r in manifest["native_anchors"]] != samples:
        raise ValueError("Complete independent pre-controller native anchors required")
    anchors = {r["id"]: r["readout"] for r in manifest["native_anchors"]}
    for key, actual in anchors.items():
        audit_readout(actual, queries[key])
        require_exact_full(actual, reference_rows["native"][key], "native_anchor", key)
    all_rows = {}
    proof_count = prefix_count = detach_count = before_count = none_count = 0
    for name in order:
        state, spec = manifest["models"][name], source_plan["models"][name]
        policy = spec["token_policy"] if spec else "none"
        if (state["completed_queries_per_regime"] != len(physical) or state["exact_canonical_full_replays"] != len(physical) or not state["hooks_restored"] or
            state["token_policy"] != policy or list(state["results_sha256"]) != list(REGIMES)):
            raise ValueError("Actual two-model/two-regime coverage or full exact replay changed")
        if spec:
            training = json.loads((Path(spec["training_run"]) / "manifest.json").read_text())
            if (state["description"] != training["arms"][spec["training_arm"]]["description"] or
                state["adapter_parameters"] != spec["parameters"] or state["checkpoint_sha256"] != spec["checkpoint_sha256"] or
                state["adapter_state_sha256_before"] != state["adapter_state_sha256_after"] or
                state["adapter_state_sha256_after"] != registered["references"]["training"][name]["checkpoint_state_sha256"] or
                not state["all_adapter_parameters_frozen"] or not state["arithmetic_hooks_removed"]):
                raise ValueError("Actual frozen adapter tensors, gradients, allocation or cleanup changed")
            audit_affines(state, profile["profile"])
        elif any(k.startswith(("adapter_", "checkpoint_", "arithmetic", "affine_", "actual_affine")) for k in state):
            raise ValueError("Native arm unexpectedly has adapter tensors or canonical hooks")
        all_rows[name] = {}
        for regime in REGIMES:
            result_path = path / f"{name}.{regime}.jsonl"
            if streaming_hash(result_path) != state["results_sha256"][regime]:
                raise ValueError("Actual readout file changed")
            rows = [json.loads(line) for line in result_path.read_text().splitlines()]
            if any(r["arithmetic_profile"] != profile["profile"] or
                   r["reference_policy"] != "fresh_full_same_canonical_arithmetic_profile" or
                   r["canonical_full_reference_sha256"] != data_fingerprint(readout(reference_rows[name][r["id"]])) for r in rows):
                raise ValueError("Cross-profile rows entered primary references or contrasts")
            if regime == "full":
                audit_full_rows(rows, physical, name, policy)
                for row in rows:
                    require_exact_full(row, reference_rows[name][row["id"]], name, row["id"])
            else:
                audit_rows(rows, physical, masks, name, regime, policy)
            all_rows[name][regime] = rows
        indexed = {r: {row["id"]: row for row in rows} for r, rows in all_rows[name].items()}
        audit_proofs(state['proofs'],samples,indexed['full'],indexed)
        proof_count+=len(state['proofs'])
        if name=='native' and any(readout(indexed['full'][key])!=value for key,value in anchors.items()):
            raise ValueError('Native full reference differs from original anchors')
        prefix_queries=unique_prefix_queries(physical) if policy=='question' else []
        audit_prefix(state['prefix_checks'],prefix_queries)
        old_prefixes={p['source_key']:p for p in source_manifest['models'][name]['prefix_checks']}
        if state['prefix_checks']!=[old_prefixes[q['source_key']] for q in prefix_queries]:
            raise ValueError('Complete native prefix differs from the actual completed timing reference')
        hashes={p['source_key']:p['native_prefix_sha256'] for p in state['prefix_checks']}
        for proof in state['proofs']:
            source=queries[proof['id']]['source_key']
            if source in hashes and any(proof['regimes'][r]['prefix_sha256']!=hashes[source] for r in MASKED_REGIMES):
                raise ValueError('Masked prefix differs from independently verified native prefix')
        prefix_count+=len(state['prefix_checks'])
        if [p['id'] for p in state['native_before_attach_checks']]!=samples or any(p['readout']!=anchors[p['id']] for p in state['native_before_attach_checks']):
            raise ValueError('Actual native before-attach anchor coverage or values changed')
        before_count+=len(state['native_before_attach_checks'])
        expected_none=samples if spec else []
        if [p['id'] for p in state['attached_none_checks']]!=expected_none or any(p['readout']!=anchors[p['id']] for p in state['attached_none_checks']):
            raise ValueError('Attached-none identity coverage or values changed')
        none_count+=len(state['attached_none_checks'])
        if ([p["id"] for p in state["native_restore_checks"]] != samples or
            any(not p["exact"] or p["readout"] != anchors[p["id"]] for p in state["native_restore_checks"])):
            raise ValueError("Detached original native full-history readouts failed exact replay")
        detach_count += len(state["native_restore_checks"])
    if (proof_count,prefix_count,detach_count,before_count,none_count)!=(50,6 if manifest["smoke"] else 56,50,50,40):
        raise ValueError("Actual complete-memory/native-proof coverage does not match registration")
    metrics = ({"primary_evaluated": False} if manifest["smoke"] else
               summarize({m: all_rows[m] for m in MODELS}, cases, registry, plan))
    return {"status": "audited_trained_timing_evidence_access_smoke" if manifest["smoke"] else "completed_and_audited_trained_timing_evidence_access",
        "source_run": str(path), "manifest_sha256": streaming_hash(path / "manifest.json"),
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"], "mask_registry_sha256": registered["mask_registry_sha256"],
        "precision_addendum_sha256": ADDENDUM_SHA256, "arithmetic_profile": profile["profile"],
        "weights_sha256": weights["sha256"], "references": registered["references"], "model_order": order,
        "new_physical_readouts_audited": len(physical) * 15, "fresh_full_reference_readouts_audited": len(physical) * 5,
        "new_masked_readouts_audited": len(physical) * 10, "new_related_bindings_audited": len(bindings) * 15,
        "cross_profile_rows_in_primary": 0, "native_anchors": len(anchors),
        "all_history_exact_controls": proof_count, "actual_masked_attention_checks": 2*proof_count,
        "complete_prefix_and_historical_KV_proofs": 2*proof_count, "detached_native_exact_replays": detach_count,
        "question_only_native_prefix_checks": prefix_count, "exact_completed_canonical_full_replays": len(physical) * 5,
        "native_before_attach_exact_replays":before_count,"attached_none_identities":none_count,
        "source_files_verified":len(sources()),"prior_timing_criteria_unchanged":registered["references"]["prior_timing_criteria_unchanged"],
        "metrics": metrics, "interpretation": plan["interpretation_limits"],
        "analysis_source_sha256": streaming_hash(Path(__file__))}


def compare_observations(first,second,a,b):
    if (first['model_order']!=model_order() or second['model_order']!=model_order(True) or
        first['selected_physical_ids']!=second['selected_physical_ids'] or len(first['selected_physical_ids'])!=18 or
        first['native_anchors']!=second['native_anchors']):raise ValueError('Both complete opposite operational orders required')
    count=0
    for name in MODELS:
        for regime in REGIMES:
            if a[name][regime]!=b[name][regime] or [r['id'] for r in a[name][regime]]!=first['selected_physical_ids']:
                raise ValueError(f'Exact complete readout, mask or execution receipts differ: {name},{regime}')
            count+=len(a[name][regime])
        for field in ('proofs','prefix_checks','native_before_attach_checks','attached_none_checks','native_restore_checks',
            'description','arithmetic','adapter_state_sha256_before','adapter_state_sha256_after','actual_affine_calls'):
            if first['models'][name].get(field)!=second['models'][name].get(field):raise ValueError(f'Complete prefix/scope control differs: {name},{field}')
    if count!=270:raise ValueError('All270exact readouts required')
    return {'exact_readouts':count,'exact_complete_memory_samples':50,'exact_masked_prefix_pairs':100,
        'exact_question_only_prefix_checks':6,'exact_native_anchors':10}


def audit_equivalence(forward_path,reverse_path):
    manifests=[];rows=[]
    for path in [Path(forward_path),Path(reverse_path)]:
        report=json.loads(path.read_text())
        if (report['status']!='audited_trained_timing_evidence_access_smoke' or report['new_physical_readouts_audited']!=270 or
            streaming_hash(path.with_suffix('.source.py'))!=report['analysis_source_sha256'] or audit_run(Path(report['source_run']),True)!=report):
            raise ValueError('Both independent270-readout operational audits required')
        run=Path(report['source_run']);manifests.append(json.loads((run/'manifest.json').read_text()))
        rows.append({n:{r:[json.loads(line) for line in (run/f'{n}.{r}.jsonl').read_text().splitlines()] for r in REGIMES} for n in MODELS})
    first,second=manifests
    if first['reverse_model_order'] or not second['reverse_model_order'] or any(first[k]!=second[k] for k in
        ('source_sha256','dataset_manifest_sha256','mask_registry_sha256','references','arithmetic_profile','weights_before','generation','generation_config_sha256')):
        raise ValueError('Operational order comparison changed frozen execution scope')
    exact=compare_observations(first,second,rows[0],rows[1])
    return {'status':'audited_trained_timing_evidence_access_smoke_equivalence','forward_audit':str(forward_path),
        'reverse_audit':str(reverse_path),'forward_audit_sha256':streaming_hash(Path(forward_path)),
        'reverse_audit_sha256':streaming_hash(Path(reverse_path)),'dataset_manifest_sha256':first['dataset_manifest_sha256'],
        'runtime_source_sha256':first['source_sha256'],'smoke_completed_utc':[m['completed_utc'] for m in manifests],
        **exact,'primary_evaluated':False,'accuracy_gate':False,'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path,nargs='?');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-smoke',action='store_true');parser.add_argument('--equivalence',type=Path,nargs=2);args=parser.parse_args()
    if args.output.exists() or bool(args.run)==bool(args.equivalence):raise ValueError('Preserve audits; choose one run or two smoke audits')
    report=audit_equivalence(*args.equivalence) if args.equivalence else audit_run(args.run,args.allow_smoke)
    write_json(args.output,report);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:report[k] for k in ['status','new_physical_readouts_audited','exact_readouts','source_files_verified'] if k in report}),flush=True)


if __name__=='__main__':main()
