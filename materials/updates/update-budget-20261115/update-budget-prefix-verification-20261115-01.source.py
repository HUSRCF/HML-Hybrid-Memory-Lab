"""Independent reader-only prefix preservation under each single-gate policy."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.runtime import TextRuntime,cache_tensors
from sas_memory.kv_readout import QuestionKVReadout
from sas_memory.write_locality_data import streaming_hash
from scripts.update_budget_data import read_plan,registered_inputs,MODELS,actual_prompt,PROTOCOL,FAMILIES,REGIMES
from scripts.update_budget_runtime import ComponentHold
from scripts.copy_readout_runtime import preserve_history
from scripts.semantic_order_path_common import cache_fingerprint
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_load_screen import hook_state,weights_fingerprint
from scripts.run_controller_trial import write_json,state_fingerprint
from scripts.stable_adapter_linears import StableAdapterLinears
from scripts.analyze_update_budget import audit_run


def main():
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve prior auxiliary outputs')
    torch.set_num_threads(4);plan=read_plan();registered,_,_,queries,policies,_=registered_inputs()
    audit=json.loads(args.reference_audit.read_text());source=Path(audit['source_run']);m=json.loads((source/'manifest.json').read_text())
    if audit_run(source,True)!=audit or not m['smoke'] or m['reverse_model_order']:raise ValueError('Exact completed forward smoke required')
    if not datetime.fromisoformat(plan['registered_utc'])<datetime.now(timezone.utc):raise ValueError('Protocol chronology failed')
    controls=registered['control_query_ids'];runtime=TextRuntime(plan['model_path'],'it')
    packages={n:importlib.metadata.version(n) for n in ('torch','transformers','fla-core')}
    if (packages!=plan['packages'] or runtime.model.config.layer_types!=(['linear_attention']*3+['full_attention'])*6 or
        streaming_hash(Path(registered['references']['installed_modeling_source']))!=registered['references']['installed_modeling_source_sha256']):
        raise ValueError('Actual auxiliary native runtime changed')
    hooks=hook_state(runtime.model);before=weights_fingerprint(runtime.model)
    if before['sha256']!=plan['native_loaded_weights_sha256'] or not before['all_parameters_frozen'] or not before['no_parameter_gradients']:
        raise ValueError('Native model changed')
    out={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'reference_audit':str(args.reference_audit),
        'reference_audit_sha256':streaming_hash(args.reference_audit),'protocol_sha256':streaming_hash(PROTOCOL),
        'source_sha256':streaming_hash(Path(__file__)),'primary_source_sha256':m['source_sha256'],
        'weights_before':before,'packages':packages,'checks':[],'adapter_states':{}}
    write_json(args.output,out)
    try:
        for name in MODELS:
            spec=plan['models'][name];controller=None;arithmetic=None
            reference={r:{v['id']:v['readout'] for v in m['models'][name]['condition_controls'][r]} for r in REGIMES}
            try:
                if spec:
                    controller=GroupController(runtime.model,spec['group_index'],spec['rank'],spec['mode'],adapt_gdn_keys=spec['adapt_gdn_keys'])
                    controller.load_state_dict(load_file(spec['checkpoint'],device=str(runtime.model.device)),strict=True);controller.eval().requires_grad_(False)
                    saved=state_fingerprint(controller)
                    if saved!=m['models'][name]['adapter_state_sha256_before']:raise ValueError('Frozen adapter state changed')
                    arithmetic=StableAdapterLinears(controller)
                with torch.inference_mode():
                    for key in controls:
                        q=queries[key];ledger=actual_prompt(runtime,q);prompt=ledger+runtime.query_suffix(q['question'])
                        keep=[b=='1' for b in policies[key]['history_keep_bits']]
                        if controller:controller.set_token_policy(spec['token_policy'],q['ledger_tokens'])
                        for family in FAMILIES:
                            regime=family+'_full';restricted=family+'_oracle'
                            write_keep=[b=='1' for b in policies[key]['write_keep_bits'][family]]
                            with ComponentHold(runtime.model,write_keep,'delta') as hold:
                                full,_,_=runtime.advance(prompt);prefix,logits,count=runtime.advance(ledger)
                                if count!=q['ledger_tokens'] or len(dict(cache_tensors(prefix)))!=48:
                                    raise ValueError('Original boundary or complete48field cache changed')
                                baseline=measure(runtime,ledger,q)
                                if baseline!=reference[regime][key]:raise ValueError('Auxiliary baseline five-field replay failed')
                                proof=preserve_history(runtime,ledger,prompt,keep,prefix,logits,full,cache_fingerprint(prefix))
                                with QuestionKVReadout(runtime.model,keep):value=measure(runtime,ledger,q)
                                if value!=reference[restricted][key]:raise ValueError('Auxiliary restricted five-field replay failed')
                                gates=hold.report()
                            out['checks'].append({'model':name,'regime':regime,'component':'delta','write_family':family,'id':key,**proof,'readout':baseline,
                                'restricted_readout':value,'actual_native_gates':gates})
                            write_json(args.output,out)
                            print(json.dumps({'model':name,'component':'delta','write_family':family,'completed_prefix_checks':len(out['checks'])}),flush=True)
                if controller:
                    if state_fingerprint(controller)!=saved or any(p.grad is not None or p.requires_grad for p in controller.parameters()):raise ValueError('Adapter freeze failed')
                    out['adapter_states'][name]={'before':saved,'after':state_fingerprint(controller)}
            finally:
                if arithmetic:arithmetic.close()
                if controller:controller.close()
            if hook_state(runtime.model)!=hooks:raise ValueError('Auxiliary hooks not restored')
        after=weights_fingerprint(runtime.model)
        if after!=before:raise ValueError('Native weights changed')
        for n,digest in m['source_sha256'].items():
            if streaming_hash(ROOT/n)!=digest:raise ValueError('Preserve every197primary source')
        if len(out['checks'])!=240:raise ValueError('All5models12queries4matchedpolicies required')
        out.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),weights_after=after,
            hooks_restored=True,complete_cache_fields_per_check=48,historical_kv_fields_per_check=12,
            primary_readouts_added=0,training=False)
    except BaseException as e:out.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:
        write_json(args.output,out);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':out['status'],'checks':len(out['checks']),'complete_cache_fields_per_check':48,'historical_kv_fields_per_check':12}),flush=True)

if __name__=='__main__':main()
