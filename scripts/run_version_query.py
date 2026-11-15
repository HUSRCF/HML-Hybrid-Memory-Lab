"""Frozen-runner reuse for all registered token-matched version questions."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.version_query_data import (read_plan,registered_inputs,sources,context_queries,selected_contexts,
    DESTINATION,PROTOCOL,MODELS,READERS)
from scripts import run_natural_load as frozen
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash

RUNNER_REUSE={'function':'scripts.run_natural_load.run_model','code_object_reused':True,
    'rebound_globals':['READERS','context_queries'],'original_globals_untouched':True}


def cloned_run_model():
    original=frozen.run_model;bindings=dict(original.__globals__)
    bindings.update(READERS=READERS,context_queries=context_queries)
    changed=[k for k in bindings if bindings[k] is not original.__globals__[k]]
    if sorted(changed)!=RUNNER_REUSE['rebound_globals']:
        raise ValueError('Only the registered reader/question bindings may change')
    clone=types.FunctionType(original.__code__,bindings,original.__name__,original.__defaults__,original.__closure__)
    clone.__kwdefaults__=original.__kwdefaults__
    if clone.__code__ is not original.__code__ or clone.__globals__ is original.__globals__:
        raise ValueError('Independent bindings with exact original code required')
    return clone


def invoke_model(*args):
    before=dict(frozen.run_model.__globals__);clone=cloned_run_model()
    try:
        return clone(*args)
    finally:
        actual=frozen.run_model.__globals__
        if set(actual)!=set(before) or any(actual[k] is not v for k,v in before.items()):
            raise ValueError('Original frozen module globals were modified')


def core_cache(record):return {k:v for k,v in record.items() if k!='source_guard_checks'}


def verify_model_sources(state,reference):
    if any(k not in reference['prefixes'] for k in state['prefixes']) or any(k not in reference['sources'] for k in state['sources']):
        raise ValueError('Unknown normally encoded source')
    for key,cache in state['prefixes'].items():
        if core_cache(cache)!=core_cache(reference['prefixes'][key]):
            raise ValueError('Actual48field normal prefix differs from predecessor')
    for key,source in state['sources'].items():
        old=reference['sources'][key]
        if (core_cache(source['cache'])!=core_cache(old['cache']) or source['tail_proof']!=old['tail_proof'] or
            not source['all_sources_built_before_any_question'] or not source['pairwise_storage_isolated']):
            raise ValueError('Actual48field normal tail or proof differs from predecessor')
    return {'normal_prefix_caches_exact':len(state['prefixes']),'normal_after_caches_exact':len(state['sources']),
        'all48_tensor_fields_logits_layouts_metadata_exact':True,'tail_gate_and_historical_kv_proofs_exact':True,
        'guard_count_only_difference':True}


def checked_gate(path,packet_digest):
    from scripts.analyze_version_query import audit_equivalence
    path=Path(path);g=json.loads(path.read_text())
    if (g!=audit_equivalence(Path(g['forward_run']),Path(g['reverse_run'])) or
        g['packet_manifest_sha256']!=packet_digest or streaming_hash(path.with_suffix('.source.py'))!=g['analysis_source_sha256']):
        raise ValueError('Exact independent opposite-order108readout smokes required')
    return {'path':str(path),'sha256':streaming_hash(path)}


def primary_subset_replay(output,manifest,gate):
    g=json.loads(Path(gate['path']).read_text());forward=Path(g['forward_run'])
    sm=json.loads((forward/'manifest.json').read_text())
    if streaming_hash(forward/'manifest.json')!=g['forward_manifest_sha256']:
        raise ValueError('Completed smoke bytes changed')
    result={}
    for name in MODELS:
        smoke=[json.loads(l) for l in (forward/f'{name}.jsonl').read_text().splitlines()]
        primary=[json.loads(l) for l in (output/f'{name}.jsonl').read_text().splitlines()]
        ids={r['id'] for r in smoke};overlap=[r for r in primary if r['id'] in ids]
        if len(smoke)!=54 or overlap!=smoke:
            raise ValueError('Primary shared108readout subset differs from smoke')
        for field in ('prefixes','sources'):
            for key,old in sm['models'][name][field].items():
                if manifest['models'][name][field][key]!=old:
                    raise ValueError('Shared smoke/primary source bytes or guard bookkeeping differ')
        result[name]={'exact_readouts':54,'exact_prefix_caches':1,'exact_after_caches':8}
    return result


def execute_run(args):
    import torch
    torch.set_num_threads(4)
    if args.output.exists() or args.smoke==bool(args.smoke_equivalence) or (args.reverse_model_order and not args.smoke):
        raise ValueError('New smoke or gated primary required')
    plan=read_plan();registered,cases,registry,queries,refs=registered_inputs(args.packet)
    gate=None if args.smoke else checked_gate(args.smoke_equivalence,streaming_hash(args.packet/'manifest.json'))
    profile=json.loads(Path(plan['precision_addendum']).read_text());correction=frozen.validation_addendum()
    order=list(MODELS[::-1] if args.reverse_model_order else MODELS)
    m={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'training':False,'new_adapters':False,
        'fresh_streams':False,'reused_cohort':True,'smoke':args.smoke,'reverse_model_order':args.reverse_model_order,
        'model_order':order,'packet':str(args.packet),'packet_manifest_sha256':streaming_hash(args.packet/'manifest.json'),
        'protocol_sha256':streaming_hash(PROTOCOL),'source_sha256':{n:streaming_hash(ROOT/n) for n in sorted(sources())},
        'models':{},'completed_models':[],'native_anchors':[],'model_path':plan['model_path'],
        'generation':plan['generation'],'arithmetic_profile':profile['profile'],
        'precision_addendum_sha256':streaming_hash(Path(plan['precision_addendum'])),
        'validation_addendum_sha256':frozen.VALIDATION_ADDENDUM_SHA256,'smoke_equivalence_gate':gate,
        'generation_config_sha256':{p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')},
        'runner_reuse':RUNNER_REUSE}
    args.output.mkdir(parents=True,exist_ok=False);mp=args.output/'manifest.json'
    for name,value in (('plan',plan),('precision_addendum',profile),('kernel_validation_addendum',correction)):
        write_json(args.output/(name+'.json'),value)
    for name in m['source_sha256']:
        path=args.output/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    write_json(mp,m)
    try:
        runtime=frozen.TextRuntime(plan['model_path'],'it')
        m.update(python=sys.executable,gpu=torch.cuda.get_device_name(0),
            packages={n:importlib.metadata.version(n) for n in ('torch','transformers','fla-core')},
            actual_eos_token_id=runtime.model.generation_config.eos_token_id)
        if (m['packages']!=plan['packages'] or m['actual_eos_token_id']!=refs['runtime']['actual_eos_token_id'] or
            m['generation_config_sha256']!=refs['runtime']['generation_config_sha256'] or
            runtime.model.config.layer_types!=(['linear_attention']*3+['full_attention'])*6):
            raise ValueError('Actual native runtime changed')
        installed=Path(plan['installed_modeling_source'])
        if streaming_hash(installed)!=plan['installed_modeling_source_sha256']:raise ValueError('Native source changed')
        (args.output/'installed_modeling_source.py').write_bytes(installed.read_bytes())
        native_hooks=frozen.hook_state(runtime.model);m['weights_before']=frozen.weights_fingerprint(runtime.model)
        if (m['weights_before']['sha256']!=plan['native_loaded_weights_sha256'] or
            not m['weights_before']['all_parameters_frozen'] or not m['weights_before']['no_parameter_gradients']):
            raise ValueError('Native loaded weights or freeze changed')
        m['native_component_kernel_checks']=frozen.component_kernel_contract(runtime);anchors={}
        with torch.inference_mode():
            for key in refs['legacy_ids']:
                q=refs['legacy_queries'][key];v=frozen.measure(runtime,frozen.actual_prompt(runtime,q),q)
                if v!=refs['legacy_full']['native'][key]:raise ValueError('Native anchor changed')
                anchors[key]=v;m['native_anchors'].append({'id':key,'readout':v})
            for name in order:
                state={'token_policy':'all' if plan['models'][name] else 'none','completed_primary':0,
                    'native_before_attach_checks':[],'attached_none_checks':[],'legacy_full_checks':[],
                    'native_restore_checks':[],'prefixes':{},'sources':{}}
                m['models'][name]=state;write_json(mp,m)
                invoke_model(runtime,name,plan,registry,queries,refs,args.smoke,args.output,state,anchors,lambda:write_json(mp,m))
                state['runner_bindings']=RUNNER_REUSE
                state['predecessor_sources']=verify_model_sources(state,refs['source_caches'][name])
                with (args.output/f'{name}.references.jsonl').open('w') as archived:
                    for row in refs['archived_full'][name]:
                        archived.write(json.dumps({**row,'archived_reference':True,'predecessor_source_run':plan['source_run'],
                            'predecessor_manifest_sha256':plan['source_manifest_sha256']})+'\n')
                state['archived_references']=864;state['references_sha256']=streaming_hash(args.output/f'{name}.references.jsonl')
                m['completed_models'].append(name);write_json(mp,m)
        m['weights_after']=frozen.weights_fingerprint(runtime.model);m['hooks_restored']=frozen.hook_state(runtime.model)==native_hooks
        if m['weights_after']!=m['weights_before'] or not m['hooks_restored']:raise ValueError('Native weights or hooks changed')
        m['primary_same_subset_replay']=None if args.smoke else primary_subset_replay(args.output,m,gate)
        m.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:
        m.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:write_json(mp,m)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,default=DESTINATION)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--reverse-model-order',action='store_true');p.add_argument('--smoke-equivalence',type=Path)
    execute_run(p.parse_args())


if __name__=='__main__':main()
