"""Frozen inputs, complete cache origins and passive continuation observations."""

from datetime import datetime
import json
from pathlib import Path

from sas_memory.runtime import cache_tensors, fork_cache
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.fresh_layer_controls import tensor_hash
from scripts.question_path_controls import input_scope as mediation_inputs, sources as mediation_sources
from scripts.semantic_order_path_common import cache_fingerprint, cache_metadata

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/fresh_state_continuation_plan.json'
PROTOCOL_SHA256 = '62f2684e0ad9315facb2e56cec76633519b07b48ff496f4374530d355ee779d9'
LAYERS = (20, 21, 22)
FIELDS = ('conv_states', 'recurrent_states')
TARGETS = {(i, field, 0) for i in LAYERS for field in FIELDS}
CONDITIONS = {
    'native_reference': dict(base_cache='unmasked', conv_states='unmasked', recurrent_states='unmasked'),
    'intervened_states': dict(base_cache='both_paths_restored', conv_states='both_paths_restored', recurrent_states='both_paths_restored'),
    'restore_convolution': dict(base_cache='both_paths_restored', conv_states='unmasked', recurrent_states='both_paths_restored'),
    'restore_recurrent': dict(base_cache='both_paths_restored', conv_states='both_paths_restored', recurrent_states='unmasked'),
    'restore_both': dict(base_cache='both_paths_restored', conv_states='unmasked', recurrent_states='unmasked'),
}


def sources():
    return mediation_sources() | {'configs/fresh_state_continuation_plan.json',
        'scripts/state_continuation_controls.py', 'scripts/run_state_continuation.py',
        'scripts/analyze_state_continuation.py'}


def input_scope():
    old_scope = mediation_inputs(); old = old_scope['plan']; plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or
        plan['status'] != 'registered_before_state_continuation_inference' or
        plan['conditions'] != CONDITIONS or plan['condition_order'] != list(CONDITIONS) or
        plan['state_layers'] != list(LAYERS) or plan['state_fields'] != list(FIELDS) or
        not all(plan[k] for k in ('posthoc','selection_is_posthoc','no_accuracy_gate','full_generation_scored')) or
        any(plan[k] for k in ('independent_scientific_confirmation','can_change_prior_gates','new_queries',
            'new_training','new_models','new_environment')) or
        (plan['physical_cached_continuation_readouts'],plan['required_exact_source_first_readouts'],
         plan['preserved_runtime_source_count']) != (20,32,125) or
        any(plan[k] != old[k] for k in ('model_path','packages','native_loaded_weights_sha256','arithmetic_profile',
            'precision_addendum_sha256','dataset','dataset_manifest_sha256','mask_registry_sha256',
            'models','fixed_checkpoints','queries','attention_layers','installed_modeling_source_sha256')) or
        streaming_hash(Path(plan['prior_mediation_plan'])) != plan['prior_mediation_plan_sha256']):
        raise ValueError('Registered state-continuation scope changed')
    run = Path(plan['prior_mediation_run']); manifest = json.loads((run/'manifest.json').read_text())
    audit_path = Path(plan['prior_mediation_audit']); audit = json.loads(audit_path.read_text())
    if (streaming_hash(run/'manifest.json') != plan['prior_mediation_manifest_sha256'] or
        streaming_hash(audit_path) != plan['prior_mediation_audit_sha256'] or
        audit['manifest_sha256'] != plan['prior_mediation_manifest_sha256'] or
        not audit['operational_controls_passed'] or audit['counts']['archived_full_vectors'] != 32 or
        audit['original_primary_gates_unchanged'] != old_scope['primary_gates'] or
        set(manifest['source_sha256']) != mediation_sources() or
        streaming_hash(run/'installed_modeling_source.py') != plan['installed_modeling_source_sha256'] or
        not datetime.fromisoformat(manifest['completed_utc']) < datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Completed mediation provenance or chronology changed')
    for name, digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT/name) != digest or streaming_hash(run/'sources'/name) != digest:
            raise ValueError('Preserve all125sealed mediation sources')
    references = []
    for name in plan['models']:
        meta = manifest['models'][name]; path = run/f'{name}.jsonl'
        if (streaming_hash(path) != meta['results_sha256'] or
            streaming_hash(run/f'{name}.logits.safetensors') != meta['logits_file_sha256']):
            raise ValueError('Prior source rows or archived full vectors changed')
        for row, proof in zip([json.loads(x) for x in path.read_text().splitlines()],meta['proofs'],strict=True):
            references.append({k:row[k] for k in ('model','id','context','patch','first_token','logits_key',
                'fixed_competitor_id','fixed_pair_margin')} | {k:proof[k] for k in ('prefix_sha256',
                'full_question_cache_sha256','attention23_input_sha256','recurrent_conv_sha256','earlier_position_kv_sha256')})
    if references != plan['source_references']:
        raise ValueError('All32registered source observations must match actual completed mediation')
    return {**old_scope,'mediation_scope':old_scope,'plan':plan,'mediation_manifest':manifest,'mediation_audit':audit}


def complete_metadata(cache):
    """Include all actual instance attributes, container types and tensor layouts."""
    import torch
    def plain(value):
        if isinstance(value,torch.Tensor):
            return {'tensor':True,'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device)}
        if isinstance(value,dict):
            return {'dict':[[repr(k),plain(v)] for k,v in value.items()]}
        if isinstance(value,(list,tuple)):
            return {type(value).__name__:[plain(x) for x in value]}
        if isinstance(value,(torch.device,torch.dtype)):return str(value)
        if value is None or isinstance(value,(bool,int,float,str)):return value
        raise ValueError(f'Unaccounted cache bookkeeping type: {type(value).__name__}')
    return {'type':type(cache).__name__,'original_metadata':cache_metadata(cache),
        'attributes':plain({k:v for k,v in vars(cache).items() if k!='layers'}),
        'layers':[{'type':type(x).__name__,'attributes':plain(vars(x))} for x in cache.layers]}


def cache_record(cache):
    return {'sha256':cache_fingerprint(cache),'metadata':complete_metadata(cache),
        'tensors':{str(k):{'sha256':tensor_hash(v),'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in cache_tensors(cache)}}


def verify_sources(native, intervened):
    import torch
    left,right=dict(cache_tensors(native)),dict(cache_tensors(intervened))
    if left.keys()!=right.keys() or not TARGETS.issubset(left) or complete_metadata(native)!=complete_metadata(intervened):
        raise ValueError('Source cache layout or complete bookkeeping mismatch')
    kv=[k for k in left if k[1] in ('keys','values')]
    if len(kv)!=12:raise ValueError('All12complete K/V fields required')
    changed=[str(k) for k in left if not torch.equal(left[k],right[k])]
    if any(not torch.equal(left[k],right[k]) for k in left if k not in TARGETS):
        raise ValueError('Both-path source differs outside registered six state tensors')
    return {'unmasked':cache_record(native),'both_paths_restored':cache_record(intervened),
        'complete_KV_exact_fields':12,'all_other_tensors_exact':True,'complete_bookkeeping_exact':True,
        'changed_tensor_keys':changed}


def make_condition(origins, condition):
    import torch
    if condition not in CONDITIONS:raise ValueError('Unknown registered cache condition')
    rule=CONDITIONS[condition]; before={k:cache_record(v) for k,v in origins.items()}
    work=fork_cache(origins[rule['base_cache']]); actual=dict(cache_tensors(work))
    members={k:dict(cache_tensors(v)) for k,v in origins.items()}; receipts={}
    for key,value in actual.items():
        origin=rule[key[1]] if key in TARGETS else 'unmasked'
        source=members[origin][key]
        if key in TARGETS:value.copy_(source)
        if not torch.equal(value,source) or any(value.data_ptr()==v[key].data_ptr() for v in members.values()):
            raise ValueError('Cache transplant wrong source or alias')
        receipts[str(key)]={'origin':origin,'sha256':tensor_hash(value),'shape':list(value.shape),'dtype':str(value.dtype)}
    if complete_metadata(work)!=complete_metadata(origins['unmasked']):
        raise ValueError('Transplant changed complete bookkeeping')
    if any(cache_record(v)!=before[k] for k,v in origins.items()):
        raise ValueError('Transplant changed actual source cache')
    record=cache_record(work)
    if condition in ('native_reference','restore_both') and record!=before['unmasked']:
        raise ValueError('Complete-cache restoration failed')
    return work,{'condition':condition,'origins':receipts,'starting_cache':record,
        'source_caches_unchanged_after_transplant':True,'all_tensors_unaliased':True}


class ForwardTrace:
    """Passively observe original runtime forwards and their actual RoPE positions."""
    def __init__(self,model):
        self.calls=[];self.vectors=[];self.handles=[
            model.register_forward_pre_hook(self.before,with_kwargs=True),
            model.model.rotary_emb.register_forward_pre_hook(self.positions,with_kwargs=True),
            model.register_forward_hook(self.after,with_kwargs=True)]
    def before(self,module,args,kwargs):
        ids=kwargs['input_ids'];cache=kwargs['past_key_values']
        if (ids.ndim!=2 or ids.shape[0]!=1 or not kwargs['use_cache'] or
            kwargs.get('attention_mask') is not None or kwargs.get('position_ids') is not None):
            raise ValueError('Original unmasked cached decode shape/policy required')
        self.calls.append({'input_ids':ids.tolist(),'start':int(cache.get_seq_length()),
            'logits_to_keep':kwargs['logits_to_keep'],'position_ids':None})
    def positions(self,module,args,kwargs):
        import torch
        positions=kwargs.get('position_ids',args[1] if len(args)>1 else None)
        call=self.calls[-1];n=len(call['input_ids'][0])
        expected=torch.arange(call['start'],call['start']+n,device=positions.device).view(1,1,n).expand(3,1,n)
        if call['position_ids'] is not None or not torch.equal(positions,expected):
            raise ValueError('Actual cached RoPE positions changed')
        call['position_ids']=positions.tolist()
    def after(self,module,args,kwargs,result):
        if self.calls[-1]['position_ids'] is None:raise ValueError('Actual RoPE observation missing')
        value=result.logits.float().detach().cpu().contiguous().clone()
        self.vectors.append(value)
        self.calls[-1].update(logits_sha256=tensor_hash(value),logits_shape=list(value.shape))
    def __enter__(self):return self
    def __exit__(self,*_):
        for handle in self.handles:handle.remove()


def gold_losses(first,continuation,answer_ids,device):
    """Reconstruct the original singleton candidate_scores operations and shapes."""
    import torch
    first=first.to(device);ids=torch.tensor([answer_ids],dtype=torch.long,device=device)
    first_logp=first.log_softmax(-1);losses=-first_logp[0,ids[:,0]]
    first_nll=losses.item();terms=[];continuation_nll=0.
    if len(answer_ids)>1:
        if continuation is None or list(continuation.shape)!=[1,len(answer_ids)-1,first.shape[-1]]:
            raise ValueError('Complete teacher-forced continuation logits required')
        logp=continuation.to(device).log_softmax(-1)
        selected=logp.gather(-1,ids[:,1:,None]).squeeze(-1)
        contribution=-selected.sum(-1)
        continuation_nll=contribution.item();terms=(-selected).tolist()[0]
        losses+=contribution
    elif continuation is not None:raise ValueError('Unexpected continuation for single-token answer')
    total=losses.item()
    return {'nll_sum':total,'tokens':len(answer_ids),'nll_mean':total/len(answer_ids),
        'first_token_nll':first_nll,'continuation_nll':continuation_nll,'continuation_token_nlls':terms}


def decode_vectors(tokenizer,first,forward_vectors,eos,max_tokens=12):
    """Reconstruct greedy_answer including its final unused max-length forward."""
    eos=set(eos if isinstance(eos,list) else [eos]);generated=[];decisions=[];calls=0;reason='max_tokens'
    for step in range(max_tokens):
        vector=first if step==0 else forward_vectors[step-1][:,-1]
        token=vector.argmax(-1).item();decisions.append(token)
        if token in eos:reason='eos';break
        generated.append(token);text=tokenizer.decode(generated,skip_special_tokens=True)
        if '\n' in text and text.strip():reason='newline';break
        calls+=1
    if calls!=len(forward_vectors):raise ValueError('Observed greedy forwards violate original stopping rule')
    return {'generated':tokenizer.decode(generated,skip_special_tokens=True),'generated_ids':generated,
        'decision_ids':decisions,'forward_count':calls,'stop_reason':reason}
