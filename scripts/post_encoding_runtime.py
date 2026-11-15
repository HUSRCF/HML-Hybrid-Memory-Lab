"""Tail-only writes and cached readers with complete parent-cache guards."""
from contextlib import ExitStack
import math
from sas_memory.runtime import fork_cache,cache_tensors,assert_cache_equal
from sas_memory.training_data import data_fingerprint
from scripts.state_continuation_controls import complete_metadata,cache_record
from scripts.fresh_layer_controls import tensor_hash
from scripts.gate_component_runtime import ComponentHold,GDN_LAYERS
from scripts.run_write_locality_load_screen import hook_state
from scripts.copy_readout_runtime import SelectionWeights,readout
from sas_memory.kv_readout import QuestionKVReadout

ATTENTION_LAYERS=(3,7,11,15,19,23)


def tensor_layouts(cache):
    return {str(k):{'stride':list(t.stride()),'storage_offset':t.storage_offset()}
        for k,t in cache_tensors(cache)}


def storage_ids(cache):
    return {t.untyped_storage().data_ptr() for _,t in cache_tensors(cache) if t.numel()}


def assert_complete_tensor_inventory(cache):
    import torch
    found=[]
    def visit(value):
        if isinstance(value,torch.Tensor):found.append(id(value))
        elif isinstance(value,dict):
            for v in value.values():visit(v)
        elif isinstance(value,(tuple,list)):
            for v in value:visit(v)
    visit({k:v for k,v in vars(cache).items() if k!='layers'})
    for layer in cache.layers:visit(vars(layer))
    declared=[id(t) for _,t in cache_tensors(cache)]
    if sorted(found)!=sorted(declared):raise ValueError('Untracked tensor in actual cache bookkeeping')


def assert_isolated(parent,child):
    assert_complete_tensor_inventory(parent);assert_complete_tensor_inventory(child)
    if storage_ids(parent)&storage_ids(child):raise ValueError('Parent and child share mutable tensor storage')


def exact_cache(left,right):
    assert_cache_equal(left,right)
    if complete_metadata(left)!=complete_metadata(right):raise ValueError('Complete cache bookkeeping changed')


class CacheGuard:
    """Snapshot all real cache attributes; pointer identities stay local only."""
    def __init__(self,cache,logits=None):
        self.cache=cache;self.saved=fork_cache(cache);self.metadata=complete_metadata(cache)
        self.layouts=tensor_layouts(cache);self.pointers=storage_ids(cache)
        self.logits=logits;self.saved_logits=None if logits is None else logits.clone();self.checks=0
        assert_isolated(cache,self.saved)

    def check(self):
        import torch
        assert_complete_tensor_inventory(self.cache)
        assert_cache_equal(self.cache,self.saved)
        if (complete_metadata(self.cache)!=self.metadata or tensor_layouts(self.cache)!=self.layouts or
            storage_ids(self.cache)!=self.pointers or
            (self.logits is not None and not torch.equal(self.logits,self.saved_logits))):
            raise ValueError('A child call modified its complete source cache or logits')
        self.checks+=1

    def record(self):
        self.check()
        result=cache_record(self.cache)
        result.update(layouts=self.layouts,source_guard_checks=self.checks,
            logits_sha256=None if self.logits is None else tensor_hash(self.logits))
        return result


def check_gate_stats(report,mode,length):
    if (mode not in ('none','decay','delta','both') or length<2 or
        report['component_mode']!=mode or report['enabled']!=(mode!='none') or
        not report['actual_gates_finite_and_exact'] or set(map(int,report['layers']))!=set(GDN_LAYERS)):
        raise ValueError('Every actual tail gate in all18GDN layers must be observed')
    held=length if mode!='none' else 0
    for s in report['layers'].values():
        if s!={'chunk_calls':1,'recurrent_calls':0,'rows':length,'held_rows':held,'unheld_rows':length-held,
            'forced_decay_rows':length if mode in ('decay','both') else 0,
            'forced_delta_rows':length if mode in ('delta','both') else 0}:
            raise ValueError('Intervention escaped the single complete tail chunk')


def historical_kv_proof(prefix,tail,prefix_tokens):
    import torch
    left,right=dict(cache_tensors(prefix)),dict(cache_tensors(tail));result={}
    for key,value in left.items():
        if key[1] not in ('keys','values'):continue
        actual=right[key][...,:prefix_tokens,:]
        if value.shape[-2]!=prefix_tokens or not torch.equal(value,actual):raise ValueError('Normally encoded prefix KV changed')
        result[str(key)]=tensor_hash(actual)
    if len(result)!=12:raise ValueError('All12historical KV slices required')
    return result


def advance_tail(runtime,prefix_cache,prefix_text,tail_text,prefix_tokens,mode):
    """No query, answer, role or support is accepted by the write interface."""
    import torch
    if prefix_cache.get_seq_length()!=prefix_tokens:raise ValueError('Actual prefix boundary changed')
    count=len(runtime.continuation_ids(prefix_text,tail_text))
    if count<2:raise ValueError('Registered nonempty multi-token tail required')
    keep=[True]*prefix_tokens+[False]*count;active=hook_state(runtime.model)
    kernels=[(runtime.model.model.layers[i].linear_attn,name,
        getattr(runtime.model.model.layers[i].linear_attn,name)) for i in GDN_LAYERS
        for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule')]
    with torch.inference_mode(),ComponentHold(runtime.model,keep,mode) as intervention:
        cache,logits,actual=runtime.advance(tail_text,prefix_cache,prefix_text);gates=intervention.report()
    if (hook_state(runtime.model)!=active or any(getattr(gdn,name)!=original for gdn,name,original in kernels) or
        actual!=count or cache.get_seq_length()!=prefix_tokens+count):
        raise ValueError('Tail boundary or gate cleanup failed')
    check_gate_stats(gates,mode,count);assert_isolated(prefix_cache,cache)
    kv=historical_kv_proof(prefix_cache,cache,prefix_tokens)
    recurrent_exact=[]
    if mode=='both':
        a,b=dict(cache_tensors(prefix_cache)),dict(cache_tensors(cache))
        for key in a:
            if key[1]=='recurrent_states':
                if not torch.equal(a[key],b[key]):raise ValueError('Both-held tail modified a recurrent matrix')
                recurrent_exact.append(str(key))
        if len(recurrent_exact)!=18:raise ValueError('All18recurrent matrices required')
    return cache,logits,{'gates':gates,'tail_tokens':count,'write_keep_sha256':data_fingerprint('1'*prefix_tokens+'0'*count),
        'historical_kv_sha256':kv,'both_mode_exact_recurrent_keys':recurrent_exact,'isolated_from_prefix':True}


def check_cached_attention(stats,restricted):
    if set(map(int,stats))!=set(ATTENTION_LAYERS):raise ValueError('All6actual Attention layers required')
    for s in stats.values():
        if (s['prefill_calls']!=0 or s['cached_calls']<1 or s['query_rows']<1 or
            any(not math.isfinite(s[k]) for k in ('max_suppressed_probability','max_suppressed_mass','max_kept_record_mass')) or
            s['max_kept_record_mass']<=0 or
            (restricted and (s['max_suppressed_probability']!=0 or s['max_suppressed_mass']!=0)) or
            (not restricted and (s['max_suppressed_probability']<=0 or s['max_suppressed_mass']<=0))):
            raise ValueError('Cached-only kept/suppressed probability control failed')


def measure_cached(runtime,context,cache,query,reader):
    if reader not in ('full','oracle') or context['id']!=query['context_id'] or cache.get_seq_length()!=query['ledger_tokens']:
        raise ValueError('Fixed registered cache/query/reader required')
    keep=[b=='1' for b in query['history_keep_bits']];active=hook_state(runtime.model)
    with ExitStack() as stack:
        restriction=stack.enter_context(QuestionKVReadout(runtime.model,keep)) if reader=='oracle' else None
        observer=stack.enter_context(SelectionWeights(runtime.model,{'actual':keep},query['prelude_tokens']))
        value=readout(runtime.probe(context['ledger'],cache,query['question'],query['answer'],[query['answer']],
            generate=True,prefill='cached',score_batch=False))
        stats=observer.stats['actual'];check_cached_attention(stats,reader=='oracle')
        calls=restriction.calls if restriction else None
    if hook_state(runtime.model)!=active or (calls and (calls['prefill']!=0 or calls['cached']<1)):
        raise ValueError('Cached reader cleanup or execution partition failed')
    if value['query_tokens']!=len(query['input_ids'])-query['ledger_tokens']:raise ValueError('Question input changed')
    return value,{'reader_calls':calls,'attention':stats,'reader_keep_sha256':data_fingerprint(query['history_keep_bits'])}
