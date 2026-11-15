"""Answer-free Query donors and first-question-only native eager interventions."""
from contextlib import ExitStack
from dataclasses import dataclass
import torch
from sas_memory.runtime import fork_cache,cache_tensors
from sas_memory.training_data import data_fingerprint
from sas_memory.kv_readout import QuestionKVReadout
from scripts.copy_readout_runtime import SelectionWeights,readout
from scripts.post_encoding_runtime import (CacheGuard,assert_isolated,assert_complete_tensor_inventory,
    complete_metadata,historical_kv_proof,check_cached_attention,ATTENTION_LAYERS,GDN_LAYERS)
from scripts.fresh_layer_controls import tensor_hash
from scripts.run_write_locality_load_screen import hook_state


def tensor_record(value):
    return {'sha256':tensor_hash(value),'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device),
        'stride':list(value.stride()),'storage_offset':value.storage_offset(),'finite':bool(torch.isfinite(value).all())}


def clone_layout(value):
    capacity=value.storage_offset()+1+sum((n-1)*s for n,s in zip(value.shape,value.stride()))
    result=torch.empty(capacity,dtype=value.dtype,device=value.device).as_strided(value.shape,value.stride(),value.storage_offset())
    result.copy_(value)
    return result


class ArgumentGuard:
    """Guard actual argument contents, identities, layouts and nested bookkeeping."""
    def __init__(self,value):
        self.tensors=[];self.retain=True;self.tree=self.snapshot(value)
    def snapshot(self,value):
        if isinstance(value,torch.Tensor):
            if self.retain:self.tensors.append((value,value.clone(),id(value),value.data_ptr(),value.untyped_storage().data_ptr(),
                (tuple(value.shape),value.dtype,value.device,tuple(value.stride()),value.storage_offset())))
            return ('tensor',id(value))
        if isinstance(value,dict):return ('dict',id(value),tuple((k,self.snapshot(v)) for k,v in value.items()))
        if isinstance(value,(list,tuple)):return (type(value).__name__,id(value),tuple(self.snapshot(v) for v in value))
        if value is None or type(value) in (bool,int,float,str):return (type(value).__name__,value)
        raise ValueError('Untracked non-Query argument type')
    def check(self,value):
        scratch=ArgumentGuard.__new__(ArgumentGuard);scratch.tensors=[];scratch.retain=False
        if scratch.snapshot(value)!=self.tree:raise ValueError('Native argument identities or bookkeeping changed')
        for tensor,saved,ident,ptr,storage,layout in self.tensors:
            if (id(tensor)!=ident or tensor.data_ptr()!=ptr or tensor.untyped_storage().data_ptr()!=storage or
                (tuple(tensor.shape),tensor.dtype,tensor.device,tuple(tensor.stride()),tensor.storage_offset())!=layout or
                not torch.equal(tensor,saved)):
                raise ValueError('Native non-Query argument content/layout changed')
        return len(self.tensors)


@dataclass(frozen=True)
class ReaderContext:
    mode:str
    keep_bits:str
    prelude:int
    def __post_init__(self):
        if (self.mode not in ('full','oracle') or set(self.keep_bits)!=set('01') or
            not 0<self.prelude<len(self.keep_bits) or self.keep_bits[:self.prelude]!='1'*self.prelude):
            raise ValueError('Registered full/oracle reader context required')


class QueryBank:
    def __init__(self,queries,boundary,question_ids,prompt_sha,reader):
        if list(queries)!=list(ATTENTION_LAYERS):raise ValueError('All6actual Query tensors required')
        self.queries=queries;self.boundary=boundary;self.question_ids=list(question_ids);self.prompt_sha=prompt_sha;self.reader=reader
        if reader not in ('full','oracle'):raise ValueError('Reader provenance required')
        self.fixed_metadata=(boundary,tuple(question_ids),prompt_sha,reader)
        self.guard=ArgumentGuard(queries);self.checks=0
        for value in queries.values():
            if value.ndim!=4 or value.shape[0]!=1 or value.shape[-2]!=len(question_ids) or not torch.isfinite(value).all():
                raise ValueError('Finite actual question-only Query tensor required')
    def verify(self):
        if (self.boundary,tuple(self.question_ids),self.prompt_sha,self.reader)!=self.fixed_metadata:raise ValueError('Donor provenance changed')
        self.guard.check(self.queries)
    def check(self):self.verify();self.checks+=1
    def record(self):
        self.check()
        return {'boundary':self.boundary,'question_ids_sha256':data_fingerprint(self.question_ids),'question_tokens':len(self.question_ids),
            'prompt_ids_sha256':self.prompt_sha,'reader':self.reader,'guard_checks':self.checks,
            'tensors':{str(k):tensor_record(v) for k,v in self.queries.items()}}


class QuestionQueryExchange:
    def __init__(self,model,boundary,question_ids,mode,donor=None):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native
        if model.config._attn_implementation!='eager' or mode not in ('capture','replace') or (mode=='replace')!=(donor is not None):
            raise ValueError('Native eager capture or actual replacement required')
        self.model=model;self.native=native;self.original=native.eager_attention_forward
        self.boundary=boundary;self.question_ids=list(question_ids);self.mode=mode;self.donor=donor;self.model_calls=0;self.current=None
        if donor and (donor.boundary!=boundary or donor.question_ids!=self.question_ids):raise ValueError('Donor input boundary differs')
        self.queries={};self.stats={i:{'eager_calls':0,'question_calls':0,'question_rows':0,'passthrough_calls':0,
            'non_query_tensor_checks':0,'native_query_checks':0} for i in ATTENTION_LAYERS}
        self.handle=model.register_forward_pre_hook(self.before,with_kwargs=True);native.eager_attention_forward=self.wrapped
    def before(self,module,args,kwargs):
        ids=kwargs.get('input_ids');cache=kwargs.get('past_key_values');past=cache.get_seq_length() if cache is not None else 0
        if args or ids is None or ids.shape[0]!=1 or not past:raise ValueError('Only natural cached keyword forwards permitted')
        first=self.model_calls==0
        if first:
            if past!=self.boundary or ids.tolist()!=[self.question_ids]:raise ValueError('First call must be exactly the known question suffix')
        elif self.mode=='capture' or past<self.boundary+len(self.question_ids):raise ValueError('Capture cannot consume answer/decode inputs')
        self.model_calls+=1;self.current=(past,ids.shape[1],first)
    def wrapped(self,module,query,key,value,attention_mask,**kwargs):
        index=module.layer_idx
        if index not in self.stats or self.current is None:raise ValueError('Untracked native eager layer/call')
        past,rows,first=self.current;stat=self.stats[index]
        if query.ndim!=4 or query.shape[-2]!=rows or key.shape[-2]!=past+rows or value.shape[-2]!=past+rows:
            raise ValueError('Actual native Query/KV call boundary differs')
        args=(key,value,attention_mask,kwargs);guard=ArgumentGuard(args);qguard=ArgumentGuard(query)
        stat['eager_calls']+=1;used=query
        if first:
            if stat['question_calls']:raise ValueError('Duplicate question invocation in one layer')
            stat['question_calls']=1;stat['question_rows']=rows;stat['native_query']=tensor_record(query)
            if self.mode=='capture':self.queries[index]=clone_layout(query.detach())
            else:
                self.donor.verify();source=self.donor.queries[index]
                if source.shape!=query.shape or source.dtype!=query.dtype or source.device!=query.device:
                    raise ValueError('Query donor shape/dtype/device differs')
                used=clone_layout(query);used.copy_(source)
                if used.stride()!=query.stride() or used.storage_offset()!=query.storage_offset() or used.untyped_storage().data_ptr() in (
                    query.untyped_storage().data_ptr(),source.untyped_storage().data_ptr()):raise ValueError('Replacement must preserve native layout using isolated storage')
                stat['used_query']=tensor_record(used);stat['actual_replacement']=True
        else:stat['passthrough_calls']+=1
        usedguard=ArgumentGuard(used)
        result=self.original(module,used,key,value,attention_mask,**kwargs)
        stat['non_query_tensor_checks']+=guard.check(args);qguard.check(query);usedguard.check(used);stat['native_query_checks']+=1
        if self.donor:self.donor.verify()
        return result
    def report(self):
        if not self.model_calls or any(v['question_calls']!=1 or v['question_rows']!=len(self.question_ids) for v in self.stats.values()):
            raise ValueError('Exactly one complete question call per Attention layer required')
        if self.mode=='capture' and (self.model_calls!=1 or list(self.queries)!=list(ATTENTION_LAYERS)):
            raise ValueError('Capture must end before any answer computation')
        return {'mode':self.mode,'boundary':self.boundary,'question_ids_sha256':data_fingerprint(self.question_ids),
            'question_tokens':len(self.question_ids),'model_calls':self.model_calls,'layers':self.stats,
            'non_query_arguments_preserved':True,'decode_queries_unmodified':True}
    def close(self):
        self.native.eager_attention_forward=self.original;self.handle.remove()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def open_reader(stack,model,context):
    keep=[v=='1' for v in context.keep_bits]
    restricted=stack.enter_context(QuestionKVReadout(model,keep)) if context.mode=='oracle' else None
    observer=stack.enter_context(SelectionWeights(model,{'actual':keep},context.prelude))
    return restricted,observer


def reader_proof(context,restricted,observer):
    stats=observer.stats['actual'];check_cached_attention(stats,context.mode=='oracle')
    calls=restricted.calls if restricted else None
    if calls and (calls['prefill'] or not calls['cached']):raise ValueError('Only cached reader calls permitted')
    return {'reader_calls':calls,'attention':stats,'reader_keep_sha256':data_fingerprint(context.keep_bits)}


def capture_question(runtime,ledger,cache,question,reader_context):
    """No answer, role, support lookup or generated token is accepted."""
    suffix=runtime.query_suffix(question);ids=runtime.continuation_ids(ledger,suffix);boundary=cache.get_seq_length()
    if boundary!=len(reader_context.keep_bits) or len(ids)<2:raise ValueError('Fixed multi-token known question required')
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    active=hook_state(runtime.model);eager=native.eager_attention_forward
    with ExitStack() as stack:
        restricted,observer=open_reader(stack,runtime.model,reader_context)
        exchange=stack.enter_context(QuestionQueryExchange(runtime.model,boundary,ids,'capture'))
        _,_,count=runtime.advance(suffix,cache,ledger)
        proof=reader_proof(reader_context,restricted,observer);proof['query_exchange']=exchange.report()
        if count!=len(ids):raise ValueError('Capture question length changed')
        bank=QueryBank(exchange.queries,boundary,ids,data_fingerprint(runtime.ids(ledger+suffix)),reader_context.mode)
        proof['donor']=bank.record()
    if hook_state(runtime.model)!=active or native.eager_attention_forward!=eager:raise ValueError('Capture hooks/eager function not restored')
    return bank,proof


def measure_query(runtime,context,cache,query,reader,donor=None):
    rc=ReaderContext(reader,query['history_keep_bits'],query['prelude_tokens']);ids=query['input_ids'][query['ledger_tokens']:]
    if context['id']!=query['context_id'] or cache.get_seq_length()!=query['ledger_tokens']:raise ValueError('Registered cache/question required')
    if donor and (donor.reader!=reader or donor.prompt_sha!=query['prompt_ids_sha256']):raise ValueError('Reader-specific same-prompt donor required')
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    active=hook_state(runtime.model);eager=native.eager_attention_forward
    with ExitStack() as stack:
        restricted,observer=open_reader(stack,runtime.model,rc)
        exchange=stack.enter_context(QuestionQueryExchange(runtime.model,query['ledger_tokens'],ids,'replace',donor)) if donor else None
        value=readout(runtime.probe(context['ledger'],cache,query['question'],query['answer'],[query['answer']],generate=True,prefill='cached',score_batch=False))
        proof=reader_proof(rc,restricted,observer);proof['query_exchange']=exchange.report() if exchange else None
    if hook_state(runtime.model)!=active or native.eager_attention_forward!=eager or value['query_tokens']!=len(ids):raise ValueError('Reader hook/eager cleanup or question tokens changed')
    return value,proof


def restore_recurrent(prefix,normal):
    assert_complete_tensor_inventory(prefix);assert_complete_tensor_inventory(normal);assert_isolated(prefix,normal)
    before=dict(cache_tensors(prefix));original=dict(cache_tensors(normal));result=fork_cache(normal)
    if len(before)!=48 or len(original)!=48:raise ValueError('Complete actual48tensor sources required')
    restored=[]
    for key,source in before.items():
        if key[1]!='recurrent_states':continue
        if source.shape!=original[key].shape or source.dtype!=original[key].dtype:raise ValueError('Compatible recurrent source shape/dtype required')
        layer,field,slot=key;getattr(result.layers[layer],field)[slot]=source.clone();restored.append(str(key))
    if len(restored)!=18 or complete_metadata(result)!=complete_metadata(normal):raise ValueError('Only18recurrent matrices may be restored')
    for key,value in cache_tensors(result):
        if not torch.equal(value,(before if key[1]=='recurrent_states' else original)[key]):raise ValueError('Wrong restored component origin')
    assert_isolated(prefix,result);assert_isolated(normal,result)
    return result,{'restored_recurrent_keys':restored,'normal_non_recurrent_fields_exact':30,
        'historical_kv_sha256':historical_kv_proof(prefix,result,prefix.get_seq_length()),'normal_bookkeeping_exact':True}
