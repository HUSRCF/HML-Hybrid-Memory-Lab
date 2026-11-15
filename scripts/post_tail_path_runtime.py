"""Compose recurrent/conv/KV caches after tail processing, before any query."""
from sas_memory.runtime import fork_cache,cache_tensors
from scripts.post_encoding_runtime import (assert_complete_tensor_inventory,assert_isolated,complete_metadata,
    CacheGuard,historical_kv_proof,GDN_LAYERS)
from sas_memory.training_data import data_fingerprint

CONDITIONS=('NNN','NNH','NHN','NHH','HNN','HNH','HHN','HHH')
COMPONENTS=('recurrent_states','conv_states','kv')
ATTENTION_LAYERS=(3,7,11,15,19,23)


def component(key):
    if key[1] in ('recurrent_states','conv_states'):return COMPONENTS.index(key[1])
    if key[1] in ('keys','values'):return 2
    raise ValueError('Undeclared cache component')


def check_sources(sources):
    if set(sources)!=set('NH'):raise ValueError('Both original cache sources required')
    for cache in sources.values():assert_complete_tensor_inventory(cache)
    keys={(i,f,0) for i in GDN_LAYERS for f in COMPONENTS[:2]}|{(i,f,0) for i in ATTENTION_LAYERS for f in ('keys','values')}
    tensors={name:dict(cache_tensors(cache)) for name,cache in sources.items()}
    if (any(set(v)!=keys for v in tensors.values()) or
        complete_metadata(sources['N'])!=complete_metadata(sources['H'])):
        raise ValueError('All48tensor fields and complete N/H bookkeeping must agree')
    assert_isolated(sources['N'],sources['H'])
    return tensors


def compose_cache(sources,label):
    """Only cache sources and the origin label; no question/answer interface."""
    import torch
    if label not in CONDITIONS:raise ValueError('Registered three-component origin label required')
    tensors=check_sources(sources);result=fork_cache(sources['H'] if label=='HHH' else sources['N'])
    if label not in ('NNN','HHH'):
        for key in tensors['N']:
            if label[component(key)]!='H':continue
            index,field,slot=key;value=tensors['H'][key].clone()
            if field in ('keys','values'):setattr(result.layers[index],field,value)
            else:getattr(result.layers[index],field)[slot]=value
    actual=dict(cache_tensors(result))
    if complete_metadata(result)!=complete_metadata(sources['N']):raise ValueError('Composition changed cache bookkeeping')
    for key,value in actual.items():
        if not torch.equal(value,tensors[label[component(key)]][key]):raise ValueError('Wrong component origin during composition')
    for parent in sources.values():assert_isolated(parent,result)
    return result


def compare_components(sources,prefix_tokens):
    import torch
    tensors=check_sources(sources);result={name:{'equal_keys':[],'different_keys':[]} for name in COMPONENTS}
    for key,left in tensors['N'].items():
        right=tensors['H'][key];name=COMPONENTS[component(key)]
        result[name]['equal_keys' if torch.equal(left,right) else 'different_keys'].append(str(key))
        if key[1] in ('keys','values') and not torch.equal(left[...,:prefix_tokens,:],right[...,:prefix_tokens,:]):
            raise ValueError('N/H historical prefix KV differs')
    return result


def memory_identity(record):
    return data_fingerprint({k:record[k] for k in ('tensors','metadata','layouts')})


def expected_tensors(source_records,label):
    # Tensor keys have a sealed (layer,field,slot) representation; no eval is used.
    import ast
    return {key:source_records[label[component(ast.literal_eval(key))]]['tensors'][key]
        for key in source_records['N']['tensors']}
