"""Read-only native softmax pooling; executable labels are used after generation."""
import math
import re
import torch
import numpy as np
from sas_memory.training_data import data_fingerprint
from scripts.fresh_layer_controls import tensor_hash
from scripts.question_state_freeze_runtime import canonical,full_attention_contract
from scripts.copy_readout_runtime import SelectionWeights,readout
from scripts.run_write_locality_load_screen import hook_state
from scripts.version_address_data import LAYERS,POSITIONS


class RecordObservation:
    """No role,truth,support or answer is accepted by this observation interface."""
    def __init__(self,model,records,prelude,boundary,known_ids):
        if (len(known_ids)!=24 or type(boundary) is not int or boundary<=prelude or
            not records or any(b['index']!=i or b['start']!=prelude+14*i or b['stop']!=prelude+14*(i+1)
                for i,b in enumerate(records)) or records[-1]['stop']!=boundary):
            raise ValueError('Complete contiguous known history and24token question required')
        self.model=model;self.records=records;self.prelude=prelude;self.boundary=boundary;self.ids=known_ids
        self.before_hooks=hook_state(model);self.handles=[];self.values={};self.aux={};self.calls=0;self.closed=False
        self.handles.append(model.register_forward_pre_hook(self.before,with_kwargs=True))
        for index in LAYERS:self.handles.append(model.model.layers[index].self_attn.register_forward_hook(self.after(index)))
        self.handles.append(model.register_forward_hook(self.done,with_kwargs=True,always_call=True))

    def before(self,module,args,kwargs):
        cache=kwargs.get('past_key_values');actual=kwargs.get('input_ids')
        if (args or self.calls or cache is None or cache.get_seq_length()!=self.boundary or
            not isinstance(actual,torch.Tensor) or actual.tolist()!=[self.ids]):
            raise ValueError('Observer may see only the known24token question on its normal history')
        self.calls+=1

    def after(self,index):
        def observe(module,args,result):
            weights=result[1]
            if (index in self.values or not isinstance(weights,torch.Tensor) or
                tuple(weights.shape)!=(1,8,24,self.boundary+24)):
                raise ValueError('Actual native8head cached softmax required')
            # Neither the output tuple nor the native softmax tensor is modified or replaced.
            selected=weights[0,:,list(POSITIONS),:].detach().float()
            if not torch.isfinite(selected).all() or (selected<0).any() or (selected>1).any():
                raise ValueError('Invalid native probabilities')
            for j,pos in enumerate(POSITIONS):
                if torch.count_nonzero(selected[:,j,self.boundary+pos+1:]).item():
                    raise ValueError('Native softmax unexpectedly accesses future question tokens')
            pooled=selected.mean(dim=0)
            mass=pooled[:,self.prelude:self.boundary].reshape(2,len(self.records),14).sum(-1)
            # Arrays are exact FP32 observer outputs,not normalized or quantized for storage.
            self.values[index]=mass.cpu().numpy().copy()
            self.aux[index]={'prelude_mass':pooled[:,:self.prelude].sum(-1).cpu().tolist(),
                'question_mass':pooled[:,self.boundary:].sum(-1).cpu().tolist(),
                'total_mass':pooled.sum(-1).cpu().tolist(),'native_weights_dtype':str(weights.dtype),
                'native_weights_shape':list(weights.shape),'pooled_record_mass_sha256':tensor_hash(mass)}
        return observe

    def done(self,module,args,kwargs,result):
        if result is None:
            self.close();return
        try:
            if set(self.values)!=set(LAYERS) or self.calls!=1:raise ValueError('All six native layers required')
            if result.past_key_values.get_seq_length()!=self.boundary+24:raise ValueError('Known suffix boundary changed')
        finally:self.close()

    def close(self):
        if self.closed:return
        for handle in self.handles:handle.remove()
        self.closed=True

    def report(self):
        if (not self.closed or self.calls!=1 or set(self.values)!=set(LAYERS) or
            hook_state(self.model)!=self.before_hooks):raise ValueError('Observer must finish and remove all own hooks before answers')
        array=np.stack([self.values[i] for i in LAYERS])
        if array.dtype!=np.float32 or array.shape!=(6,2,len(self.records)):raise ValueError('Complete raw pooled array required')
        return array,canonical({'question_calls':1,'question_tokens':24,'known_ids_sha256':data_fingerprint(self.ids),
            'layers':self.aux,'positions':list(POSITIONS),'all_own_hooks_removed_before_answers':True,
            'native_attention_return_not_replaced':True,'no_roles_answers_or_support_in_capture_interface':True})

    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def measure_observed(runtime,context,cache,query):
    ids=runtime.continuation_ids(context['ledger'],runtime.query_suffix(query['question']))
    if ids!=query['input_ids'][context['ledger_tokens']:]:raise ValueError('Literal known question changed')
    keep=[v=='1' for v in query['history_keep_bits']];active=hook_state(runtime.model)
    # This legacy mask is used only by the already sealed passive observer,not a ReaderGate.
    with SelectionWeights(runtime.model,{'full':keep},query['prelude_tokens']) as old:
        with RecordObservation(runtime.model,context['records'],context['prelude_tokens'],context['ledger_tokens'],ids) as observer:
            value=readout(runtime.probe(context['ledger'],cache,query['question'],query['answer'],[query['answer']],
                generate=True,prefill='cached',score_batch=False))
            array,capture=observer.report()
    if hook_state(runtime.model)!=active or value['query_tokens']!=24:raise ValueError('Normal observer cleanup or boundary changed')
    full_attention_contract(old.stats['full'])
    return value,canonical({'reader_calls':None,'attention':old.stats['full'],
        'reader_keep_sha256':data_fingerprint(query['history_keep_bits'])}),array,capture


def summarize_record_array(array,capture,query,context,case,generated):
    """Called after complete generation; reconstruct labels and every statistic from raw observations."""
    array=np.asarray(array)
    count=len(context['records'])
    if array.dtype!=np.float32 or array.shape!=(6,2,count) or not np.isfinite(array).all() or (array<0).any():
        raise ValueError('Exact finite FP32 raw record array required')
    events=case['prefix_events']+next(e['tail'] for e in case['episodes'] if e['id']==context['episode'])
    if len(events)!=count or any(e['kind']!='SET' for e in events):raise ValueError('Executable SET records required')
    values=[e['value'] for e in events]
    if len(set(values))!=len(values):raise ValueError('This diagnostic requires unique within-history numeric values')
    entity=case['protected'] if query['role']=='protected_current' else case['target']
    own=[i for i,e in enumerate(events) if e['target']==entity]
    support=own[0 if query['role']=='target_initial' else -1]
    if support!=query['support_event_index'] or str(events[support]['value'])!=query['answer']:
        raise ValueError('Independent entity/version truth differs')
    match=re.fullmatch(r'\s*(-?\d+)\s*',generated)
    emitted=values.index(int(match[1])) if match and int(match[1]) in values else None
    category=('correct' if match and match[1]==query['answer'] else 'non_integer' if not match else
        'noncanonical_correct_integer' if int(match[1])==int(query['answer']) else
        'same_account_other_version_value' if emitted in own else 'other_account_recorded_value' if emitted is not None else 'unrecorded_integer')
    rows=[]
    for li,layer in enumerate(LAYERS):
        aux=capture['layers'][str(layer)]
        for pi,pos in enumerate(POSITIONS):
            mass=array[li,pi];total=float(mass.sum(dtype=np.float32));pre=float(aux['prelude_mass'][pi]);question=float(aux['question_mass'][pi]);allmass=float(aux['total_mass'][pi])
            if any(not math.isfinite(x) or x<0 for x in (total,pre,question,allmass)):
                raise ValueError('Invalid raw/native pooled mass')
            rank=lambda i:1+int(np.count_nonzero(mass>mass[i]))
            top=sorted(range(count),key=lambda i:(-float(mass[i]),i))[:3]
            wrong=[i for i in own if i!=support]
            rows.append({'layer':layer,'suffix_position':pos,'record_count':count,'entity':entity,
                'entity_record_indices':own,'support_record_index':support,'support_mass':float(mass[support]),
                'support_rank_competition':rank(support),'support_equal_mass_ties':int(np.count_nonzero(mass==mass[support])),
                'entity_version_masses':[float(mass[i]) for i in own],'entity_mass':float(mass[own].sum(dtype=np.float32)),
                'other_version_mass':float(mass[wrong].sum(dtype=np.float32)) if wrong else 0.,
                'top_record_indices':top,'top_record_masses':[float(mass[i]) for i in top],
                'emitted_record_index':emitted,'emitted_record_mass':None if emitted is None else float(mass[emitted]),
                'emitted_record_rank_competition':None if emitted is None else rank(emitted),
                'record_mass':total,'prelude_mass':pre,'question_mass':question,'total_mass':allmass,
                'fp32_partition_rounding_residual':total+pre+question-allmass,
                'output_category':category,'raw_capture_sha256':aux['pooled_record_mass_sha256']})
    return rows
