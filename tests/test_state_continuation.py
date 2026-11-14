"""Actual cache-origin cuts, original-runtime decode traces and corrupt raw-vector rejection."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
import torch

from sas_memory.runtime import TextRuntime,cache_tensors,fork_cache
from sas_memory.training_data import data_fingerprint
from scripts.analyze_state_continuation import audit_cache_records,audit_continuation
from scripts.run_state_continuation import observe_continuation
from scripts.run_write_locality_load_screen import hook_state
from scripts.state_continuation_controls import (CONDITIONS,TARGETS,ForwardTrace,cache_record,
    complete_metadata,verify_sources,make_condition,gold_losses,decode_vectors,input_scope,sources)


class Cache:
    def __init__(self):
        self.seq_len=5;self.layers=[];self.offloading=False
        for i in range(24):
            if i%4==3:
                layer=SimpleNamespace(keys=torch.zeros(1,2,5,2),values=torch.zeros(1,2,5,2),is_initialized=True)
            else:
                layer=SimpleNamespace(conv_states={0:torch.zeros(1,2,4)},recurrent_states={0:torch.zeros(1,2,2,2)},
                    is_conv_states_initialized={0:True},is_recurrent_states_initialized={0:True},
                    has_previous_state={0:True},conv_kernel_size={0:4},record_past=False)
            self.layers.append(layer)
    def get_seq_length(self):return self.seq_len


class Rotary(torch.nn.Module):
    def forward(self,x,position_ids):return x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Module();self.model.rotary_emb=Rotary()
        self.device=torch.device('cpu');self.generation_config=SimpleNamespace(eos_token_id=0)
    def forward(self,input_ids,past_key_values,use_cache,logits_to_keep):
        n=input_ids.shape[1];start=past_key_values.get_seq_length()
        positions=torch.arange(start,start+n).view(1,1,n).expand(3,1,n)
        self.model.rotary_emb(input_ids,positions)
        logits=torch.full((1,n,16),-4.,dtype=torch.float32)
        for i,t in enumerate(input_ids[0]):
            logits[0,i,2 if t==1 else 3]=4.
            logits[0,i,7]+=past_key_values.layers[21].recurrent_states[0].sum()
        past_key_values.seq_len+=n
        return SimpleNamespace(logits=logits[:,-logits_to_keep:],past_key_values=past_key_values)


def runtime_fixture():
    runtime=TextRuntime.__new__(TextRuntime);runtime.model=Model()
    runtime.tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:''.join({0:'',1:'1',2:'2',3:'\n',7:'7'}.get(i,'x') for i in ids))
    runtime.continuation_ids=lambda prompt,suffix:[int(c) for c in suffix]
    q={'id':'q','answer':'12','answer_ids':[1,2],'question_tokens':2,'prompt_tokens':5,'prompt_ids_sha256':'ids'}
    first=torch.full((1,16),-4.);first[0,1]=4.
    return runtime,q,first


class StateContinuationTests(unittest.TestCase):
    def origins(self):
        native=Cache();changed=fork_cache(native)
        for k,v in cache_tensors(changed):
            if k in TARGETS:v.add_(1)
        return {'unmasked':native,'both_paths_restored':changed}

    def test_all_five_actual_cache_origins_preserve_sources_and_complete_layout(self):
        origins=self.origins();check=verify_sources(*origins.values());check['id']='q';before={k:cache_record(v) for k,v in origins.items()}
        rows=[]
        for label in CONDITIONS:
            work,proof=make_condition(origins,label);r=cache_record(work)
            rows.append({'condition':label,'origin_proof':proof,**{k:r['sha256'] for k in
                ('starting_cache_sha256','after_gold_cache_sha256','after_greedy_cache_sha256')}})
            rows[-1].update(origin_cache_fingerprints={s:{k:v['sha256'] for k,v in before.items()} for s in
                ('before','after_gold','after_greedy')},origin_cache_metadata_unchanged_after_each_operation=True)
            for k,v in cache_tensors(work):
                expected=CONDITIONS[label][k[1]] if k in TARGETS else 'unmasked'
                self.assertTrue(torch.equal(v,dict(cache_tensors(origins[expected]))[k]))
                v.add_(9)
        self.assertEqual(before,{k:cache_record(v) for k,v in origins.items()})
        refs={key:{'full_question_cache_sha256':before[name]['sha256'],
            'recurrent_conv_sha256':{k:v['sha256'] for k,v in before[name]['tensors'].items() if 'states' in k}}
            for key,name in [(('unmasked','none'),'unmasked'),(('earlier_only_layer19','both'),'both_paths_restored')]}
        audit_cache_records(check,rows,refs,{'id':'q','prompt_tokens':5})
        broken=deepcopy(rows);broken[1]['origin_proof']['origins'][str((20,'conv_states',0))]['origin']='unmasked'
        with self.assertRaises(ValueError):audit_cache_records(check,broken,refs,{'id':'q','prompt_tokens':5})

    def test_full_KV_current_row_and_non_target_state_corruption_are_rejected(self):
        for field in ('last_kv','outside_state','bookkeeping'):
            origins=self.origins();cache=origins['both_paths_restored']
            if field=='last_kv':cache.layers[23].keys[...,-1,:]+=1
            elif field=='outside_state':cache.layers[18].conv_states[0]+=1
            else:cache.layers[20].record_past=True
            with self.assertRaises(ValueError):verify_sources(*origins.values())

    def test_metadata_covers_container_types_and_previously_unlisted_bookkeeping(self):
        cache=Cache();before=complete_metadata(cache);cache.layers[20].record_past=True
        self.assertNotEqual(before,complete_metadata(cache))
        cache=Cache();cache.layers[20].conv_states=list(cache.layers[20].conv_states.values())
        self.assertNotEqual(before,complete_metadata(cache))
        cache.callback=object()
        with self.assertRaises(ValueError):complete_metadata(cache)

    def continuation_fixture(self):
        runtime,q,first=runtime_fixture();vectors={'q--unmasked--none':first.clone()}
        origins=self.origins();cache=fork_cache(origins['unmasked']);before=cache_record(cache)
        observed=observe_continuation(runtime,q,'prompt',cache,first,'q--unmasked--none','q--native_reference',vectors,origins)
        self.assertEqual(cache_record(cache),before)
        entry={'model':'toy','id':'q','regime':'oracle','direction':'harm','expected_full':observed['readout']}
        row={'model':'toy','id':'q','condition':'native_reference','source_regime':'oracle','original_direction':'harm',
            'token_policy':'question','registration_sha256':data_fingerprint(q),'prompt_ids_sha256':q['prompt_ids_sha256'],
            'source_caches_unchanged_after_continuation':True,**observed}
        return runtime,q,row,entry,vectors

    def test_original_runtime_greedy_and_gold_are_reconstructed_from_actual_passive_vectors(self):
        runtime,q,row,entry,vectors=self.continuation_fixture()
        self.assertEqual(row['readout']['generated'],'12\n');self.assertTrue(row['readout']['greedy_exact'])
        self.assertEqual(row['decode']['decision_ids'],[1,2,3]);self.assertEqual(len(row['greedy_forwards']),2)
        self.assertEqual(len(row['gold_forwards']),1);self.assertEqual(hook_state(runtime.model),{})
        used=audit_continuation(row,q,entry,vectors,runtime.tokenizer,16,0,'cpu')
        self.assertEqual(used,set(vectors))

    def test_vector_corruption_wrong_losses_positions_and_generations_are_rejected(self):
        runtime,q,row,entry,vectors=self.continuation_fixture()
        for kind in ('vector','loss','position','generation','origin'):
            bad=deepcopy(row);v={k:x.clone() for k,x in vectors.items()}
            if kind=='vector':v[bad['gold_forwards'][0]['logits_key']][0,0,15]+=0.25
            elif kind=='loss':bad['gold_losses']['nll_sum']+=0.001
            elif kind=='position':bad['greedy_forwards'][0]['position_ids'][0][0][0]+=1
            elif kind=='generation':bad['readout']['generated']='17\n'
            else:bad['initial_logits_key']='wrong'
            with self.assertRaises(ValueError):audit_continuation(bad,q,entry,v,runtime.tokenizer,16,0,'cpu')

    def test_passive_hook_exception_cleanup_and_original_position_validation(self):
        runtime,_,_=runtime_fixture();hooks=hook_state(runtime.model)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with ForwardTrace(runtime.model):raise RuntimeError('injected')
        self.assertEqual(hook_state(runtime.model),hooks)

    def test_actual_origin_cache_mutation_during_gold_is_rejected_before_greedy(self):
        runtime,q,first=runtime_fixture();origins=self.origins();cache=fork_cache(origins['unmasked'])
        original=runtime.candidate_scores
        def mutate(*args,**kwargs):
            result=original(*args,**kwargs);origins['both_paths_restored'].layers[20].conv_states[0].add_(1)
            return result
        runtime.candidate_scores=mutate
        with self.assertRaisesRegex(ValueError,'reused origin'):
            observe_continuation(runtime,q,'prompt',cache,first,'first','test',{},origins)
        self.assertEqual(hook_state(runtime.model),{})

    def test_eos_newline_and_max_length_include_original_unused_final_forward(self):
        runtime,_,first=runtime_fixture();eos=torch.full_like(first,-4);eos[0,0]=9
        self.assertEqual(decode_vectors(runtime.tokenizer,eos,[],0)['stop_reason'],'eos')
        vector=first[:,None].clone()
        decoded=decode_vectors(runtime.tokenizer,first,[vector]*12,0)
        self.assertEqual(decoded['stop_reason'],'max_tokens');self.assertEqual(decoded['forward_count'],12)
        with self.assertRaises(ValueError):decode_vectors(runtime.tokenizer,first,[vector]*11,0)

    def test_gold_scoring_uses_original_float32_addition_and_complete_forward_shape(self):
        _,_,first=runtime_fixture();cont=first[:,None].clone();loss=gold_losses(first,cont,[1,2],'cpu')
        a=-first.log_softmax(-1)[0,torch.tensor([1])]
        a+=-cont.log_softmax(-1).gather(-1,torch.tensor([[[2]]])).squeeze(-1).sum(-1)
        self.assertEqual(loss['nll_sum'],a.item())
        with self.assertRaises(ValueError):gold_losses(first,cont[:,:,:-1],[1,2],'cpu')

    def test_registered_inputs_pin_all32_actual_source_observations_and129_sources(self):
        scope=input_scope();self.assertEqual(len(scope['plan']['source_references']),32)
        self.assertEqual(len(sources()),129);self.assertEqual(scope['plan']['physical_cached_continuation_readouts'],20)


if __name__=='__main__':unittest.main()
