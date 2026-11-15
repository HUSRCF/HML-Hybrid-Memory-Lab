"""CPU controls for answer-free donors, true Query swaps and fresh accounting."""
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch
import torch
from sas_memory.runtime import fork_cache,cache_tensors
from sas_memory.training_data import data_fingerprint
from scripts.fresh_query_runtime import (ArgumentGuard,QueryBank,QuestionQueryExchange,ReaderContext,
    capture_question,measure_query,restore_recurrent,clone_layout)
from scripts.fresh_query_data import read_plan,sources,build_registry,selected_contexts,context_queries,MODELS,READERS,CONDITIONS
from scripts.fresh_query_metrics import summarize
from scripts.analyze_fresh_query import audit_exchange,audit_restoration
from scripts.post_encoding_runtime import CacheGuard,ATTENTION_LAYERS
from tests.test_post_encoding import ToyCache,CharRuntime,prior_index


class TinyAttention(torch.nn.Module):
    def __init__(self,index):
        super().__init__();self.layer_idx=index;self.num_key_value_groups=1
    def forward(self,hidden,cache,mask):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native
        query=(hidden*.01+(self.layer_idx+1)*.0001).unsqueeze(1)
        layer=cache.layers[self.layer_idx]
        layer.keys=torch.cat([layer.keys,hidden.unsqueeze(1)],-2)
        layer.values=torch.cat([layer.values,(hidden+.1).unsqueeze(1)],-2)
        return native.eager_attention_forward(self,query,layer.keys,layer.values,mask,scaling=.5,dropout=0.)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(_attn_implementation='eager',layer_types=(['linear_attention']*3+['full_attention'])*6)
        self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList()
        self.embedding=torch.nn.Embedding(256,2)
        with torch.no_grad():self.embedding.weight.copy_(torch.arange(512).reshape(256,2)/512)
        for i in range(24):
            layer=torch.nn.Module()
            if i in ATTENTION_LAYERS:layer.self_attn=TinyAttention(i)
            self.model.layers.append(layer)
        self.eval().requires_grad_(False)
    def get_input_embeddings(self):return self.embedding
    def forward(self,*,input_ids,past_key_values,attention_mask=None):
        cache=past_key_values;past=cache.get_seq_length();rows=input_ids.shape[1]
        hidden=self.embedding(input_ids)+cache.layers[0].recurrent_states[0].mean()*.01
        mask=attention_mask['full_attention'] if attention_mask is not None else None
        if mask is None:
            allowed=torch.arange(past+rows)[None,:]<=past+torch.arange(rows)[:,None]
            mask=torch.zeros(1,1,rows,past+rows).masked_fill_(~allowed[None,None],torch.finfo(torch.float32).min)
        for i in ATTENTION_LAYERS:
            output,_=self.model.layers[i].self_attn(hidden,cache,mask);hidden=hidden+output.reshape_as(hidden)*.01
        cache.length+=rows
        return cache,hidden[:,-1]


class TinyRuntime(CharRuntime):
    def __init__(self):self.model=TinyModel()
    def query_suffix(self,question):return 'Q:'+question+'?'
    @torch.inference_mode()
    def advance(self,text,cache,prefix_text):
        work=fork_cache(cache);ids=self.continuation_ids(prefix_text,text)
        result,logits=self.model(input_ids=torch.tensor([ids]),past_key_values=work)
        return result,logits,len(ids)


class QueryRuntimeTests(unittest.TestCase):
    def setUp(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native
        self.native=native;self.original=native.eager_attention_forward
    def tearDown(self):self.assertIs(self.native.eager_attention_forward,self.original)

    def test_argument_guard_catches_mutations_identity_and_layout(self):
        for change in ('content','identity','bookkeeping','container'):
            value={'key':torch.arange(4.),'extra':[1]};guard=ArgumentGuard(value)
            if change=='content':value['key'].add_(1)
            elif change=='identity':value['key']=value['key'].clone()
            elif change=='bookkeeping':value['extra'].append(2)
            else:value['extra']=[1]
            with self.assertRaises(ValueError,msg=change):guard.check(value)

    def test_query_copy_preserves_actual_singleton_strides_and_offset(self):
        source=torch.arange(40.).as_strided((1,2,3,2),(37,9,2,1),1)
        result=clone_layout(source)
        self.assertEqual(result.stride(),source.stride());self.assertEqual(result.storage_offset(),1)
        self.assertNotEqual(result.untyped_storage().data_ptr(),source.untyped_storage().data_ptr())
        torch.testing.assert_close(source,result,rtol=0,atol=0)

    def test_capture_accepts_only_known_text_and_one_cached_question_forward(self):
        self.assertEqual(list(inspect.signature(capture_question).parameters),['runtime','ledger','cache','question','reader_context'])
        runtime=TinyRuntime();source=ToyCache();guard=CacheGuard(source)
        bank,proof=capture_question(runtime,'abc',source,'x',ReaderContext('full','110',1));guard.check()
        self.assertEqual(proof['query_exchange']['model_calls'],1);self.assertEqual(bank.question_ids,runtime.ids('Q:x?'))
        self.assertEqual(list(bank.queries),list(ATTENTION_LAYERS));self.assertEqual(bank.reader,'full')
        q={'ledger_tokens':3,'input_ids':runtime.ids('abcQ:x?')}
        audit_exchange(json.loads(json.dumps(proof['query_exchange'])),q,bank.record(),'capture',json.loads(json.dumps(proof)))

    def test_actual_same_source_replacement_matches_output_and_leaves_decode_queries(self):
        runtime=TinyRuntime();source=ToyCache();bank,_=capture_question(runtime,'abc',source,'x',ReaderContext('full','110',1))
        ids=runtime.ids('Q:x?');natural,natural_logits,_=runtime.advance('Q:x?',source,'abc')
        with QuestionQueryExchange(runtime.model,3,ids,'replace',bank) as exchange:
            actual,logits,_=runtime.advance('Q:x?',source,'abc')
            runtime.model(input_ids=torch.tensor([[65]]),past_key_values=fork_cache(actual))
            proof=exchange.report()
        torch.testing.assert_close(logits,natural_logits,rtol=0,atol=0)
        for a,b in zip(cache_tensors(actual),cache_tensors(natural)):torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
        for i,s in proof['layers'].items():
            self.assertEqual((s['question_calls'],s['question_rows'],s['passthrough_calls']),(1,4,1))
            self.assertTrue(s['actual_replacement']);self.assertEqual(s['used_query'],bank.record()['tensors'][str(i)])

    def test_cross_query_replacement_uses_donor_and_guarded_reader_provenance(self):
        runtime=TinyRuntime();normal=ToyCache();other=fork_cache(normal);other.layers[0].recurrent_states[0].add_(100)
        bank,_=capture_question(runtime,'abc',normal,'x',ReaderContext('full','110',1))
        with QuestionQueryExchange(runtime.model,3,bank.question_ids,'replace',bank) as exchange:
            runtime.advance('Q:x?',other,'abc');proof=exchange.report()
        self.assertNotEqual(proof['layers'][3]['native_query']['sha256'],bank.record()['tensors']['3']['sha256'])
        self.assertEqual(proof['layers'][3]['used_query'],bank.record()['tensors']['3'])
        bank.reader='oracle'
        with self.assertRaisesRegex(ValueError,'provenance'):bank.verify()

    def test_capture_rejects_answer_call_wrong_question_and_cleans_up_on_failure(self):
        runtime=TinyRuntime();source=ToyCache();ids=runtime.ids('Q:x?')
        for kind in ('answer','question','prefill'):
            with self.assertRaises(ValueError,msg=kind):
                with QuestionQueryExchange(runtime.model,3,ids,'capture'):
                    if kind=='question':runtime.advance('Q:y?',source,'abc')
                    elif kind=='prefill':runtime.model(input_ids=torch.tensor([ids]),past_key_values=ToyCache(0))
                    else:
                        actual,_,_=runtime.advance('Q:x?',source,'abc')
                        runtime.model(input_ids=torch.tensor([[65]]),past_key_values=actual)
            self.assertFalse(runtime.model._forward_pre_hooks)

    def test_mutating_native_non_query_tensor_is_rejected_and_wrappers_restore(self):
        runtime=TinyRuntime();real=self.original
        def corrupt(module,query,key,value,mask,**kwargs):
            result=real(module,query,key,value,mask,**kwargs);key.add_(1);return result
        self.native.eager_attention_forward=corrupt
        try:
            with self.assertRaisesRegex(ValueError,'non-Query'):
                capture_question(runtime,'abc',ToyCache(),'x',ReaderContext('full','110',1))
            self.assertIs(self.native.eager_attention_forward,corrupt);self.assertFalse(runtime.model._forward_pre_hooks)
        finally:self.native.eager_attention_forward=real

    def test_donor_mutation_or_missing_layer_cannot_be_used(self):
        runtime=TinyRuntime();source=ToyCache()
        for change in ('tensor','metadata','layer'):
            bank,_=capture_question(runtime,'abc',source,'x',ReaderContext('oracle','110',1))
            if change=='tensor':
                with torch.inference_mode():bank.queries[3].add_(1)
            elif change=='metadata':bank.question_ids.append(1)
            else:del bank.queries[23]
            with self.assertRaises(ValueError,msg=change):bank.verify()

    def test_oracle_donor_cannot_enter_full_reader(self):
        runtime=TinyRuntime();source=ToyCache();bank,_=capture_question(runtime,'abc',source,'x',ReaderContext('oracle','110',1))
        q={'id':'q','context_id':'ctx','input_ids':runtime.ids('abcQ:x?'),'ledger_tokens':3,
            'history_keep_bits':'110','prelude_tokens':1,'prompt_ids_sha256':data_fingerprint(runtime.ids('abcQ:x?'))}
        with self.assertRaisesRegex(ValueError,'Reader-specific'):measure_query(runtime,{'id':'ctx'},source,q,'full',bank)

    def test_restoration_preserves_all_non_recurrent_fields_and_refuses_borrowed_logits(self):
        prefix=ToyCache();normal=fork_cache(prefix);normal.length=5
        for key,tensor in list(cache_tensors(normal)):
            if key[1] in ('keys','values'):
                setattr(normal.layers[key[0]],key[1],torch.cat([tensor,torch.zeros_like(tensor[...,:2,:])],-2))
            else:tensor.add_(1)
        pg=CacheGuard(prefix,torch.ones(1,3));ng=CacheGuard(normal,torch.ones(1,3));held,proof=restore_recurrent(prefix,normal)
        pg.check();ng.check();pr=pg.record();nr=ng.record();hr=CacheGuard(held).record();audit_restoration(hr,nr,pr,5,1)
        self.assertEqual(len(proof['restored_recurrent_keys']),18)
        for change in ('logits','conv','metadata'):
            bad=copy.deepcopy(hr)
            if change=='logits':bad['logits_sha256']='0'*64
            elif change=='conv':bad['tensors'][str((0,'conv_states',0))]=pr['tensors'][str((0,'conv_states',0))]
            else:bad['metadata']['original_metadata']['sequence_length']=3
            with self.assertRaises(ValueError):audit_restoration(bad,nr,pr,5,1)

    def test_auditor_rejects_patch_in_decode_and_wrong_actual_donor(self):
        runtime=TinyRuntime();bank,proof=capture_question(runtime,'abc',ToyCache(),'x',ReaderContext('full','110',1))
        q={'ledger_tokens':3,'input_ids':runtime.ids('abcQ:x?')};record=bank.record();proof=json.loads(json.dumps(proof))
        for change in ('calls','boundary','donor','nonquery','finite'):
            bad=copy.deepcopy(proof['query_exchange'])
            if change=='calls':bad['model_calls']=2
            elif change=='boundary':bad['boundary']=2
            elif change=='donor':bad['layers']['3']['native_query']['sha256']='0'*64
            elif change=='finite':bad['layers']['3']['native_query']['finite']=False
            else:bad['non_query_arguments_preserved']=False
            with self.assertRaises(ValueError,msg=change):audit_exchange(bad,q,record,'capture',proof)


class FreshQueryAccountingTests(unittest.TestCase):
    def setUp(self):
        self.plan=read_plan();self.cases,self.registry,self.queries=build_registry(CharRuntime(),self.plan,prior_index())
    def test_all_fresh_events_roles_styles_and_baseline_failures_are_retained(self):
        self.assertEqual([c['seed'] for c in self.cases],list(range(75000,75008)))
        self.assertEqual((len(self.registry['contexts']),len(self.queries)),(40,240))
        rows={}
        for model in MODELS:
            rows[model]=[{'id':key,'condition':condition,'reader':reader,'greedy_exact':condition=='HN','answer_nll':1. if condition=='HN' else 2.}
                for key,q in self.queries.items() for condition in (('before',) if q['episode']=='before' else CONDITIONS) for reader in READERS]
        report=summarize(rows,self.queries,{**self.plan,'seeds':list(range(75000,75008))})
        self.assertEqual((report['primary_readouts'],report['cell_count'],report['baseline_failures_excluded']),(3264,20,0))
        self.assertEqual(report['cells']['native']['HH.full']['greedy_exact'],0)
        self.assertEqual(report['cells']['native']['HN.full']['joint6'],{'correct':32,'total':32})
        self.assertEqual(report['paired_contrasts']['native']['HN.full_vs_HH']['all']['gain'],192)
        self.assertEqual(report['before_after']['native']['HH.full']['related-8']['fixed_prefix_evidence']['pairs'],32)
        self.assertFalse(report['architecture_effectiveness_claim'])
    def test_missing_or_duplicated_primary_rows_cannot_pass(self):
        rows={model:[{'id':key,'condition':condition,'reader':reader,'greedy_exact':False,'answer_nll':1.}
            for key,q in self.queries.items() for condition in (('before',) if q['episode']=='before' else CONDITIONS) for reader in READERS] for model in MODELS}
        rows['native'].pop()
        with self.assertRaises(ValueError):summarize(rows,self.queries,{**self.plan,'seeds':list(range(75000,75008))})
    def test_complete222source_import_closure_and_exact_smoke_counts(self):
        root=Path(__file__).resolve().parents[1];self.assertEqual(len(sources()),222)
        for name in sources():
            path=root/name;self.assertTrue(path.is_file(),name)
            if path.suffix!='.py':continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else [v.name for v in node.names] if isinstance(node,ast.Import) else [])
                for module in modules:
                    target=module.replace('.','/')+'.py'
                    if (root/target).is_file():self.assertIn(target,sources(),f'{name}: {target}')
        contexts=selected_contexts(self.registry,True)
        count=sum(len(context_queries(c,self.queries,True))*(1 if c['episode']=='before' else 4)*2*2 for c in contexts)
        self.assertEqual(count,204);self.assertFalse(self.plan['smoke']['accuracy_gate'])

    def test_real_runner_cpu_smoke_checks_all_source_and_donor_guard_counts(self):
        from scripts import run_fresh_query as runner
        from sas_memory.runtime import TextRuntime
        class RunnerRuntime(TinyRuntime):
            @torch.inference_mode()
            def advance(self,text,cache=None,prefix_text=''):
                return super().advance(text,ToyCache(0) if cache is None else cache,prefix_text)
            probe=TextRuntime.probe
            def candidate_scores(self,prompt,query_cache,logits,candidates,batch=False):
                self.model(input_ids=torch.tensor([[65]]),past_key_values=fork_cache(query_cache))
                return {c:{'nll_sum':float(logits.abs().mean()),'nll_mean':float(logits.abs().mean())} for c in candidates}
            def greedy_answer(self,cache,logits):
                self.model(input_ids=torch.tensor([[65]]),past_key_values=fork_cache(cache));return '123'
        def tail(runtime,prefix,prefix_text,text,tokens,mode):
            cache,logits,count=runtime.advance(text,prefix,prefix_text)
            for key,tensor in cache_tensors(cache):
                if key[1] in ('recurrent_states','conv_states'):tensor.add_(1)
            return cache,logits,{'unit_fixture':True}
        previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            runtime=RunnerRuntime();_,registry,queries=build_registry(runtime,self.plan,prior_index())
            control={'answer':'123','answer_nll':1.,'greedy_exact':True,'generated':'123','query_tokens':1}
            ids=[str(i) for i in range(12)];legacy={key:{'ledger_tokens':3} for key in ids};anchors={k:control for k in ids}
            refs={'legacy_ids':ids,'legacy_queries':legacy,'legacy_full':{'native':anchors},'model_states':{'native':{}}}
            state={'token_policy':'none','completed_primary':0,'completed_same_source_replays':0,'completed_captures':0,
                'native_before_attach_checks':[],'attached_none_checks':[],'legacy_full_checks':[],'native_restore_checks':[],
                'prefixes':{},'sources':{}}
            with tempfile.TemporaryDirectory() as folder,patch.object(runner,'measure',return_value=control),\
                patch.object(runner,'actual_prompt',return_value='control'),patch.object(runner,'advance_tail',side_effect=tail),torch.inference_mode():
                runner.run_model(runtime,'native',self.plan,registry,queries,refs,True,Path(folder),state,anchors,lambda:None)
                self.assertEqual((state['completed_primary'],state['completed_same_source_replays'],state['completed_captures']),(102,48,48))
                self.assertEqual([v['source_guard_checks'] for v in state['prefixes'].values()],[208])
                for source in state['sources'].values():
                    self.assertEqual(source['caches']['N']['source_guard_checks'],57);self.assertEqual(source['caches']['H']['source_guard_checks'],56)
                donors=[json.loads(v) for v in (Path(folder)/'native.donors.jsonl').read_text().splitlines()]
                self.assertEqual({v['bank']['guard_checks'] for v in donors if v['origin']=='N'},{10})
                self.assertEqual({v['bank']['guard_checks'] for v in donors if v['origin']=='H'},{9})
                self.assertTrue(state['hooks_restored'])
        finally:torch.set_num_threads(previous_threads)


if __name__=='__main__':unittest.main()
