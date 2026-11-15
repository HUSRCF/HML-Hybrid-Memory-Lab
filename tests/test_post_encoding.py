"""Executable-truth and causal-input controls for post-encoding experiments."""
import copy
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from scripts.post_encoding_tasks import make_case,question,truth,validate_case,ROLES,STYLES
from scripts.post_encoding_data import read_plan,select_seeds,build_registry,sources


class CharRuntime:
    prompt_start='[user]\n'
    def ids(self,text):return list(text.encode())
    def continuation_ids(self,prefix,suffix):return self.ids(prefix+suffix)[len(self.ids(prefix)):]
    def query_suffix(self,q):return 'Question: '+q+'\nAnswer:'


def prior_index():
    return {'seeds':[],'identifiers':[],'complete_prefix_ids_sha256':[],'complete_prompt_ids_sha256':[]}


class PostEncodingDataTests(unittest.TestCase):
    def test_normal_prefix_and_required_update_only(self):
        for seed in range(74000,74008):
            case=make_case(seed);before=case['episodes'][0]
            self.assertEqual(len(case['prefix_events']),9)
            original={r:truth(case,before,r)[0] for r in ROLES}
            self.assertEqual(len(set(original.values())),3)
            target_prefix=[e['value'] for e in case['prefix_events'] if e['target']==case['target']]
            self.assertEqual(original['target_initial'],str(target_prefix[0]))
            self.assertEqual(original['target_current'],str(target_prefix[-1]))
            for ep in case['episodes'][1:]:
                got={r:truth(case,ep,r)[0] for r in ROLES}
                self.assertEqual(got['target_initial'],original['target_initial'])
                self.assertEqual(got['protected_current'],original['protected_current'])
                if ep['kind']=='related':
                    self.assertNotEqual(got['target_current'],original['target_current'])
                    self.assertEqual(got['target_current'],str(ep['tail'][-1]['value']))
                else:self.assertEqual(got,original)

    def test_pair_only_final_update_target_changes(self):
        case=make_case(74000)
        for count in (8,64):
            u,v=[next(e for e in case['episodes'] if e['id']==f'{k}-{count}')['tail'] for k in ('unrelated','related')]
            self.assertEqual(u[:-1],v[:-1]);self.assertEqual(u[-1]['value'],v[-1]['value'])
            self.assertNotEqual(u[-1]['target'],v[-1]['target'])
            self.assertEqual(v[-1]['target'],case['target'])

    def test_illegal_protected_update_is_rejected(self):
        case=make_case(74000);bad=copy.deepcopy(case)
        bad['episodes'][1]['tail'][0]['target']=case['protected']
        with self.assertRaises(ValueError):validate_case(bad)
        bad=copy.deepcopy(case);bad['episodes'][2]['tail'][-1]['value']=case['prefix_events'][-1]['value']
        with self.assertRaises(ValueError):validate_case(bad)

    def test_seed_choice_uses_prior_registry_only(self):
        plan=read_plan();index=prior_index();index['seeds']=[74000,74002]
        self.assertEqual(select_seeds(index,plan),[74001,*range(74003,74010)])
        self.assertEqual(len(sources()),206)

    def test_complete206source_closure_includes_every_local_import(self):
        root=Path(__file__).resolve().parents[1]
        for name in sources():
            path=root/name;self.assertTrue(path.is_file(),name)
            if path.suffix!='.py':continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                    [v.name for v in node.names] if isinstance(node,ast.Import) else [])
                for module in modules:
                    target=module.replace('.','/')+'.py'
                    if (root/target).is_file():self.assertIn(target,sources(),f'{name}: {target}')

    def test_grid_one_context_shared_by_all_questions(self):
        cases,registry,queries=build_registry(CharRuntime(),read_plan(),prior_index())
        self.assertEqual((len(cases),len(registry['contexts']),len(queries)),(8,40,240))
        self.assertEqual(sum(q['prefix_evidence'] for q in queries.values()),208)
        for context in registry['contexts']:
            rows=[queries[k] for k in context['query_ids']]
            self.assertEqual({(q['role'],q['question_style']) for q in rows},{(r,s) for r in ROLES for s in STYLES})
            self.assertEqual({q['source_key'] for q in rows},{context['source_key']})
            self.assertTrue(all(q['input_ids'][:context['prefix_tokens']]==registry['prefixes'][context['prefix_key']]['input_ids'] for q in rows))
            self.assertEqual(len({q['history_keep_bits'] for q in rows}),3)

    def test_exact_prompt_and_prefix_overlap_are_rejected(self):
        _,registry,queries=build_registry(CharRuntime(),read_plan(),prior_index())
        for key,value in (('complete_prefix_ids_sha256',next(iter(registry['prefixes']))),
                          ('complete_prompt_ids_sha256',next(iter(queries.values()))['prompt_ids_sha256'])):
            old=prior_index();old[key]=[value]
            with self.assertRaises(ValueError):build_registry(CharRuntime(),read_plan(),old)

    def test_cross_record_tokenization_is_rejected(self):
        class Crossing(CharRuntime):
            def ids(self,text):
                ids=super().ids(text)
                if text.count('The balance of account')>1:ids[0]=0
                return ids
        with self.assertRaisesRegex(ValueError,'boundary'):build_registry(Crossing(),read_plan(),prior_index())

    def test_unknown_task_roles_fail(self):
        case=make_case(74000)
        with self.assertRaises(ValueError):truth(case,case['episodes'][0],'unknown')
        with self.assertRaises(ValueError):question(case,'unknown','original')


class ToyCache:
    def __init__(self,length=3):
        self.length=length;self.bookkeeping={'counter':0};self.layers=[]
        for i in range(24):
            if i%4==3:
                self.layers.append(SimpleNamespace(keys=torch.full((1,1,length,2),float(i)),
                    values=torch.full((1,1,length,2),float(i+1))))
            else:self.layers.append(SimpleNamespace(recurrent_states=[torch.full((1,1,2,2),float(i))],
                conv_states=[torch.full((1,2,2),float(i))]))
    def get_seq_length(self):return self.length


class PostEncodingRuntimeTests(unittest.TestCase):
    def test_guards_detect_each_cache_field_and_bookkeeping_and_logits(self):
        from scripts.post_encoding_runtime import CacheGuard
        from sas_memory.runtime import cache_tensors
        for key in dict(cache_tensors(ToyCache())):
            source=ToyCache();guard=CacheGuard(source);dict(cache_tensors(source))[key].add_(1)
            with self.assertRaises(AssertionError):guard.check()
        for mutate in (lambda c:c.bookkeeping.update(counter=1),lambda c:setattr(c,'length',4)):
            source=ToyCache();guard=CacheGuard(source);mutate(source)
            with self.assertRaises(ValueError):guard.check()
        source=ToyCache();logits=torch.zeros(1,3);guard=CacheGuard(source,logits);logits.add_(1)
        with self.assertRaises(ValueError):guard.check()

    def test_storage_alias_rejected_even_for_disjoint_views(self):
        from scripts.post_encoding_runtime import assert_isolated,CacheGuard
        from sas_memory.runtime import fork_cache
        parent=ToyCache();child=fork_cache(parent);assert_isolated(parent,child);CacheGuard(parent).check()
        base=torch.zeros(4);parent.layers[0].recurrent_states=[base[:2]];child.layers[0].recurrent_states=[base[2:]]
        with self.assertRaises(ValueError):assert_isolated(parent,child)

    def test_untracked_bookkeeping_tensor_cannot_escape_guard(self):
        from scripts.post_encoding_runtime import CacheGuard
        source=ToyCache();source.bookkeeping['hidden_counter']=torch.zeros(())
        with self.assertRaisesRegex(ValueError,'Untracked tensor'):CacheGuard(source)
        source=ToyCache();guard=CacheGuard(source);source.layers[3].extra_tensor=torch.zeros(())
        with self.assertRaisesRegex(ValueError,'Untracked tensor'):guard.check()

    def test_all_historical_kv_slices_checked(self):
        from scripts.post_encoding_runtime import historical_kv_proof
        from sas_memory.runtime import fork_cache
        prefix=ToyCache();tail=fork_cache(prefix)
        for i in (3,7,11,15,19,23):
            for field in ('keys','values'):
                t=getattr(tail.layers[i],field);setattr(tail.layers[i],field,torch.cat([t,torch.zeros_like(t[...,:1,:])],-2))
        self.assertEqual(len(historical_kv_proof(prefix,tail,3)),12)
        tail.layers[23].values[...,1,:].add_(1)
        with self.assertRaises(ValueError):historical_kv_proof(prefix,tail,3)

    def test_tail_interface_has_no_question_and_closes_writer_before_reading(self):
        import inspect
        from tests.test_oracle_write_hold import FakeModel
        from sas_memory.runtime import fork_cache
        from scripts.post_encoding_runtime import advance_tail,CacheGuard
        from scripts.gate_component_runtime import GDN_LAYERS
        self.assertEqual(list(inspect.signature(advance_tail).parameters),
            ['runtime','prefix_cache','prefix_text','tail_text','prefix_tokens','mode'])
        class Runtime:
            def __init__(self):self.model=FakeModel()
            def continuation_ids(self,prefix,suffix):return list(suffix.encode())
            def advance(self,text,cache,prefix_text):
                work=fork_cache(cache);n=len(text);self.model(input_ids=torch.ones((1,n),dtype=torch.long),past_key_values=work)
                for i in GDN_LAYERS:
                    g,b=self.model.model.layers[i].linear_attn.last
                    if (g!=0).any() or (b!=0).any():work.layers[i].recurrent_states[0].add_(1)
                    work.layers[i].conv_states[0].add_(1)
                for i in (3,7,11,15,19,23):
                    for field in ('keys','values'):
                        t=getattr(work.layers[i],field);setattr(work.layers[i],field,torch.cat([t,torch.zeros(1,1,n,2)],-2))
                work.length+=n;return work,torch.ones(1,3),n
        runtime=Runtime();prefix=ToyCache();guard=CacheGuard(prefix)
        for mode in ('none','decay','delta','both'):
            cache,logits,proof=advance_tail(runtime,prefix,'abc','tail',3,mode);guard.check()
            self.assertEqual(cache.get_seq_length(),7);self.assertEqual(len(proof['historical_kv_sha256']),12)
            self.assertEqual(len(proof['both_mode_exact_recurrent_keys']),18 if mode=='both' else 0)
            self.assertFalse(runtime.model._forward_pre_hooks)
            runtime.model(input_ids=torch.ones((1,2),dtype=torch.long),past_key_values=cache)
            for i in GDN_LAYERS:
                g,b=runtime.model.model.layers[i].linear_attn.last
                self.assertTrue((g<0).all());self.assertTrue((b>0).all())

    def test_actual_tail_gate_coverage_and_native_component_are_checked(self):
        from tests.test_oracle_write_hold import FakeModel
        from scripts.gate_component_runtime import ComponentHold
        from scripts.post_encoding_runtime import check_gate_stats
        model=FakeModel();cache=ToyCache()
        with ComponentHold(model,[True]*3+[False]*4,'delta') as hold:
            model(input_ids=torch.ones((1,4),dtype=torch.long),past_key_values=cache);proof=hold.report()
        check_gate_stats(proof,'delta',4)
        for key in ('rows','chunk_calls','recurrent_calls','unheld_rows','forced_decay_rows','forced_delta_rows'):
            bad=copy.deepcopy(proof);bad['layers'][0][key]+=1
            with self.assertRaises(ValueError):check_gate_stats(bad,'delta',4)

    def test_cached_attention_rejects_prefill_missing_layer_and_leakage(self):
        from scripts.post_encoding_runtime import check_cached_attention,ATTENTION_LAYERS
        proof={i:{'prefill_calls':0,'cached_calls':1,'query_rows':2,'max_suppressed_probability':0.,
            'max_suppressed_mass':0.,'max_kept_record_mass':.1} for i in ATTENTION_LAYERS}
        check_cached_attention(proof,True)
        for key,value in (('prefill_calls',1),('cached_calls',0),('max_suppressed_mass',.1),
                          ('max_kept_record_mass',0.),('max_kept_record_mass',float('nan'))):
            bad=copy.deepcopy(proof);bad[23][key]=value
            with self.assertRaises(ValueError):check_cached_attention(bad,True)
        bad=copy.deepcopy(proof);del bad[3]
        with self.assertRaises(ValueError):check_cached_attention(bad,True)
        with self.assertRaises(ValueError):check_cached_attention(proof,False)
        for p in proof.values():p.update(max_suppressed_probability=.01,max_suppressed_mass=.1)
        check_cached_attention(proof,False)

    def test_cache_auditor_rejects_missing_tensor_and_wrong_guard_count(self):
        from scripts.post_encoding_runtime import CacheGuard
        from scripts.analyze_post_encoding import audit_cache
        record=CacheGuard(ToyCache(),torch.ones(1,3)).record();audit_cache(record,3,1)
        for change in ('tensor','checks','length'):
            bad=copy.deepcopy(record)
            if change=='tensor':bad['tensors'].pop(next(iter(bad['tensors'])))
            elif change=='checks':bad['source_guard_checks']=0
            else:bad['metadata']['original_metadata']['sequence_length']=4
            with self.assertRaises(ValueError):audit_cache(bad,3,1)

    def test_metrics_retain_failed_baselines_and_all_cells(self):
        from scripts.post_encoding_metrics import summarize
        from scripts.post_encoding_data import MODES,READERS
        plan=read_plan();cases,registry,queries=build_registry(CharRuntime(),plan,prior_index());all_rows={}
        for name in plan['model_order']:
            rows=[]
            for context in registry['contexts']:
                for mode in (('none',) if context['episode']=='before' else MODES):
                    for key in context['query_ids']:
                        for reader in READERS:
                            rows.append({'id':key,'mode':mode,'reader':reader,'greedy_exact':mode=='delta','answer_nll':1. if mode=='delta' else 2.})
            all_rows[name]=rows
        report=summarize(all_rows,queries,plan)
        self.assertEqual((report['primary_readouts'],report['cells']),(8160,170))
        self.assertEqual(report['baseline_failures_excluded'],0)
        self.assertEqual(report['before_after']['native']['unrelated-8.delta.full']['all']['gain'],48)
        self.assertEqual(report['before_after']['native']['related-8.delta.full']['fixed_prefix_evidence']['pairs'],32)
        self.assertEqual(report['update_positive_controls']['native']['related-8.delta.full']['new_tail_evidence']['readouts'],16)
        self.assertEqual(report['cell_metrics']['native']['unrelated-8.delta.full']['joint6_streams'],8)
        self.assertFalse(report['training_or_scaleup_trigger'])


if __name__=='__main__':unittest.main()
