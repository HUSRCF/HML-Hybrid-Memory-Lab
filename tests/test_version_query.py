"""Paired semantics, frozen-function reuse and complete two-model CPU audit."""
import ast
from copy import copy,deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import version_query_data as data
from scripts import run_version_query as runner
from scripts import analyze_version_query as analysis
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json,state_fingerprint
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


class VersionQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=data.read_plan()
        with patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model constructor')), \
             patch('transformers.Qwen3_5ForCausalLM.from_pretrained',side_effect=AssertionError('No model loader')), \
             patch('safetensors.torch.load_file',side_effect=AssertionError('No weights loader')), \
             patch('torch.cuda._lazy_init',side_effect=AssertionError('No GPU initialization')):
            cls.cases,cls.old_registry,cls.old_queries,cls.refs=data.reference_inputs()
            cls.runtime=tokenizer_runtime(cls.plan['model_path']);cls.old_registry_hash=data_fingerprint(cls.old_registry)
            cls.registry,cls.queries=data.build_registry(cls.runtime,cls.plan,cls.old_registry,cls.old_queries)

    def test_all_literal_pairs_have_one_token_and_identical_truth(self):
        self.assertEqual(len(self.queries),864);self.assertEqual(len(self.registry['contexts']),144)
        for q in self.queries.values():
            other=self.queries[q['paired_query_id']]
            self.assertEqual(len(q['input_ids']),len(other['input_ids']))
            self.assertEqual([i for i,(a,b) in enumerate(zip(q['input_ids'],other['input_ids'])) if a!=b],[q['different_token_index']])
            self.assertEqual(q['different_question_token_index'],5)
            self.assertEqual(len(q['input_ids'])-q['ledger_tokens'],24)
            for key in ('answer','answer_ids','support_event_index','history_keep_bits','source_key'):
                self.assertEqual(q[key],other[key])

    def test_every_history_is_unchanged_and_all_old_errors_retained(self):
        self.assertEqual(data_fingerprint(self.old_registry),self.old_registry_hash)
        self.assertEqual(self.registry['prefixes'],self.old_registry['prefixes'])
        for new,old in zip(self.registry['contexts'],self.old_registry['contexts']):
            self.assertEqual({k:v for k,v in new.items() if k!='query_ids'},{k:v for k,v in old.items() if k!='query_ids'})
        self.assertFalse(set(self.queries)&set(self.old_queries))
        self.assertEqual(len(self.refs['archived_full']['gdn_all_tokens']),864)
        self.assertEqual(sum(not r['greedy_exact'] for r in self.refs['archived_full']['gdn_all_tokens']),43)

    def test_bad_literal_pair_raises_without_filtering(self):
        broken=copy(self.runtime);real_ids=self.runtime.ids
        broken.ids=lambda text: real_ids(text)+([0] if ' the first balance assigned ' in text else [])
        with self.assertRaises(ValueError):data.build_registry(broken,self.plan,self.old_registry,self.old_queries)

    def test_frozen_code_object_and_original_global_bindings(self):
        original=runner.frozen.run_model;before=dict(original.__globals__);clone=runner.cloned_run_model()
        self.assertIs(clone.__code__,original.__code__);self.assertIsNot(clone.__globals__,original.__globals__)
        changed=sorted(k for k,v in clone.__globals__.items() if v is not before[k])
        self.assertEqual(changed,['READERS','context_queries'])
        self.assertEqual(clone.__globals__['READERS'],('full',));self.assertEqual(runner.frozen.READERS,('full','oracle'))
        self.assertTrue(all(original.__globals__[k] is v for k,v in before.items()))

    def test_controller_and_arithmetic_cleanup_on_cloned_forward_failure(self):
        import torch
        from types import SimpleNamespace
        instances=[];arithmetics=[]
        class Controller(torch.nn.Module):
            def __init__(self,*args,**kwargs):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.closed=False;instances.append(self)
            def description(self):return {'fixture':True}
            def set_token_policy(self,*args):pass
            def close(self):self.closed=True
        class Arithmetic:
            def __init__(self,controller):self.closed=False;arithmetics.append(self)
            def description(self):return {'fixture':True}
            def close(self):self.closed=True
        initial=Controller();fingerprint=state_fingerprint(initial)
        native=torch.nn.Module();native.device=torch.device('cpu')
        def fail(*args):raise RuntimeError('intentional cached-prefix failure')
        runtime=SimpleNamespace(model=native,advance=fail);before=dict(runner.frozen.run_model.__globals__)
        clone=runner.cloned_run_model();clone.__globals__.update(StableAdapterLinears=Arithmetic,check_affine_scope=lambda *args:{'fixture':True})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'fixture.bin';checkpoint.write_bytes(b'fixture')
            spec={'group_index':0,'rank':1,'mode':'gdn_only','adapt_gdn_keys':True,'checkpoint':str(checkpoint),
                'checkpoint_sha256':streaming_hash(checkpoint),'parameters':1}
            refs={'legacy_ids':[],'legacy_queries':{},'legacy_full':{'gdn_all_tokens':{}},
                'model_states':{'gdn_all_tokens':{'description':{'fixture':True},'adapter_state_sha256_before':fingerprint}}}
            state={'native_before_attach_checks':[],'attached_none_checks':[],'legacy_full_checks':[]}
            with patch('sas_memory.controllers.GroupController',Controller),patch('safetensors.torch.load_file',return_value={'weight':torch.zeros(1)}), \
                 patch.object(runner,'cloned_run_model',return_value=clone),torch.inference_mode():
                with self.assertRaisesRegex(RuntimeError,'intentional'):
                    runner.invoke_model(runtime,'gdn_all_tokens',{'models':{'gdn_all_tokens':spec}},self.registry,self.queries,refs,True,root,state,{},lambda:None)
        self.assertTrue(instances[-1].closed);self.assertTrue(arithmetics[-1].closed)
        self.assertTrue(state['arithmetic_hooks_removed'])
        self.assertTrue(all(runner.frozen.run_model.__globals__[k] is v for k,v in before.items()))

    def test_complete_source_closure_including_original_240(self):
        self.assertEqual(len(data.sources()),246);self.assertEqual(len(data.predecessor_sources()),240)
        for name,digest in data.predecessor_sources().items():self.assertEqual(streaming_hash(data.ROOT/name),digest)
        for name in data.NEW_SOURCES:
            if not name.endswith('.py'):continue
            for node in ast.walk(ast.parse((data.ROOT/name).read_text())):
                if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith(('scripts.','sas_memory.')):
                    self.assertIn(node.module.replace('.','/')+'.py',data.sources())

    def fixture(self,destination):
        root=Path(destination);run=root/'run';run.mkdir();packet=root/'packet';packet.mkdir()
        write_json(packet/'manifest.json',{'synthetic_cpu_fixture':True})
        predecessor=Path(self.plan['source_run']);old=json.loads((predecessor/'manifest.json').read_text())
        digests={n:streaming_hash(data.ROOT/n) for n in sorted(data.sources())};now=datetime.now(timezone.utc)
        registered={'created_utc':now.isoformat(),'source_sha256':digests}
        m=deepcopy(old);m.update(status='completed',started_utc=(now+timedelta(seconds=1)).isoformat(),
            completed_utc=(now+timedelta(seconds=2)).isoformat(),smoke=True,reverse_model_order=False,
            fresh_streams=False,reused_cohort=True,model_order=list(data.MODELS),completed_models=list(data.MODELS),
            packet=str(packet),packet_manifest_sha256=streaming_hash(packet/'manifest.json'),protocol_sha256=streaming_hash(data.PROTOCOL),
            source_sha256=digests,smoke_equivalence_gate=None,primary_same_subset_replay=None,runner_reuse=runner.RUNNER_REUSE,models={})
        for name in ('plan','precision_addendum','kernel_validation_addendum'):
            value=self.plan if name=='plan' else json.loads((predecessor/(name+'.json')).read_text())
            write_json(run/(name+'.json'),value)
        (run/'installed_modeling_source.py').write_bytes(Path(self.plan['installed_modeling_source']).read_bytes())
        for name in digests:
            target=run/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((data.ROOT/name).read_bytes())
        contexts=data.selected_contexts(self.registry,True);before=contexts[0]
        for model in data.MODELS:
            state=deepcopy(old['models'][model]);state.update(completed_primary=54,runner_bindings=runner.RUNNER_REUSE,
                prefixes={before['prefix_key']:deepcopy(old['models'][model]['prefixes'][before['prefix_key']])},
                sources={c['id']:deepcopy(old['models'][model]['sources'][c['id']]) for c in contexts[1:]})
            for c in state['prefixes'].values():c['source_guard_checks']=64
            for c in state['sources'].values():c['cache']['source_guard_checks']=8
            state['predecessor_sources']=runner.verify_model_sources(state,self.refs['source_caches'][model])
            proof_template=deepcopy(self.refs['archived_full'][model][0]['proof']);rows=[]
            for c in contexts:
                for q in data.context_queries(c,self.queries,True):
                    proof=deepcopy(proof_template);proof['reader_keep_sha256']=data_fingerprint(q['history_keep_bits'])
                    rows.append({'id':q['id'],'model':model,'reader':'full','answer':q['answer'],'answer_nll':.5,
                        'generated':q['answer'],'greedy_exact':True,'query_tokens':24,'registration_sha256':data_fingerprint(q),
                        'source_key':q['source_key'],'prompt_ids_sha256':q['prompt_ids_sha256'],'prefill':'cached',
                        'measurement_origin':'normally_encoded_without_intervention','proof':proof})
            raw=run/f'{model}.jsonl';raw.write_text(''.join(json.dumps(r)+'\n' for r in rows));state['results_sha256']=streaming_hash(raw)
            reference=run/f'{model}.references.jsonl';reference.write_text(''.join(json.dumps({**r,'archived_reference':True,
                'predecessor_source_run':self.plan['source_run'],'predecessor_manifest_sha256':self.plan['source_manifest_sha256']})+'\n'
                for r in self.refs['archived_full'][model]))
            state.update(archived_references=864,references_sha256=streaming_hash(reference));m['models'][model]=state
        write_json(run/'manifest.json',m)
        return run,packet,(registered,self.cases,self.registry,self.queries,self.refs)

    def test_complete_both_model_audit_without_loading_model(self):
        with tempfile.TemporaryDirectory() as d:
            run,packet,values=self.fixture(d)
            with patch.object(analysis,'DESTINATION',packet),patch.object(analysis,'registered_inputs',return_value=values), \
                 patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model')):
                result=analysis.audit_run(run,True)
        self.assertEqual(result['primary_readouts_audited'],108);self.assertEqual(result['archived_full_references_audited'],1728)
        self.assertEqual(len(result['metrics']['cells']),72);self.assertEqual(len(result['metrics']['per_stream_state_to_event_pairs']),54)

    def test_missing_primary_row_rejected_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as d:
            run,packet,values=self.fixture(d);m=json.loads((run/'manifest.json').read_text());raw=run/'gdn_all_tokens.jsonl'
            raw.write_text('\n'.join(raw.read_text().splitlines()[:-1])+'\n');m['models']['gdn_all_tokens']['results_sha256']=streaming_hash(raw)
            write_json(run/'manifest.json',m)
            with patch.object(analysis,'DESTINATION',packet),patch.object(analysis,'registered_inputs',return_value=values):
                with self.assertRaisesRegex(ValueError,'unfiltered'):analysis.audit_run(run,True)

    def test_changed_predecessor_state_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run,packet,values=self.fixture(d);m=json.loads((run/'manifest.json').read_text())
            cache=next(iter(m['models']['gdn_all_tokens']['sources'].values()))['cache']
            key=next(k for k in cache['tensors'] if 'recurrent_states' in k);cache['tensors'][key]['sha256']='0'*64
            write_json(run/'manifest.json',m)
            with patch.object(analysis,'DESTINATION',packet),patch.object(analysis,'registered_inputs',return_value=values):
                with self.assertRaisesRegex(ValueError,'predecessor'):analysis.audit_run(run,True)

    def test_original_baseline_error_cannot_be_edited(self):
        with tempfile.TemporaryDirectory() as d:
            run,packet,values=self.fixture(d);m=json.loads((run/'manifest.json').read_text());path=run/'gdn_all_tokens.references.jsonl'
            rows=[json.loads(l) for l in path.read_text().splitlines()];bad=next(r for r in rows if not r['greedy_exact'])
            bad.update(greedy_exact=True,generated=bad['answer']);path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            m['models']['gdn_all_tokens']['references_sha256']=streaming_hash(path);write_json(run/'manifest.json',m)
            with patch.object(analysis,'DESTINATION',packet),patch.object(analysis,'registered_inputs',return_value=values):
                with self.assertRaisesRegex(ValueError,'archived'):analysis.audit_run(run,True)

    def test_metric_order_invariance_and_archived_error_attribution(self):
        rows={m:[{'id':q['id'],'model':m,'reader':'full','answer_nll':.5,'greedy_exact':True,'generated':q['answer']}
            for q in self.queries.values()] for m in data.MODELS}
        forward=analysis.summarize(rows,self.queries,self.refs,self.cases)
        reverse=analysis.summarize(dict(reversed(list(rows.items()))),self.queries,self.refs,self.cases)
        self.assertEqual(forward,reverse);self.assertEqual(len(forward['per_stream_state_to_event_pairs']),864)
        archived=[r for r in forward['value_attribution_all_readouts'] if r['origin']=='archived' and r['model']=='gdn_all_tokens' and not r['greedy_exact']]
        self.assertEqual(len(archived),43);self.assertEqual(sum(r['category']=='queried_account_other_version' for r in archived),40)


if __name__=='__main__':unittest.main()
