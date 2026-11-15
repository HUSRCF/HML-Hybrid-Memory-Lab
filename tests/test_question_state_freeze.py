"""Real CPU zero-gate behavior, answer cleanup and complete independent audit."""
import ast
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from tests.test_oracle_write_hold import FakeModel
from tests.test_post_encoding import ToyCache
from scripts import question_state_freeze_data as data
from scripts import question_state_freeze_runtime as runtime
from scripts import run_question_state_freeze as runner
from scripts import analyze_question_state_freeze as analysis
from scripts.post_encoding_runtime import CacheGuard
from scripts.run_controller_trial import write_json
from sas_memory.runtime import fork_cache,assert_cache_equal
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


class CachedGateModel(FakeModel):
    def __init__(self):
        super().__init__();self.corrupt=None;self.fail_layer=None
        self.registered_gdns=torch.nn.ModuleList([layer.linear_attn for layer in self.model.layers])
        for i in runtime.GDN_LAYERS:
            gdn=self.model.layers[i].linear_attn
            gdn.in_proj_a.weight.data.zero_();gdn.in_proj_b.weight.data.zero_()
            def capture(*args,_gdn=gdn,**kwargs):
                g,b=kwargs['g'],kwargs['beta'];_gdn.last=(g.detach().clone(),b.detach().clone())
                state=_gdn.current_layer.recurrent_states[0]
                for t in range(g.shape[1]):state=g[:,t,0].exp().reshape(1,1,1,1)*state+3*b[:,t,0].reshape(1,1,1,1)
                _gdn.current_layer.recurrent_states[0]=state
            gdn.chunk_gated_delta_rule=capture;gdn.recurrent_gated_delta_rule=capture

    def forward(self,input_ids,past_key_values=None,**kwargs):
        cache=past_key_values;length=input_ids.shape[1];x=torch.ones((1,length,2))
        for i in runtime.GDN_LAYERS:
            if i==self.fail_layer:raise RuntimeError('intentional native-layer failure')
            gdn=self.model.layers[i].linear_attn;gdn.current_layer=cache.layers[i];gdn(x)
            cache.layers[i].conv_states[0]=cache.layers[i].conv_states[0]+1
        for i in (3,7,11,15,19,23):
            for name in ('keys','values'):
                old=getattr(cache.layers[i],name)
                setattr(cache.layers[i],name,torch.cat((old,torch.zeros((*old.shape[:-2],length,old.shape[-1]))),dim=-2))
        cache.length+=length
        if self.corrupt=='kv':cache.layers[3].keys[...,0,:].add_(1)
        if self.corrupt=='state':cache.layers[12].recurrent_states[0].add_(1)
        return SimpleNamespace(past_key_values=cache,logits=torch.zeros((1,1,10)))


class QuestionHoldRuntimeTests(unittest.TestCase):
    def inputs(self):
        source=ToyCache();return source,CacheGuard(source).record(),list(range(24))

    def assert_clean(self,model,originals):
        self.assertFalse(model._forward_pre_hooks);self.assertFalse(model._forward_hooks)
        for i in runtime.GDN_LAYERS:
            gdn=model.model.layers[i].linear_attn
            self.assertFalse(gdn.in_proj_a._forward_hooks);self.assertFalse(gdn.in_proj_b._forward_hooks)
            self.assertEqual(gdn.chunk_gated_delta_rule,originals[i])

    def test_each_equal_group_and_all18_hold_only_selected_actual_matrices(self):
        for arm,selected in data.ARMS.items():
            with self.subTest(arm=arm),torch.inference_mode():
                model=CachedGateModel();source,record,ids=self.inputs();saved=fork_cache(source)
                originals={i:model.model.layers[i].linear_attn.chunk_gated_delta_rule for i in runtime.GDN_LAYERS}
                work=fork_cache(source)
                with runtime.QuestionHold(model,source,ids,selected,record) as hold:
                    output=model(input_ids=torch.tensor([ids]),past_key_values=work,use_cache=True)
                    proof=hold.report();self.assert_clean(model,originals)
                self.assertEqual(proof['selected_layers'],selected)
                for i in runtime.GDN_LAYERS:
                    equal=torch.equal(source.layers[i].recurrent_states[0],output.past_key_values.layers[i].recurrent_states[0])
                    self.assertEqual(equal,i in selected)
                    self.assertEqual(proof['gates']['layers'][str(i)]['held_rows'],24 if i in selected else 0)
                self.assertEqual(proof['convolution_changed_layers'],list(runtime.GDN_LAYERS))
                self.assertEqual(len(proof['historical_kv_sha256']),12);assert_cache_equal(source,saved)

    def test_gate_hooks_are_removed_before_answer_and_decode_uses_native_gates(self):
        with torch.inference_mode():
            model=CachedGateModel();source,record,ids=self.inputs();selected=[12,13,14]
            with runtime.QuestionHold(model,source,ids,selected,record) as hold:
                result=model(input_ids=torch.tensor([ids]),past_key_values=fork_cache(source))
                proof=hold.report();states={i:result.past_key_values.layers[i].recurrent_states[0].clone() for i in selected}
                model(input_ids=torch.tensor([[8]]),past_key_values=result.past_key_values)
                self.assertEqual(hold.report(),proof)
                for i in selected:
                    self.assertFalse(torch.equal(states[i],result.past_key_values.layers[i].recurrent_states[0]))
                    g,b=model.model.layers[i].linear_attn.last
                    self.assertTrue((g<0).all());self.assertTrue((b>0).all())

    def test_noop_observer_preserves_all_native_cache_fields_and_logits(self):
        with torch.inference_mode():
            model=CachedGateModel();source,record,ids=self.inputs()
            normal=model(input_ids=torch.tensor([ids]),past_key_values=fork_cache(source))
            with runtime.QuestionHold(model,source,ids,[],record) as hold:
                observed=model(input_ids=torch.tensor([ids]),past_key_values=fork_cache(source));hold.report()
            assert_cache_equal(normal.past_key_values,observed.past_key_values)
            self.assertTrue(torch.equal(normal.logits,observed.logits))

    def test_native_exception_restores_all_hooks_and_kernel_methods(self):
        model=CachedGateModel();model.fail_layer=14;source,record,ids=self.inputs()
        originals={i:model.model.layers[i].linear_attn.chunk_gated_delta_rule for i in runtime.GDN_LAYERS}
        with self.assertRaisesRegex(RuntimeError,'intentional'),torch.inference_mode():
            with runtime.QuestionHold(model,source,ids,[12,13,14],record):
                model(input_ids=torch.tensor([ids]),past_key_values=fork_cache(source))
        self.assert_clean(model,originals)

    def test_actual_matrix_and_historical_kv_corruption_are_rejected(self):
        for kind in ('kv','state'):
            with self.subTest(kind=kind),torch.inference_mode():
                model=CachedGateModel();model.corrupt=kind;source,record,ids=self.inputs()
                originals={i:model.model.layers[i].linear_attn.chunk_gated_delta_rule for i in runtime.GDN_LAYERS}
                with self.assertRaises(ValueError):
                    with runtime.QuestionHold(model,source,ids,[12,13,14],record):
                        model(input_ids=torch.tensor([ids]),past_key_values=fork_cache(source))
                self.assert_clean(model,originals)

    def test_bad_scope_rejected_and_wrong_question_cleans_up(self):
        model=CachedGateModel();source,record,ids=self.inputs()
        for selected,known in (([3],ids),([12,12],ids),([12],ids[:-1])):
            with self.assertRaises(ValueError):runtime.QuestionHold(model,source,known,selected,record)
            self.assertFalse(model._forward_pre_hooks)
        originals={i:model.model.layers[i].linear_attn.chunk_gated_delta_rule for i in runtime.GDN_LAYERS}
        with self.assertRaisesRegex(ValueError,'known-question'),torch.inference_mode():
            with runtime.QuestionHold(model,source,ids,[12],record):
                model(input_ids=torch.tensor([list(reversed(ids))]),past_key_values=fork_cache(source))
        self.assert_clean(model,originals)

    def test_canonical_observation_is_nonmutating_and_rejects_collisions_nan(self):
        original={'layers':{12:{'value':.1234567890123456}}};saved=deepcopy(original)
        self.assertEqual(runtime.canonical(original),{'layers':{'12':{'value':.1234567890123456}}})
        self.assertEqual(original,saved)
        for broken in ({0:1,'0':2},{'x':float('nan')},{(0,):1}):
            with self.assertRaises(ValueError):runtime.canonical(broken)

    def test_zero_support_probability_is_reportable_not_an_accuracy_gate(self):
        stats={str(i):{'prefill_calls':0,'cached_calls':2,'query_rows':26,'max_suppressed_probability':0.,
            'max_suppressed_mass':0.,'max_kept_record_mass':0.} for i in (3,7,11,15,19,23)}
        runtime.full_attention_contract(stats)
        stats['3']['max_kept_record_mass']=1.0001;runtime.full_attention_contract(stats)
        stats['3']['max_suppressed_probability']=float('nan')
        with self.assertRaises(ValueError):runtime.full_attention_contract(stats)


class QuestionFreezeDataAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=data.read_plan()
        with patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model constructor')), \
             patch('transformers.Qwen3_5ForCausalLM.from_pretrained',side_effect=AssertionError('No model loader')), \
             patch('torch.cuda._lazy_init',side_effect=AssertionError('No GPU init')):
            cls.cases,cls.registry,cls.queries,cls.refs=data.reference_inputs()
        cls.previous=json.loads((Path(cls.plan['source_run'])/'manifest.json').read_text())

    def test_all_inputs_old_errors_and261source_closure_retained(self):
        self.assertEqual((len(self.cases),len(self.registry['contexts']),len(self.queries)),(16,144,864))
        self.assertEqual(sum(not r['greedy_exact'] for r in self.refs['normal_full']),50)
        self.assertEqual(len(data.predecessor_sources()),254);self.assertEqual(len(data.sources()),261)
        for name in data.NEW_SOURCES:
            if not name.endswith('.py'):continue
            for node in ast.walk(ast.parse((data.ROOT/name).read_text())):
                if isinstance(node,ast.ImportFrom) and node.module:
                    target=node.module.replace('.','/')+'.py'
                    if (data.ROOT/target).is_file():self.assertIn(target,data.sources(),f'{name}: {target}')
        self.assertEqual(sum(len(data.context_queries(c,self.queries)) for c in data.selected_contexts(self.registry,True)),54)

    def fixture(self,root,reverse=False,shared=None):
        root=Path(root);run=root/('reverse' if reverse else 'forward');run.mkdir()
        if shared is None:
            packet=root/'packet';packet.mkdir();write_json(packet/'manifest.json',{'synthetic_cpu_fixture':True})
            now=datetime.now(timezone.utc);digests={n:streaming_hash(data.ROOT/n) for n in sorted(data.sources())}
            registered={'created_utc':now.isoformat(),'source_sha256':digests}
            seal={'path':'cpu-fixture-only','sha256':'fixture','created_utc':(now+timedelta(seconds=1)).isoformat(),'full_tests_passed':331}
            shared=(packet,registered,seal,now)
        packet,registered,seal,now=shared;digests=registered['source_sha256'];old=self.previous
        contexts=data.selected_contexts(self.registry,True);before=contexts[0]
        state=deepcopy(old['models'][data.MODEL]);state.update(token_policy='all',completed_primary=432,
            normal_reference_replays=54,completed_by_arm={a:54 for a in data.ARMS},
            readout_wall_seconds_with_observers={a:1. for a in data.ARMS},
            prefixes={before['prefix_key']:deepcopy(old['models'][data.MODEL]['prefixes'][before['prefix_key']])},
            sources={c['id']:deepcopy(old['models'][data.MODEL]['sources'][c['id']]) for c in contexts[1:]})
        for value in state['prefixes'].values():value['source_guard_checks']=442
        for value in state['sources'].values():value['cache']['source_guard_checks']=50
        state['predecessor_sources']=runner.source_replay(state,self.refs['source_caches'])
        template={r['id']:r for r in self.refs['normal_full']};rows={a:[] for a in data.ARMS}
        for context in contexts:
            source=state['prefixes'][context['prefix_key']] if context['episode']=='before' else state['sources'][context['id']]['cache']
            for q in data.context_queries(context,self.queries,True):
                for arm,selected in data.ARMS.items():
                    proof=deepcopy(template[q['id']]['proof']);matrices={}
                    for i in selected:
                        key=str((i,'recurrent_states',0));matrices[str(i)]={**source['tensors'][key],**source['layouts'][key]}
                    proof['question_hold']={'question_calls':1,'boundary':q['ledger_tokens'],'question_tokens':24,
                        'known_ids_sha256':data_fingerprint(q['input_ids'][q['ledger_tokens']:]),'selected_layers':selected,
                        'gates':{'enabled':bool(selected),'actual_gates_finite_and_exact':True,'layers':{str(i):{
                            'chunk_calls':1,'recurrent_calls':0,'rows':24,'held_rows':24 if i in selected else 0,
                            'unheld_rows':0 if i in selected else 24} for i in runtime.GDN_LAYERS}},
                        'selected_before':deepcopy(matrices),'selected_after':deepcopy(matrices),
                        'historical_kv_sha256':{k:v['sha256'] for k,v in source['tensors'].items() if "'keys'" in k or "'values'" in k},
                        'convolution_changed_layers':list(runtime.GDN_LAYERS),'hooks_removed_before_answer':True,
                        'kernel_methods_restored_before_answer':True,'source_and_work_storage_isolated':True,
                        'no_direct_state_copy_or_rollback':True,'policy_uses_no_answers_or_support_labels':True}
                    rows[arm].append({**deepcopy(template[q['id']]),'arm':arm,'proof':proof,
                        'measurement_origin':'normally_encoded_history_question_only_gate_policy'})
        state['file_sha256']={}
        for arm,values in rows.items():
            raw=run/f'{arm}.jsonl';raw.write_text(''.join(json.dumps(r)+'\n' for r in values));state['file_sha256'][raw.name]=streaming_hash(raw)
        order=list(data.ARMS)[::-1] if reverse else list(data.ARMS)
        execution=run/'execution.jsonl';execution.write_text(''.join(json.dumps(rows[a][i])+'\n' for i in range(54) for a in order))
        state['file_sha256'][execution.name]=streaming_hash(execution)
        reference=run/'references.jsonl';reference.write_text(''.join(json.dumps({**r,'archived_reference':True,
            'source_run':self.plan['source_run'],'source_manifest_sha256':self.plan['source_manifest_sha256']})+'\n' for r in self.refs['normal_full']))
        state.update(archived_references=864,references_sha256=streaming_hash(reference))
        fields=('model_path','generation','arithmetic_profile','precision_addendum_sha256','validation_addendum_sha256',
            'packages','actual_eos_token_id','generation_config_sha256','weights_before','weights_after',
            'native_anchors','native_component_kernel_checks')
        m={k:deepcopy(old[k]) for k in fields}
        m.update(status='completed',started_utc=(now+timedelta(seconds=2)).isoformat(),completed_utc=(now+timedelta(seconds=3)).isoformat(),
            training=False,new_adapters=False,fresh_confirmation=False,oracle_evidence=False,donor_forward=False,
            smoke=True,reverse_arm_order=reverse,arm_order=order,model=data.MODEL,packet=str(packet),
            packet_manifest_sha256=streaming_hash(packet/'manifest.json'),protocol_sha256=streaming_hash(data.PROTOCOL),
            registration_seal=seal,source_sha256=digests,model_state=state,hooks_restored=True,
            smoke_equivalence_gate=None,primary_same_subset_replay=None,execution_wall_seconds_with_controls=8.)
        expected=[('chunk_gated_delta_rule',1),('chunk_gated_delta_rule',7),('chunk_gated_delta_rule',65),
            ('recurrent_gated_delta_rule',1),('recurrent_gated_delta_rule',7),('chunk_gated_delta_rule',24),('recurrent_gated_delta_rule',1)]
        m['native_freeze_kernel_checks']=[{'kernel':name,'length':length,'initial_state_unmodified':True,
            'final_state_exact':True,'output_finite':True} for name,length in expected]
        for name in ('plan','precision_addendum','kernel_validation_addendum'):
            value=self.plan if name=='plan' else json.loads((Path(self.plan['source_run'])/(name+'.json')).read_text())
            write_json(run/(name+'.json'),value)
        (run/'installed_modeling_source.py').write_bytes(Path(self.plan['installed_modeling_source']).read_bytes())
        for name in digests:
            target=run/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((data.ROOT/name).read_bytes())
        write_json(run/'manifest.json',m)
        values=(registered,self.cases,self.registry,self.queries,self.refs)
        return run,shared,values

    def test_complete432readout_audit_both_arm_orders_without_model(self):
        with tempfile.TemporaryDirectory() as d:
            forward,shared,values=self.fixture(d);reverse,_,_=self.fixture(d,True,shared)
            with patch.object(analysis,'DESTINATION',shared[0]),patch.object(analysis,'registered_inputs',return_value=values), \
                 patch.object(analysis,'verified_seal',return_value=shared[2]), \
                 patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model')):
                report=analysis.audit_run(forward,True);equiv=analysis.audit_equivalence(forward,reverse)
            self.assertEqual(report['primary_readouts_audited'],432);self.assertEqual(report['archived_references_audited'],864)
            self.assertEqual(len(report['metrics']['normal_to_intervention_pairs']),378)
            self.assertEqual(equiv['raw_arm_files_exact'],8);self.assertFalse(equiv['accuracy_gate'])

    def test_corrupted_selected_state_and_missing_rows_rejected(self):
        for kind in ('matrix','missing'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                run,shared,values=self.fixture(d);m=json.loads((run/'manifest.json').read_text());raw=run/'freeze_group3.jsonl'
                rows=[json.loads(l) for l in raw.read_text().splitlines()]
                if kind=='matrix':rows[0]['proof']['question_hold']['selected_after']['12']['sha256']='0'*64
                else:rows.pop()
                raw.write_text(''.join(json.dumps(r)+'\n' for r in rows));m['model_state']['file_sha256'][raw.name]=streaming_hash(raw)
                write_json(run/'manifest.json',m)
                with patch.object(analysis,'DESTINATION',shared[0]),patch.object(analysis,'registered_inputs',return_value=values), \
                     patch.object(analysis,'verified_seal',return_value=shared[2]):
                    with self.assertRaises(ValueError):analysis.audit_run(run,True)

    def test_live_json_sources_and_primary_subset_use_same_canonical_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            forward,shared,values=self.fixture(d);m=json.loads((forward/'manifest.json').read_text());live=deepcopy(m)
            for source in live['model_state']['sources'].values():
                layers=source['tail_proof']['gates']['layers'];source['tail_proof']['gates']['layers']={int(k):v for k,v in layers.items()}
            self.assertEqual(runner.source_replay(live['model_state'],self.refs['source_caches']),m['model_state']['predecessor_sources'])
            gate=Path(d)/'gate.json';write_json(gate,{'forward_run':str(forward),'forward_manifest_sha256':streaming_hash(forward/'manifest.json')})
            proof=runner.primary_subset_replay(forward,live,{'path':str(gate)})
            self.assertEqual(sum(proof['readouts_by_arm'].values()),432)
            cache=next(iter(live['model_state']['sources'].values()))['cache'];next(iter(cache['tensors'].values()))['sha256']='0'*64
            with self.assertRaises(ValueError):runner.primary_subset_replay(forward,live,{'path':str(gate)})


if __name__=='__main__':unittest.main()
