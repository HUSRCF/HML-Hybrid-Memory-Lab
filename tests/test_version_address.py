"""Passive native observations, independent arrays and full-grid CPU audit controls."""
import ast
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch

from scripts import version_address_data as data,version_address_runtime as runtime,analyze_version_address as audit
from scripts.run_version_address import replay_sources
from scripts.run_controller_trial import write_json
from scripts.copy_readout_runtime import readout
from scripts.question_state_freeze_runtime import canonical
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.fresh_layer_controls import tensor_hash
from scripts.run_write_locality_load_screen import hook_state
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime


class Cache:
    def __init__(self,length):self.length=length;self.fields=[torch.ones(2) for _ in range(48)]
    def get_seq_length(self):return self.length


class NativeAttention(torch.nn.Module):
    def forward(self,length,past,bad=False):
        weights=torch.zeros((1,8,length,past+length),dtype=torch.bfloat16)
        for h in range(8):
            for t in range(length):
                weights[0,h,t,0]=.5
                weights[0,h,t,1]=(h+1)/1024
                weights[0,h,t,15]=(9-h)/1024
                weights[0,h,t,past+t]=.5-10/1024
        if bad:weights[0,0,11,-1]=.125
        result=(weights.float().sum(-1),weights);self.last=result;return result


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.attentions=torch.nn.ModuleList([NativeAttention() for _ in data.LAYERS]);self.fail=None;self.bad=None
        self.model=SimpleNamespace(layers=[SimpleNamespace(self_attn=self.attentions[data.LAYERS.index(i)]) if i in data.LAYERS else None for i in range(24)])
    def forward(self,input_ids,past_key_values,**kwargs):
        n=input_ids.shape[1];past=past_key_values.get_seq_length();value=0
        for i,module in zip(data.LAYERS,self.attentions):
            if self.fail==i:raise RuntimeError('intentional native failure')
            out,_=module(n,past,self.bad==i);value=value+out
        past_key_values.length+=n
        for f in past_key_values.fields:f.add_(n)
        return SimpleNamespace(past_key_values=past_key_values,logits=value)


class ObserverTests(unittest.TestCase):
    def records(self,n=2):return [{'index':i,'start':1+14*i,'stop':1+14*(i+1)} for i in range(n)]

    def test_actual_eight_head_mean_and_record_sum_match_manual_dyadic_weights(self):
        model=Model();ids=list(range(24));before=hook_state(model)
        with runtime.RecordObservation(model,self.records(),1,29,ids) as obs:
            model(input_ids=torch.tensor([ids]),past_key_values=Cache(29));array,proof=obs.report()
        self.assertEqual(hook_state(model),before);self.assertEqual(array.dtype,np.float32)
        self.assertEqual(array.shape,(6,2,2));self.assertEqual(proof['positions'],[11,23])
        for li,a in enumerate(model.attentions):
            weights=a.last[1]
            for pi,pos in enumerate((11,23)):
                for rec in range(2):
                    exact=sum(float(weights[0,h,pos,j]) for h in range(8) for j in range(1+14*rec,15+14*rec))/8
                    self.assertEqual(float(array[li,pi,rec]),exact)

    def test_native_logits_all48cache_fields_and_return_tensors_are_unchanged(self):
        model=Model();ids=list(range(24));source=Cache(29)
        expected=model(input_ids=torch.tensor([ids]),past_key_values=deepcopy(source))
        with runtime.RecordObservation(model,self.records(),1,29,ids) as obs:
            actual=model(input_ids=torch.tensor([ids]),past_key_values=deepcopy(source));obs.report()
        self.assertTrue(torch.equal(expected.logits,actual.logits))
        for left,right in zip(expected.past_key_values.fields,actual.past_key_values.fields):self.assertTrue(torch.equal(left,right))
        self.assertEqual(expected.past_key_values.length,actual.past_key_values.length)
        self.assertEqual(source.length,29)

    def test_observer_is_removed_before_answer_and_cannot_observe_answer_tokens(self):
        model=Model();before=hook_state(model);ids=list(range(24))
        with runtime.RecordObservation(model,self.records(),1,29,ids) as obs:
            out=model(input_ids=torch.tensor([ids]),past_key_values=Cache(29));saved=obs.report()
            self.assertEqual(hook_state(model),before)
            model(input_ids=torch.tensor([[7]]),past_key_values=out.past_key_values)
            actual=obs.report();self.assertTrue(np.array_equal(saved[0],actual[0]));self.assertEqual(saved[1],actual[1])

    def test_bad_known_ids_future_access_and_native_exception_clean_all_own_hooks(self):
        for reason in ('ids','future','native'):
            model=Model();before=hook_state(model);ids=list(range(24))
            if reason=='future':model.bad=11
            if reason=='native':model.fail=15
            with self.assertRaises((ValueError,RuntimeError)):
                with runtime.RecordObservation(model,self.records(),1,29,ids):
                    model(input_ids=torch.tensor([ids[::-1] if reason=='ids' else ids]),past_key_values=Cache(29))
            self.assertEqual(hook_state(model),before)

    def test_invalid_history_or_known_boundary_rejected_before_hooks(self):
        model=Model();before=hook_state(model)
        for rec,ids,boundary in (([],list(range(24)),29),(self.records(),list(range(23)),29),(self.records(),list(range(24)),28)):
            with self.assertRaises(ValueError):runtime.RecordObservation(model,rec,1,boundary,ids)
            self.assertEqual(hook_state(model),before)


class AddressDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model constructor')), \
             patch('transformers.Qwen3_5ForCausalLM.from_pretrained',side_effect=AssertionError('No model loader')), \
             patch('torch.cuda._lazy_init',side_effect=AssertionError('No GPU init')):
            cls.plan=data.read_plan();cls.cases,cls.registry,cls.queries,cls.refs=data.build_inputs(tokenizer_runtime(cls.plan['model_path']))
        cls.wording=json.loads((Path(cls.plan['wording_run'])/'manifest.json').read_text())

    def test_full_unfiltered_old_and_fresh_inputs_novelty_and_one_token_pairs(self):
        self.assertEqual((len(self.cases),len(self.registry['contexts']),len(self.queries)),(32,288,1728))
        self.assertEqual(sorted({c['seed'] for c in self.cases if c['seed']>=82000}),list(range(82000,82008)))
        self.assertEqual(sum(not r['greedy_exact'] for r in self.refs['normal_full']['gdn_all_tokens']),50)
        for cohort in ('reused','fresh'):
            cs=[c for c in self.registry['contexts'] if self.registry['cohort_by_context'][c['id']]==cohort]
            self.assertEqual(len(cs),144)
        for q in self.queries.values():
            other=self.queries[q['paired_query_id']]
            self.assertEqual([i for i,(x,y) in enumerate(zip(q['input_ids'],other['input_ids'])) if x!=y],[q['ledger_tokens']+5])
        self.assertEqual(len(data.selected_contexts(self.registry,True)),18)

    def test_all261sources_unchanged_and268source_import_closure(self):
        self.assertEqual(len(data.predecessor_sources()),261);self.assertEqual(len(data.sources()),268)
        for name in data.NEW_SOURCES:
            if name.endswith('.py'):
                for node in ast.walk(ast.parse((data.ROOT/name).read_text())):
                    if isinstance(node,ast.ImportFrom) and node.module:
                        path=node.module.replace('.','/')+'.py'
                        if (data.ROOT/path).exists():self.assertIn(path,data.sources())

    def arrays(self,q):
        n=(q['ledger_tokens']-q['prelude_tokens'])//14
        values=np.full((6,2,n),.8/n,dtype=np.float32)
        capture={'question_calls':1,'question_tokens':24,'known_ids_sha256':data_fingerprint(q['input_ids'][q['ledger_tokens']:]),
            'positions':[11,23],'all_own_hooks_removed_before_answers':True,'native_attention_return_not_replaced':True,
            'no_roles_answers_or_support_in_capture_interface':True,'layers':{str(i):{'prelude_mass':[.1,.1],
                'question_mass':[.1,.1],'total_mass':[1.,1.],'native_weights_dtype':'torch.bfloat16',
                'native_weights_shape':[1,8,24,q['ledger_tokens']+24],'pooled_record_mass_sha256':tensor_hash(torch.from_numpy(values[j]))}
                for j,i in enumerate(data.LAYERS)}}
        return values,capture

    def test_independent_raw_aggregation_ties_zero_support_and_noncanonical_number(self):
        q=next(iter(self.queries.values()));c=next(c for c in self.registry['contexts'] if c['id']==q['context_id']);case=next(c for c in self.cases if c['id']==q['case_id'])
        for kind in ('ties','zero','noncanonical'):
            values,capture=self.arrays(q)
            if kind=='zero':
                values[:]=0
                for j,i in enumerate(data.LAYERS):capture['layers'][str(i)]['pooled_record_mass_sha256']=tensor_hash(torch.from_numpy(values[j]))
            generated='00'+q['answer'] if kind=='noncanonical' else q['answer']
            result=runtime.summarize_record_array(values,capture,q,c,case,generated)
            row={'capture':capture,'generated':generated,'address_rows':result}
            audit.verify_observations(values,row,q,c,case)
            if kind=='zero':self.assertEqual(result[0]['support_mass'],0);self.assertEqual(result[0]['support_equal_mass_ties'],len(c['records']))
            if kind=='noncanonical':self.assertEqual(result[0]['output_category'],'noncanonical_correct_integer')
            broken=deepcopy(row);broken['address_rows'][0]['support_mass']+=.1
            with self.assertRaises(ValueError):audit.verify_observations(values,broken,q,c,case)
            changed=values.copy();changed[0,0,0]+=.1
            with self.assertRaises(ValueError):audit.verify_observations(changed,row,q,c,case)

    def test_complete_all216_readout_audits_and_exact_opposite_order_raw_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);packet=folder/'packet';packet.mkdir();contexts=data.selected_contexts(self.registry,True)
            digests={n:streaming_hash(data.ROOT/n) for n in data.sources()};registered={'created_utc':(datetime.now(timezone.utc)+timedelta(seconds=1)).isoformat(),'source_sha256':digests}
            write_json(packet/'manifest.json',registered)
            seal={'path':'cpu-fixture','sha256':'cpu-fixture','created_utc':(datetime.now(timezone.utc)+timedelta(seconds=2)).isoformat(),'full_tests_passed':343}
            for reverse in (False,True):self.fixture(folder/('reverse' if reverse else 'forward'),packet,registered,seal,contexts,reverse)
            with patch.object(audit,'DESTINATION',packet),patch.object(audit,'registered_inputs',return_value=(registered,self.cases,self.registry,self.queries,self.refs)),patch.object(audit,'verified_seal',return_value=seal):
                result=audit.audit_equivalence(folder/'forward',folder/'reverse')
                self.assertEqual(result['raw_fp32_files_exact'],36);self.assertEqual(result['readouts_per_smoke'],216)
                raw=folder/'forward/native.jsonl';rows=[json.loads(l) for l in raw.read_text().splitlines()]
                rows[0]['address_rows'][0]['support_mass']+=.5
                raw.write_text(''.join(json.dumps(r)+'\n' for r in rows));mp=folder/'forward/manifest.json';m=json.loads(mp.read_text());m['models']['native']['results_sha256']=streaming_hash(raw);write_json(mp,m)
                with self.assertRaises(ValueError):audit.audit_run(folder/'forward',True)

    def fixture(self,out,packet,registered,seal,contexts,reverse):
        out.mkdir();p=self.plan;wording=self.wording;now=datetime.now(timezone.utc)
        m={k:deepcopy(wording[k]) for k in ('model_path','packages','generation','arithmetic_profile','actual_eos_token_id',
            'generation_config_sha256','weights_before','weights_after','native_anchors','native_component_kernel_checks')}
        m.update(status='completed',smoke=True,reverse_model_order=reverse,model_order=list(data.MODELS)[::-1] if reverse else list(data.MODELS),
            packet=str(packet),packet_manifest_sha256=streaming_hash(packet/'manifest.json'),protocol_sha256=streaming_hash(data.PROTOCOL),source_sha256=registered['source_sha256'],registration_seal=seal,
            precision_addendum_sha256=streaming_hash(Path(p['precision_addendum'])),validation_addendum_sha256=audit.VALIDATION_ADDENDUM_SHA256,
            serialization_numpy_version=np.__version__,smoke_equivalence_gate=None,primary_same_subset_replay=None,hooks_restored=True,
            started_utc=(now+timedelta(seconds=3)).isoformat(),completed_utc=(now+timedelta(seconds=4)).isoformat(),
            training=False,new_adapters=False,oracle_reading=False,donor_forward=False,changes_recurrent_updates=False,models={})
        for name,value in (('plan',p),('precision_addendum',json.loads(Path(p['precision_addendum']).read_text())),('kernel_validation_addendum',audit.validation_addendum())):write_json(out/(name+'.json'),value)
        (out/'installed_modeling_source.py').write_bytes(Path(p['installed_modeling_source']).read_bytes())
        for name in registered['source_sha256']:
            target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((data.ROOT/name).read_bytes())
        maps={c['id']:c for c in self.cases}
        for name in data.MODELS:
            old=wording['models'][name];s=deepcopy(old)
            s.update(completed_primary=108,reused_reference_replays=54,readout_wall_seconds_with_observers=0.,prefixes={},sources={},record_mass_files={})
            oldctx=[c for c in contexts if self.registry['cohort_by_context'][c['id']]=='reused'];templates={c['episode']:c for c in oldctx}
            oldrows={r['id']:r for r in self.refs['normal_full'][name]};values=[];massdir=out/(name+'.record-masses');massdir.mkdir()
            for c in contexts:
                template=templates[c['episode']];reused=self.registry['cohort_by_context'][c['id']]=='reused'
                if c['episode']=='before':
                    cache=deepcopy(old['prefixes'][template['prefix_key']]);cache['source_guard_checks']=64;s['prefixes'][c['prefix_key']]=cache
                else:
                    source=deepcopy(old['sources'][template['id']]);source['cache']['source_guard_checks']=8;s['sources'][c['id']]=source
                block=[];contextrows=[];relative=name+'.record-masses/'+data_fingerprint(c['id'])+'.npy'
                for qi,key in enumerate(c['query_ids']):
                    q=self.queries[key];array,capture=self.arrays(q);block.append(array)
                    source=oldrows[key] if reused else oldrows[templates[c['episode']]['query_ids'][qi]]
                    value=readout(source) if reused else {**readout(source),'answer':q['answer'],'generated':q['answer'],'greedy_exact':True,'query_tokens':24}
                    proof=deepcopy(source['proof']);proof['reader_keep_sha256']=data_fingerprint(q['history_keep_bits'])
                    row={'id':key,'model':name,'reader':'full','cohort':'reused' if reused else 'fresh',
                        'registration_sha256':data_fingerprint(q),'source_key':q['source_key'],'prompt_ids_sha256':q['prompt_ids_sha256'],'prefill':'cached',
                        'measurement_origin':'normal_history_read_only_record_observation','proof':proof,'capture':capture,
                        'address_rows':runtime.summarize_record_array(array,capture,q,c,maps[q['case_id']],value['generated']),
                        'record_mass_file':relative,'record_mass_index':qi,**value};contextrows.append(row)
                with (out/relative).open('wb') as f:np.save(f,np.stack(block),allow_pickle=False)
                digest=streaming_hash(out/relative);s['record_mass_files'][relative]=digest
                for row in contextrows:row['record_mass_file_sha256']=digest;values.append(row)
            raw=out/(name+'.jsonl');raw.write_text(''.join(json.dumps(canonical(row))+'\n' for row in values));s['results_sha256']=streaming_hash(raw)
            ref=out/(name+'.references.jsonl');ref.write_text(''.join(json.dumps({**r,'archived_reference':True,'source_run':p['wording_run'],
                'source_manifest_sha256':p['wording_manifest_sha256']})+'\n' for r in self.refs['normal_full'][name]));s['references_sha256']=streaming_hash(ref);s['archived_references']=864
            s['predecessor_sources']=replay_sources(s,self.refs['source_caches'][name]);m['models'][name]=s
        write_json(out/'manifest.json',m)


if __name__=='__main__':unittest.main()
