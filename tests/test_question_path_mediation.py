"""Exercise actual row patches and reject incomplete or contradictory mediation proofs."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace
import unittest
import torch

from scripts.analyze_question_path_mediation import audit_query, audit_vector
from scripts.fresh_layer_controls import first_token, tensor_hash
from scripts.question_path_controls import (PathTrace, PATCHES, TARGETS, CONTEXTS, input_scope,
    sources, audit_receipts)
from scripts.run_write_locality_load_screen import hook_state
from tests import test_fresh_token_timing as timing_tests


class GDN(torch.nn.Module):
    def forward(self, hidden_states): return hidden_states.cumsum(1)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.k_norm=torch.nn.Identity(); self.v_proj=torch.nn.Identity()
    def forward(self, hidden_states, position_embeddings):
        b,n,d=hidden_states.shape
        k=self.k_norm(hidden_states.reshape(b,n,2,d//2));v=self.v_proj(hidden_states)
        return hidden_states + k.flatten(2) + v


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(24)])
        for i in (20,21,22):self.model.layers[i].linear_attn=GDN()
        self.model.layers[23].self_attn=Attention()
    def forward(self, hidden):
        for i in (20,21,22):hidden=hidden+self.model.layers[i].linear_attn(hidden_states=hidden)
        pos=(torch.ones(1,hidden.shape[1],2),torch.zeros(1,hidden.shape[1],2))
        return self.model.layers[23].self_attn(hidden_states=hidden,position_embeddings=pos)


class PathHookTests(unittest.TestCase):
    def test_registered_targets_and_untargeted_rows_are_exact_and_prefix_is_never_patched(self):
        model=Model();q={'prompt_tokens':7,'ledger_tokens':4,'question_tokens':3}
        native=torch.arange(28,dtype=torch.float32).reshape(1,7,4); changed=native.clone();changed[:,4:6]+=5
        hooks=hook_state(model)
        with PathTrace(model,q) as baseline:model(native);model(native[:,:4])
        for patch in PATCHES:
            with PathTrace(model,q,patch,baseline) as trace:model(changed);model(native[:,:4])
            audit_receipts(trace.receipts,trace.input_calls,q,patch,baseline.receipts,dtype='torch.float32')
            for name in TARGETS:
                start,stop=(6,7) if name.startswith('gdn') else (4,6)
                if trace.receipts[name]['patch_enabled']:
                    self.assertTrue(torch.equal(trace.outputs[name][:,start:stop],baseline.outputs[name][:,start:stop]))
                self.assertTrue(trace.receipts[name]['untargeted_rows_exact'])
                self.assertEqual(trace.receipts[name]['prefix_calls'],1)
            self.assertTrue(torch.equal(trace.outputs['k23'][:,-1:],trace.key_before_last))
            self.assertTrue(torch.equal(trace.outputs['v23'][:,-1:],trace.value_before_last))
        self.assertEqual(hook_state(model),hooks)

    def test_self_to_self_shams_and_exception_cleanup(self):
        model=Model();q={'prompt_tokens':7,'ledger_tokens':4,'question_tokens':3};x=torch.arange(28,dtype=torch.float32).reshape(1,7,4)
        hooks=hook_state(model)
        with PathTrace(model,q) as baseline:expected=model(x);model(x[:,:4])
        for patch in PATCHES:
            with PathTrace(model,q,patch,baseline):actual=model(x);model(x[:,:4])
            self.assertTrue(torch.equal(actual,expected))
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with PathTrace(model,q,'both',baseline):raise RuntimeError('injected')
        self.assertEqual(hook_state(model),hooks)
        with self.assertRaises(ValueError):PathTrace(model,q,'both')
        with self.assertRaises(ValueError):PathTrace(model,q,'unknown',baseline)


class MediationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.scope=input_scope()

    def fixture(self,index=0):
        scope=deepcopy(self.scope);plan=scope['plan'];entry=plan['queries'][index];q=scope['queries'][entry['id']]
        tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:str(ids[0]));runtime=SimpleNamespace(tokenizer=tokenizer)
        h,n=q['ledger_tokens'],q['prompt_tokens'];gold=q['answer_ids'][0];competitor=gold+1
        hashof=lambda label:hashlib.sha256(label.encode()).hexdigest()
        native=hashof('native');changed=hashof('changed')
        statekeys=[str((i,field,0)) for i in range(24) if i%4!=3 for field in ('recurrent_states','conv_states')]
        rows=[];proofs=[];vectors={}
        for context,ctx in CONTEXTS.items():
            for patch in PATCHES:
                gdn=bool(PATCHES[patch]['gdn_current_outputs']);kv=PATCHES[patch]['question_kv'];inter=context!='unmasked'
                vector=torch.full((1,64),-10.,dtype=torch.float32);vector[0,competitor]=2.
                vector[0,gold]=3. if not inter or patch=='both' else 2.5 if gdn else 1.5 if kv else 1.
                key=f"{q['id']}--{context}--{patch}";vectors[key]=vector
                first=first_token(runtime,vector,q)
                rows.append({'model':entry['model'],'id':q['id'],'context':context,'patch':patch,
                    'source_regime':entry['regime'],'original_direction':entry['direction'],'token_policy':'question','prefill':'full',
                    'arithmetic_profile':plan['arithmetic_profile'],'registration_sha256':entry['registration_sha256'],
                    'prompt_ids_sha256':q['prompt_ids_sha256'],'mask_registration_sha256':entry['mask_registration_sha256'],
                    'keep_bits_sha256':entry['keep_bits_sha256'],'first_token':first,'logits_key':key,
                    'fixed_competitor_id':competitor,'fixed_pair_margin':vector[0,gold].item()-2.})
                receipts={}
                for name in TARGETS:
                    enabled=gdn if name.startswith('gdn') else kv
                    before=changed if inter else native;after=native if enabled else before
                    receipts[name]={'full_calls':1,'prefix_calls':1,'prefix_output_identity_exact':True,
                        'patch_enabled':enabled,'selected_rows':[n-1] if name.startswith('gdn') else list(range(h,n-1)),
                        'shape':[1,n,2048] if name.startswith('gdn') else [1,n,2,256] if name=='k23' else [1,n,512],
                        'dtype':'torch.bfloat16','target_before_sha256':before,'target_after_sha256':after,
                        'target_reference_sha256':native,'target_equals_reference':after==native,
                        'untargeted_before_sha256':before,'untargeted_after_sha256':before,'untargeted_rows_exact':True,
                        'output_object_identity_exact':not enabled}
                states={key:(hashof('gdn') if inter and gdn and (key.startswith('(21,') or key.startswith('(22,')) else changed if inter and key.startswith('(20,') else native) for key in statekeys}
                earlier={f'{i}:{k}':changed if inter and not kv and i==23 else native for i in (3,7,11,15,19,23) for k in ('keys','values')}
                full_input=native if not inter else hashof('gdn_input') if gdn else hashof('context_input')
                proof={'id':q['id'],'context':context,'patch':patch,'selected_layers':ctx['selected_layers'],
                    'calls':{'prefill':2,'cached':0},'receipts':timing_tests.TimingAuditTests.receipt(ctx['selected_layers'],[n,h]),
                    'batches':timing_tests.TimingAuditTests.batches(q,'earlier_question_tokens',[n,h]),
                    'attention_stats':{i:{'calls':1,'selected_query_rows':q['question_tokens']-1,'other_query_rows':1,
                        'selected_max_suppressed_probability':0. if i in ctx['selected_layers'] else .1,
                        'selected_max_suppressed_mass':0. if i in ctx['selected_layers'] else .2,
                        'other_max_suppressed_probability':.1,'other_max_suppressed_mass':.2,'max_kept_record_mass':.3} for i in (3,7,11,15,19,23)},
                    'patch_receipts':receipts,'input_calls':{'gdn20_full':1,'gdn20_prefix':1,'attention23_full':1,'attention23_prefix':1},
                    'complete_prefix_memory_exact':True,'prefix_logits_exact':True,'prefix_sha256':native,'masked_prefix_sha256':native,
                    'historical_kv_fields_exact':12,'full_question_cache_sha256':hashof(context+patch) if inter else native,
                    'attention23_input_sha256':full_input,'attention23_last_input_sha256':native if not inter or gdn else changed,
                    'attention23_earlier_input_sha256':changed if inter else native,'layer20_last_input_sha256':native,
                    'question_kv23_sha256':{k:changed if inter and not kv else native for k in ('keys','values')},
                    'earlier_position_kv_sha256':earlier,'final_row_kv23_original_projection_exact_fields':2,
                    'recurrent_conv_sha256':states,'gdn20_written_states_unchanged':True,'hooks_restored':True,
                    'kv_patch_recurrent_states_unchanged':True if kv else None,
                    'combined_paths_logits_exact':True if inter and patch=='both' else None}
                proofs.append(proof)
        for reference in plan['prior_references']:
            if (reference['model'],reference['id'])==(entry['model'],q['id']):
                idx=0 if reference['context']=='unmasked' else 4
                reference.update(first_token=rows[idx]['first_token'],**{k:proofs[idx][k] for k in ('prefix_sha256','full_question_cache_sha256','attention23_input_sha256')})
        return [scope,entry,rows,proofs,vectors,native,tokenizer,64]

    def test_all_four_actual_queries_and_both_contexts_retained(self):
        self.assertEqual(len(sources()),125)
        self.assertEqual([q['direction'] for q in self.scope['plan']['queries']],['harm','harm','repair','harm'])
        for i in range(4):
            result=audit_query(*self.fixture(i));self.assertEqual(set(result['effects']),set(CONTEXTS))

    def test_missing_condition_vector_or_wrong_fixed_competitor_rejected(self):
        for part in ('row','proof','vector','competitor'):
            args=self.fixture(2)
            if part=='row':args[2].pop()
            if part=='proof':args[3].pop()
            if part=='vector':args[4].pop(args[2][-1]['logits_key'])
            if part=='competitor':args[2][6]['fixed_competitor_id']+=1
            with self.subTest(part=part),self.assertRaises(ValueError):audit_query(*args)

    def test_target_reference_untargeted_rows_and_prefix_receipts_required(self):
        for field,value in (('target_after_sha256','0'*64),('untargeted_rows_exact',False),('selected_rows',[0]),('prefix_calls',0)):
            args=self.fixture();args[3][5]['patch_receipts']['gdn20'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):audit_query(*args)

    def test_native_prefix_gdn_current_input_questionKV_and_written_states_required(self):
        for field in ('prefix','input20','input23','qKV','state','lastKV'):
            args=self.fixture();p=args[3][6]
            if field=='prefix':p['masked_prefix_sha256']='0'*64
            if field=='input20':p['layer20_last_input_sha256']='0'*64
            if field=='input23':p['attention23_input_sha256']='0'*64
            if field=='qKV':p['question_kv23_sha256']['keys']='0'*64
            if field=='state':p['recurrent_conv_sha256'][str((21,'recurrent_states',0))]='0'*64
            if field=='lastKV':p['final_row_kv23_original_projection_exact_fields']=1
            with self.subTest(field=field),self.assertRaises(ValueError):audit_query(*args)

    def test_raw_full_logits_not_just_argmax_and_pair_scores_are_verified(self):
        for field in ('vector','closure','dtype','first'):
            args=self.fixture();row=args[2][-1];key=row['logits_key']
            if field in ('vector','closure'):
                args[4][key]=args[4][key].clone();args[4][key][0,63]+=.25
                if field=='closure':row['first_token']['logits_sha256']=tensor_hash(args[4][key])
            if field=='dtype':args[4][key]=args[4][key].bfloat16()
            if field=='first':row['first_token']['argmax_text']='wrong'
            with self.subTest(field=field),self.assertRaises(ValueError):audit_query(*args)

    def test_unmasked_sham_and_prior_reference_drift_are_rejected(self):
        for change in ('sham','reference'):
            args=self.fixture()
            if change=='sham':args[3][1]['full_question_cache_sha256']='0'*64
            else:args[2][4]['first_token']['logits_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):audit_query(*args)


if __name__=='__main__':unittest.main()
