import ast
import copy
import json
from pathlib import Path
import unittest

from sas_memory.tasks import Operation,execute
from sas_memory.training_data import data_fingerprint
from scripts.text_competence_factorial_data import (ROOT,MODELS,REGIMES,DATA_SOURCES,sources,read_plan,source_inputs,
    compile_events,control_ids,build_transformations,collision_registry,queries_for,unique_prefixes,actual_prompt,
    tokenizer_runtime,require_exact_full)
from scripts.analyze_text_competence_factorial import audit_identical_inputs,audit_attention,compare_smokes
from scripts.text_competence_factorial_metrics import summarize


class TextFactorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=read_plan();cls.cases,cls.registry,cls.references,_=source_inputs(cls.plan)
        cls.runtime=tokenizer_runtime(cls.plan['model_path'])
        cls.transforms=build_transformations(cls.cases,cls.registry,cls.runtime)
        cls.collisions=collision_registry(cls.transforms,cls.registry)

    def test_online_copy_keeps_snapshot_order_and_every_entity_initial(self):
        events=[Operation('SET','a',10),Operation('SET','b',20),Operation('COPY','b',source='a'),
                Operation('SET','a',99),Operation('COPY','c',source='b'),Operation('COPY','a',source='c')]
        result=compile_events(events)
        self.assertEqual([e.value for e in result],[10,20,10,99,10,10])
        self.assertEqual(execute(result),{'current':{'a':10,'b':10,'c':10},'initial':{'a':10,'b':20,'c':10}})
        swapped=[events[0],events[1],events[3],events[2]]
        self.assertEqual(compile_events(swapped)[-1].value,99)
        with self.assertRaises(KeyError):compile_events([Operation('COPY','x',source='missing')])
        with self.assertRaises(ValueError):compile_events([Operation('SET','x',3),Operation('ADD','x',2)])

    def test_complete_text_truth_boundaries_and_unchanged_full(self):
        self.assertEqual(len(self.transforms),336)
        for original in self.registry['physical']:
            entry=self.transforms[original['id']];self.assertEqual(list(entry),list(REGIMES))
            for k,v in original.items():self.assertEqual(entry['full'][k],v)
            for regime,q in entry.items():
                self.assertEqual(data_fingerprint(q['prompt_ids']),q['prompt_ids_sha256'])
                self.assertEqual(actual_prompt(self.runtime,q),q['ledger_text'])
                self.assertEqual(q['question_ids'],original['question_ids']);self.assertEqual(q['answer_ids'],original['answer_ids'])
                self.assertTrue(all(e['kind']=='SET' for e in q['events']) if q['compiled'] else True)
            for a,b in [('full','compiled_full'),('support_text','compiled_support')]:
                left=[Operation(**e) for e in entry[a]['events']];right=[Operation(**e) for e in entry[b]['events']]
                self.assertEqual(len(left),len(right))
                for end in range(1,len(left)+1):self.assertEqual(execute(left[:end]),execute(right[:end]))
        self.assertEqual(len(self.registry['bindings']),432)

    def test_source_closure_exact_and_all_repo_imports_covered(self):
        self.assertEqual(len(DATA_SOURCES),169);self.assertEqual(len(sources()),172)
        self.assertEqual(len(self.plan['models']),5)
        self.assertEqual(self.plan['models'][MODELS[2]]['checkpoint_sha256'],self.plan['models'][MODELS[3]]['checkpoint_sha256'])
        for name in sources():
            path=ROOT/name;self.assertTrue(path.is_file(),name)
            if path.suffix!='.py':continue
            for node in ast.walk(ast.parse(path.read_text())):
                imports=[]
                if isinstance(node,ast.ImportFrom) and node.module:imports=[node.module]
                if isinstance(node,ast.Import):imports=[v.name for v in node.names]
                for module in imports:
                    relative=module.replace('.','/')+'.py'
                    if (ROOT/relative).is_file():self.assertIn(relative,sources(),f'{name}: {relative}')

    def test_controls_all_depths_styles_loads_and_smoke_plus_auxiliary(self):
        ids=control_ids(self.cases,self.registry);self.assertEqual(len(ids),12)
        bound=[b for b in self.registry['bindings'] if b['physical_id'] in ids]
        self.assertEqual({b['dependency_depth'] for b in bound},{1,2,3})
        self.assertEqual({b['question_style'] for b in bound},{'original','entity_first'})
        self.assertEqual({b['distractor_updates'] for b in bound},{8,64})
        self.assertEqual({b['role'] for b in bound},{'target_current','target_initial','protected_current'})
        for r in REGIMES:
            self.assertEqual(len(queries_for(self.transforms,self.cases,self.registry,r,True)),18)
            allq=queries_for(self.transforms,self.cases,self.registry,r,True,True)
            self.assertEqual(len(allq),20);self.assertTrue(set(ids)<={q['id'] for q in allq})
            self.assertEqual(len(unique_prefixes(allq)),len({q['source_key'] for q in allq}))

    def test_collisions_preserve_all_original_task_ids(self):
        for regime,info in self.collisions['by_regime'].items():
            for keys in info['collisions'].values():
                first=self.transforms[keys[0]][regime]
                for key in keys[1:]:self.assertEqual(self.transforms[key][regime]['prompt_ids'],first['prompt_ids'])
            count=sum(len(v)-1 for v in info['collisions'].values())
            self.assertEqual(info['unique_prompts']+count,336)
        entry=self.registry['physical'][0]['id'];bad=copy.deepcopy(self.transforms[entry]['full']);bad['events'][0]['value']+=1
        with self.assertRaises(ValueError):actual_prompt(self.runtime,bad)

    def test_identical_actual_input_drift_and_full_reference_drift_rejected(self):
        key=next(q['id'] for q in self.registry['physical'] if self.transforms[q['id']]['full']['prompt_ids']==self.transforms[q['id']]['compiled_full']['prompt_ids'])
        value={'id':key,**{k:self.references['native'][0][k] for k in ('answer','answer_nll','generated','greedy_exact','query_tokens')}}
        rows={r:[copy.deepcopy(value)] for r in REGIMES}
        # Deliberately use identical inputs in all four cells to check strict drift rejection.
        transforms={key:{r:self.transforms[key]['full'] for r in REGIMES}}
        audit_identical_inputs(rows,transforms);rows['compiled_full'][0]['answer_nll']+=1e-12
        with self.assertRaises(ValueError):audit_identical_inputs(rows,transforms)
        with self.assertRaises(ValueError):require_exact_full(rows['compiled_full'][0],value,'native',key)

    def test_actual_attention_and_opposite_control_drift_rejected(self):
        stats={str(i):{'prefill_calls':1,'cached_calls':2,'query_rows':10,'max_record_probability':.2,'max_record_mass':.8} for i in (3,7,11,15,19,23)}
        audit_attention(stats);stats['11']['max_record_mass']=0
        with self.assertRaises(ValueError):audit_attention(stats)
        fm={'model_order':list(MODELS),'native_anchors':{},'source_sha256':{},'references':{},'dataset_manifest_sha256':'x',
            'weights_before':{},'weights_after':{},'models':{n:{'attention_controls':{}} for n in MODELS}}
        rm=copy.deepcopy(fm);rm['model_order']=list(MODELS[::-1]);rows={n:{r:[] for r in REGIMES} for n in MODELS}
        compare_smokes(fm,rm,rows,rows);rm['models'][MODELS[0]]['attention_controls']['depth2']='altered'
        with self.assertRaises(ValueError):compare_smokes(fm,rm,rows,rows)

    def test_all_cell_nulls_and_missing_model_rejected(self):
        rows={n:{r:[{'id':q['id'],'answer':q['answer'],'generated':'wrong','greedy_exact':False,'answer_nll':1.,
                    'query_tokens':q['question_tokens'],'actual_prompt_sha256':self.transforms[q['id']][r]['prompt_ids_sha256']}
                    for q in self.registry['physical']] for r in REGIMES} for n in MODELS}
        m=summarize(rows,self.cases,self.registry,self.plan,self.collisions)
        self.assertFalse(m['architecture_effectiveness_claim']);self.assertIsNone(m['success_thresholds'])
        self.assertTrue(all(v['mean_physical_nll_interaction']==0 for v in m['factorial_nll_interactions'].values()))
        del rows[MODELS[-1]]
        with self.assertRaises(ValueError):summarize(rows,self.cases,self.registry,self.plan,self.collisions)


if __name__=='__main__':unittest.main()
