"""CPU controls for question-independent cache crossovers and factorial accounting."""
import ast
import copy
import inspect
from pathlib import Path
import unittest
import torch
from sas_memory.runtime import cache_tensors,fork_cache
from tests.test_post_encoding import ToyCache,CharRuntime,prior_index
from scripts.post_encoding_data import build_registry,read_plan as original_plan,sources as original_sources
from scripts.post_encoding_runtime import CacheGuard,assert_isolated
from scripts.post_tail_path_runtime import (compose_cache,compare_components,memory_identity,component,
    CONDITIONS,COMPONENTS)
from scripts.post_tail_path_data import read_plan,sources,selected_contexts,context_queries,MODELS,PRIMARY,READERS
from scripts.post_tail_path_metrics import factorial_values,summarize
from scripts.analyze_post_tail_path import audit_composition,component_differences


def toy_sources(equal=False):
    normal=ToyCache();held=fork_cache(normal)
    if not equal:
        for key,tensor in cache_tensors(held):
            if key[1] in ('keys','values'):tensor[...,1:,:].add_(1000)
            elif key[1]=='recurrent_states':tensor.add_(100)
            else:tensor.add_(200)
    return {'N':normal,'H':held}


class CacheCrossoverTests(unittest.TestCase):
    def test_all_eight_origins_independent_storage_and_immutable_sources(self):
        source=toy_sources();guards={k:CacheGuard(v,torch.ones(1,3)) for k,v in source.items()};memories=[]
        self.assertEqual(list(inspect.signature(compose_cache).parameters),['sources','label'])
        records={k:g.record() for k,g in guards.items()}
        for label in CONDITIONS:
            child=compose_cache(source,label);memories.append(child)
            self.assertEqual(len(dict(cache_tensors(child))),48)
            for key,tensor in cache_tensors(child):
                torch.testing.assert_close(tensor,dict(cache_tensors(source[label[component(key)]]))[key],rtol=0,atol=0)
            for previous in memories[:-1]:assert_isolated(previous,child)
            for g in guards.values():g.check()
            record=CacheGuard(child).record();audit_composition(record,records,label,3,1)
        for child in memories:
            for _,tensor in cache_tensors(child):tensor.add_(1)
        for g in guards.values():g.check()
        self.assertEqual(len({memory_identity(CacheGuard(v).record()) for v in memories}),8)

    def test_source_validation_rejects_alias_bookkeeping_shapes_and_hidden_tensors(self):
        for corruption in ('alias','bookkeeping','shape','untracked','missing','label'):
            src=toy_sources()
            if corruption=='alias':src['H'].layers[0].recurrent_states=src['N'].layers[0].recurrent_states
            elif corruption=='bookkeeping':src['H'].bookkeeping['counter']=1
            elif corruption=='shape':src['H'].layers[0].recurrent_states=[torch.zeros(1,1,2,3)]
            elif corruption=='untracked':src['H'].bookkeeping['extra']=torch.zeros(())
            elif corruption=='missing':del src['H']
            with self.assertRaises(ValueError,msg=corruption):compose_cache(src,'bad' if corruption=='label' else 'HNN')

    def test_historical_prefix_unchanged_and_component_differences_independently_match(self):
        src=toy_sources();actual=compare_components(src,1)
        records={k:CacheGuard(v).record() for k,v in src.items()}
        self.assertEqual(actual,component_differences(records))
        self.assertEqual([len(actual[k]['different_keys']) for k in COMPONENTS],[18,18,12])
        src['H'].layers[23].values[...,0,:].add_(1)
        with self.assertRaisesRegex(ValueError,'prefix KV'):compare_components(src,1)

    def test_equivalent_combinations_retained_as_eight_conditions(self):
        src=toy_sources(equal=True);memories=[compose_cache(src,label) for label in CONDITIONS]
        self.assertEqual(len(memories),8)
        self.assertEqual(len({memory_identity(CacheGuard(v).record()) for v in memories}),1)
        self.assertTrue(all(not v['different_keys'] for v in compare_components(src,1).values()))

    def test_hybrid_record_cannot_borrow_logits_or_change_origin_or_layout(self):
        src=toy_sources();records={k:CacheGuard(v,torch.ones(1,3)).record() for k,v in src.items()}
        good=CacheGuard(compose_cache(src,'HNN')).record();audit_composition(good,records,'HNN',3,1)
        for corruption in ('logits','origin','layout','checks','length','endpoint'):
            bad=copy.deepcopy(good)
            if corruption=='logits':bad['logits_sha256']=records['H']['logits_sha256']
            elif corruption=='origin':bad['tensors'][str((0,'recurrent_states',0))]=records['N']['tensors'][str((0,'recurrent_states',0))]
            elif corruption=='layout':bad['layouts'].pop(next(iter(bad['layouts'])))
            elif corruption=='checks':bad['source_guard_checks']=0
            elif corruption=='length':bad['metadata']['original_metadata']['sequence_length']=4
            else:
                bad=CacheGuard(compose_cache(src,'NNN')).record();bad['sha256']='0'*64
            with self.assertRaises(ValueError,msg=corruption):audit_composition(bad,records,'NNN' if corruption=='endpoint' else 'HNN',3,1)


class FactorialAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,registry,all_queries=build_registry(CharRuntime(),original_plan(),prior_index())
        cls.registry=registry;cls.queries={key:all_queries[key] for c in selected_contexts(registry) for key in c['query_ids']}

    def test_selection_keeps_every_after_context_and_smoke_fixed_subset(self):
        self.assertEqual(len(selected_contexts(self.registry)),32)
        smoke=selected_contexts(self.registry,True);self.assertEqual(len(smoke),4)
        self.assertEqual({q['question_style'] for c in smoke for q in context_queries(c,self.queries,True)},{'original'})
        self.assertEqual(sum(len(context_queries(c,self.queries,True)) for c in smoke),12)
        p=read_plan();self.assertEqual(p['counts']['new_primary_readouts'],len(PRIMARY)*len(self.queries)*len(READERS)*len(MODELS))
        self.assertFalse(p['smoke']['accuracy_gate']);self.assertFalse(p['training'])

    def test_finite_differences_distinguish_main_pair_and_three_way_effects(self):
        for pair,triple in ((0,0),(3,5)):
            values={label:10+sum(v for c,v in zip(label,(1,2,4)) if c=='H')+
                (pair if label[:2]=='HH' else 0)+(triple if label=='HHH' else 0) for label in CONDITIONS}
            result=factorial_values(values)
            self.assertEqual([result[k] for k in ('recurrent','convolution','kv')],[1,2,4])
            self.assertEqual(result['recurrent_convolution'],pair);self.assertEqual(result['three_way'],triple)
            self.assertEqual(result['recurrent_kv'],0);self.assertEqual(result['convolution_kv'],0)
        with self.assertRaises(ValueError):factorial_values({'NNN':1})

    def fixture_rows(self):
        new={};ends={}
        for name in MODELS:
            new[name]=[];ends[name]={k:[] for k in ('NNN','HHH')}
            for label in CONDITIONS:
                for key in self.queries:
                    for reader in READERS:
                        row={'id':key,'condition':label,'reader':reader,'greedy_exact':label!='NNN','answer_nll':2. if label=='NNN' else 1.}
                        (new[name] if label in PRIMARY else ends[name][label]).append(row)
        return new,ends

    def test_all_cells_roles_styles_streams_and_failed_baselines_are_reported(self):
        new,ends=self.fixture_rows();r=summarize(new,ends,self.queries,read_plan())
        self.assertEqual((r['new_primary_readouts'],r['archived_endpoint_references'],r['unified_factorial_readouts']),(4608,1536,6144))
        self.assertEqual(sum(map(len,r['cells'].values())),32)
        c=r['cells']['native']['NNN.full'];self.assertEqual(c['greedy_exact'],0)
        self.assertEqual(c['joint3'],{'correct':0,'total':64});self.assertEqual(c['joint6'],{'correct':0,'total':32})
        self.assertEqual(c['all_questions_per_stream'],{'correct':0,'total':8})
        ep=r['episodes']['native']['HNN.full']['related-64']['all']
        self.assertEqual(ep['joint3'],{'correct':16,'total':16});self.assertEqual(ep['joint6'],{'correct':8,'total':8})
        contrast=r['paired_contrasts']['native']['HNN.full_vs_NNN']
        self.assertEqual(contrast['all']['gain'],192)
        self.assertEqual(contrast['fixed_prefix_evidence']['pairs'],160)
        self.assertEqual(contrast['new_tail_evidence']['pairs'],32)
        self.assertEqual(r['cells']['native']['NNN.full']['measurement_origin'],'archived_endpoint_reference')
        self.assertEqual(r['cells']['native']['HNN.full']['measurement_origin'],'new_primary')
        self.assertEqual(r['baseline_failures_excluded'],0);self.assertFalse(r['architecture_effectiveness_claim'])
        self.assertFalse(r['fresh_confirmation']);self.assertFalse(r['old_gates_can_change'])

    def test_missing_query_or_duplicated_query_cannot_pass_complete_cell(self):
        for change in ('missing','duplicate'):
            new,ends=self.fixture_rows()
            if change=='missing':new['native'].pop()
            else:new['native'][-1]=copy.deepcopy(new['native'][-3])
            with self.assertRaises(ValueError):summarize(new,ends,self.queries,read_plan())

    def test_all214sources_exist_and_contain_complete_local_import_closure(self):
        root=Path(__file__).resolve().parents[1];self.assertEqual(len(sources()),214)
        self.assertEqual(len(original_sources()),206);self.assertTrue(original_sources()<sources())
        for name in sources():
            path=root/name;self.assertTrue(path.is_file(),name)
            if path.suffix!='.py':continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                    [v.name for v in node.names] if isinstance(node,ast.Import) else [])
                for module in modules:
                    target=module.replace('.','/')+'.py'
                    if (root/target).is_file():self.assertIn(target,sources(),f'{name}: {target}')


if __name__=='__main__':unittest.main()
