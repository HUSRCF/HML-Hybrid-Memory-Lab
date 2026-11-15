"""Independent literal-data and paired-exposure check, never constructing a native model."""
from collections import Counter, OrderedDict
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import hashlib, json, sys

ROOT=Path('/home/husrcf/Code/SAS')
BASE=Path('/media/990Pro/Code/SAS')
ART=BASE/'artifacts/memory_update'
sys.path.insert(0,str(ROOT))
from scripts.readout_mixture_data import registered_inputs
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from sas_memory.training_data import data_fingerprint

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def program(events):
    current,initial={},{}
    for event in events:
        if event['kind']=='SET':
            value=event['value']
        elif event['kind']=='COPY':
            value=current[event['source']]
        else:
            raise ValueError('Unexpected operation')
        current[event['target']]=value
        initial.setdefault(event['target'],value)
    return current,initial

with (patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No native model')),
      patch('torch.cuda._lazy_init',side_effect=AssertionError('No GPU'))):
    p,packet,data=registered_inputs()
    runtime=tokenizer_runtime(p['model_path'])
    root=Path(p['dataset']).parent
    old_queries=json.loads((root/'version-address-observation-v1/queries.json').read_text())
    old_cases={c['id']:c for c in json.loads((root/'version-address-observation-v1/cases.json').read_text())}
    old_registry=json.loads((root/'version-address-observation-v1/registry.json').read_text())
    set_contexts=[c for c in old_registry['contexts'] if old_queries[c['query_ids'][0]]['seed'] in range(76000,76008)]
    set_chunks=[{'context_id':c['id'],'query_ids':c['query_ids']} for c in set_contexts]
    assert set_chunks==data['set_chunks'] and len(set_contexts)==144
    seen=set()
    for context in set_contexts:
        for qid in context['query_ids']:
            q,e=old_queries[qid],data['examples'][qid]
            case=old_cases[q['case_id']]
            tail=next(ep['tail'] for ep in case['episodes'] if ep['id']==q['episode'])
            current,initial=program(case['prefix_events']+tail)
            entity=case['protected'] if q['role']=='protected_current' else case['target']
            answer=str((initial if q['role']=='target_initial' else current)[entity])
            assert e['answer']==answer==q['answer']
            assert e['known_ids']==q['input_ids'][q['ledger_tokens']:]
            seen.add(qid)
    first=root/'fresh-copy-memory-confirmation-v1'
    second=json.loads((root/'readout-adapter-confirmation-v1/inputs.json').read_text())
    cohorts=[(json.loads((first/'cases.json').read_text()),json.loads((first/'registry.json').read_text())),
             (second['copy_cases'],second['copy_registry'])]
    copy_chunks=[]
    for cases,registry in cohorts:
        cases={c['id']:c for c in cases}
        physical={q['id']:q for q in registry['physical']}
        groups=OrderedDict()
        for binding in registry['bindings']:
            key=(binding['base_case_id'],binding['dependency_depth'],binding['variant'])
            groups.setdefault(key,[]).append(binding)
            case=cases[binding['case_id']]
            variant=case['variants'][binding['variant']]
            current,initial=program(case['prefix']+variant['tail'])
            q=next(q for q in variant['queries'] if q['role']==binding['role'])
            qid=binding['physical_id'];e=data['examples'][qid]
            assert e['answer']==str((initial if binding['role']=='target_initial' else current)[q['entity']])
            assert e['known_ids']==physical[qid]['question_ids']
            seen.add(qid)
        for key,bindings in groups.items():
            qids=[b['physical_id'] for b in bindings]
            cids={data['examples'][qid]['context_id'] for qid in qids}
            assert len(qids)==6 and len(cids)==1
            copy_chunks.append({'context_id':next(iter(cids)),'query_ids':qids,
                                'base_case_id':key[0],'dependency_depth':key[1],'variant':key[2]})
    assert copy_chunks==data['copy_chunks'] and seen==set(data['examples']) and len(seen)==1536
    for qid,e in data['examples'].items():
        c=data['contexts'][e['context_id']]
        prompt=c['ledger']+runtime.query_suffix(e['question'])
        ids=runtime.ids(prompt)
        assert ids[:c['ledger_tokens']]==runtime.ids(c['ledger'])
        assert ids[c['ledger_tokens']:]==e['known_ids']
        assert data_fingerprint(ids)==e['literal_prompt_ids_sha256']
        assert runtime.continuation_ids(prompt,e['answer']+'\n')==e['answer_ids']
        assert e['teacher_ids']==e['known_ids']+e['answer_ids'][:-1]
    totals={}
    for regime in ('set_replay','set_copy_mix'):
        expected=[]
        for block in range(144):
            for part,chunk in enumerate((set_chunks[block],set_chunks[block] if regime=='set_replay' else copy_chunks[block])):
                for qid in chunk['query_ids']:
                    expected.append({'exposure_id':f'{regime}:{len(expected):04d}','query_id':qid,
                                     'context_id':chunk['context_id'],'block':block,'part':part})
        assert expected==data['schedules'][regime] and len(expected)==1728
        totals[regime]={'exposures':1728,'unique_physical_questions':len({e['query_id'] for e in expected}),
                       'teacher_input_tokens':sum(len(data['examples'][e['query_id']]['teacher_ids']) for e in expected),
                       'supervised_answer_tokens':6912}
    assert [totals[r]['teacher_input_tokens'] for r in totals]==[46656,48816]

receipt={'status':'all1536literal_inputs_and3456paired_exposures_independently_rechecked',
         'completed_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
         'packet_manifest_sha256':sha(Path(p['dataset'])/'manifest.json'),
         'physical_truths_checked':1536,'copy_logical_truths_checked':864,'normal_histories':256,
         'regimes':totals,'native_model_constructed':False,'gpu_initialized':False,
         'new_cases_generated':False,'future_confirmation_generated':False}
out=ART/'readout-mixture-independent-inputs-20261116-01.json'
assert not out.exists()
out.write_text(json.dumps(receipt,indent=2)+'\n')
out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(receipt),flush=True)
