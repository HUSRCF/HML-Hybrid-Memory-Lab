"""Independent executable truths and literal input checks for both complete cohorts."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import hashlib, json, sys
from unittest.mock import patch

ROOT = Path('/home/husrcf/Code/SAS')
BASE = Path('/media/990Pro/Code/SAS')
sys.path.insert(0, str(ROOT))
from scripts.mixture_confirmation_data import registered_inputs, prior_index, check_novelty
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from sas_memory.tasks import Operation, execute
from scripts.run_controller_trial import prompt_for
from sas_memory.training_data import data_fingerprint

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def program(events):
    current, initial = {}, {}
    for event in events:
        if event['kind'] == 'SET':
            value = event['value']
        elif event['kind'] == 'COPY':
            value = current[event['source']]
        else:
            raise ValueError('Unexpected operation')
        current[event['target']] = value
        initial.setdefault(event['target'], value)
    actual = execute([Operation(**event) for event in events])
    assert current == actual['current'] and initial == actual['initial']
    return {'current': current, 'initial': initial}

with (patch('sas_memory.runtime.TextRuntime.__init__', side_effect=AssertionError('No native model')),
      patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU'))):
    p, packet, cohorts, lock = registered_inputs()
    train = json.loads((ROOT / 'configs/readout_adapter_training_plan.json').read_text())
    runtime = tokenizer_runtime(train['model_path'])
    index = prior_index(runtime, excluded=Path(p['dataset']))
    saved = json.loads((Path(p['dataset']) / 'prior-inputs.json').read_text())
    assert index == saved
    assert check_novelty(cohorts['fresh'], index) == packet['novelty']
    counts = {}
    for cohort, data in cohorts.items():
        cases = {c['id']: c for c in data['set_cases']}
        contexts = {c['id']: c for c in data['set_registry']['contexts']}
        for q in data['set_queries'].values():
            case = cases[q['case_id']]
            episode = next(e for e in case['episodes'] if e['id'] == q['episode'])
            state = program(case['prefix_events'] + episode['tail'])
            entity = case['protected'] if q['role'] == 'protected_current' else case['target']
            field = 'initial' if q['role'] == 'target_initial' else 'current'
            assert str(state[field][entity]) == q['answer']
            context = contexts[q['context_id']]
            prompt = context['ledger'] + runtime.query_suffix(q['question'])
            ids = runtime.ids(prompt)
            assert ids == q['input_ids'] and ids[:q['ledger_tokens']] == runtime.ids(context['ledger'])
            assert runtime.continuation_ids(prompt, q['answer']) == q['answer_ids']
            assert data_fingerprint(ids[:q['ledger_tokens']]) == q['source_key']
        cases = {c['id']: c for c in data['copy_cases']}
        for binding in data['copy_registry']['bindings']:
            case = cases[binding['case_id']]
            variant = case['variants'][binding['variant']]
            q = next(q for q in variant['queries'] if q['role'] == binding['role'])
            state = program(case['prefix'] + variant['tail'])
            assert str(state[q['version']][q['entity']]) == q['answer']
            ledger, prompt = prompt_for(runtime, case, binding['variant'], q)
            actual = data['copy_queries'][binding['physical_id']]
            assert q['answer'] == actual['answer'] and runtime.ids(prompt) == actual['input_ids']
            assert data_fingerprint(runtime.ids(ledger)) == actual['source_key']
        assert len(data['query_order']) == len(set(data['query_order'])) == 1200
        counts[cohort] = {'set_questions': 864, 'copy_physical_questions': 336,
                          'copy_logical_bindings': 432, 'normal_histories': 200}
    for name, digest in packet['source_sha256'].items():
        assert sha(ROOT / name) == sha(Path(p['dataset']) / 'sources' / name) == digest

out = BASE / 'artifacts/memory_update/mixture-confirmation-independent-inputs-20261116-01.json'
assert not out.exists()
receipt = {'status': 'all303sources_both1200physical_cohorts_and1296logical_truths_independently_rechecked',
           'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
           'packet_manifest_sha256': sha(Path(p['dataset']) / 'manifest.json'),
           'cohorts': counts, 'actual_prior_input_files': len(index['source_files_sha256']),
           'fixed_fresh_seeds': packet['novelty']['base_seeds'], 'prior_overlap': packet['novelty']['overlap'],
           'native_model_constructed': False, 'gpu_initialized': False,
           'complete_training_locked_before_generation': True}
out.write_text(json.dumps(receipt, indent=2) + '\n')
out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(receipt), flush=True)
