"""Fresh-cohort execution controls, separate from every sealed prior runtime."""

import json
from pathlib import Path

from sas_memory.kv_readout import QuestionKVReadout
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.copy_readout_runtime import readout
from scripts.fresh_copy_data import sources, prior_index, read_plan
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_load_screen import hook_state
from scripts.semantic_order_path_common import assert_complete_cache_equal


def verify_prior_index(path, registration):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    actual = prior_index(tokenizer_runtime(read_plan()['model_path']))
    if (actual != json.loads((Path(path) / 'prior_inputs.json').read_text()) or
        data_fingerprint(actual) != registration['prior_inputs_sha256']):
        raise ValueError('Independent reconstruction of actual prior novelty inputs differs')
    return {'actual_prior_index_sha256': data_fingerprint(actual),
        'registered_directories': len(actual['registered_directories']),
        'actual_prior_files': len(actual['source_files_sha256']),
        'complete_prompts': len(actual['complete_prompt_ids_sha256']),
        'complete_prefixes': len(actual['complete_prefix_ids_sha256'])}


def full_only_control(runtime, ledger, query, expected):
    import torch
    hooks = hook_state(runtime.model)
    baseline = measure(runtime, ledger, query)
    if baseline != readout(expected): raise ValueError('Fresh unmasked exact repeat changed')
    prompt = ledger + runtime.query_suffix(query['question'])
    full, logits, tokens = runtime.advance(prompt)
    with QuestionKVReadout(runtime.model, [True] * query['ledger_tokens']) as intervention:
        actual, actual_logits, actual_tokens = runtime.advance(prompt)
        assert_complete_cache_equal(full, actual)
        if tokens != actual_tokens or not torch.equal(logits, actual_logits) or measure(runtime, ledger, query) != baseline:
            raise ValueError('Fresh full-only all-history cache/logits/readout control changed')
    if intervention.calls['prefill'] != 2 or intervention.calls['cached'] < 1 or hook_state(runtime.model) != hooks:
        raise ValueError('Fresh all-history hook execution/cleanup changed')
    return {'id': query['id'], 'unmasked_reference_exact': True, 'unmasked_readout': baseline,
        'all_history_cache_logits_readout_exact': True, 'all_history_hook_calls': intervention.calls,
        'hooks_restored': True, 'regimes': {}, 'unmasked_weights': {}}


def checked_smoke_equivalence(path, dataset_digest):
    from scripts.compare_fresh_copy_smokes import audit_equivalence
    path = Path(path); report = json.loads(path.read_text())
    if (report != audit_equivalence(Path(report['forward_audit']), Path(report['reverse_audit'])) or
        report['dataset_manifest_sha256'] != dataset_digest or
        streaming_hash(path.with_suffix('.source.py')) != report['analysis_source_sha256']):
        raise ValueError('Both independent252-readout fresh smokes and exact order equivalence required')
    return {'path': str(path), 'sha256': streaming_hash(path)}
