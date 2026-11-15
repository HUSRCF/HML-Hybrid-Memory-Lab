"""Reused complete question grid and immutable single-layer registration."""
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.fresh_query_data import sources as prior_sources, registered_inputs as prior_inputs
from scripts.post_encoding_data import selected_contexts, context_queries

PROTOCOL = ROOT / 'configs/query_layer_localization_plan.json'
PROTOCOL_SHA256 = '967a7693dd3f3c4ff2cb9e71353742e68297b460495b12f35ce3c5d7ec35aa56'
DESTINATION = Path('/media/990Pro/Code/SAS/datasets/memory_update/query-layer-localization-v1')
MODELS = ('native', 'gdn_all_tokens')
LAYERS = (3, 7, 11, 15, 19, 23)
DIRECTIONS = ('HN', 'NH')
ORIGINS = ('N', 'H')
REFERENCES = ('NN', 'NH', 'HN', 'HH')
NEW_SOURCES = {'configs/query_layer_localization_plan.json', 'scripts/query_layer_data.py',
    'scripts/prepare_query_layer.py', 'scripts/query_layer_runtime.py', 'scripts/run_query_layer.py',
    'scripts/query_layer_metrics.py', 'scripts/analyze_query_layer.py', 'tests/test_query_layer.py'}


def sources():
    return prior_sources() | NEW_SOURCES


def read_plan():
    if streaming_hash(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Registered single-layer protocol changed')
    p = json.loads(PROTOCOL.read_text())
    if (p['model_order'] != list(MODELS) or p['attention_layers'] != list(LAYERS) or
            p['query_directions'] != list(DIRECTIONS) or p['reader_modes'] != ['full'] or
            p['counts']['new_primary_readouts'] != 4608 or p['counts']['exact_reference_operational_replays'] != 1536 or
            p['counts']['same_source_single_layer_replays'] != 384 or
            p['counts']['question_only_capture_forwards'] != 768 or
            p['smoke']['new_primary_readouts_each'] != 288 or p['smoke']['accuracy_gate'] or
            p['operational_controls']['source_files_planned'] != 230 or
            set(p['operational_controls']['new_sources']) != NEW_SOURCES or
            Path(p['registration_destination']) != DESTINATION or
            any(p[k] for k in ('training', 'new_adapters', 'new_models', 'new_environment',
                'fresh_confirmation', 'can_change_prior_gates', 'architecture_effectiveness_claim',
                'new_code_started', 'data_packet_generated', 'model_inference_started'))):
        raise ValueError('Complete registered single-layer scope required')
    return p


def self_control_ids(contexts, queries, smoke):
    """Two IDs per stream, chosen without consulting any output."""
    result = set()
    for seed in dict.fromkeys(c['seed'] for c in contexts):
        ids = sorted(q['id'] for c in contexts if c['seed'] == seed and c['episode'] != 'before'
                     for q in context_queries(c, queries, smoke))
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError('Complete nonduplicated after-question cohort required')
        result.update((ids[0], ids[-1]))
    return result


def reference_inputs():
    p = read_plan()
    run = Path(p['source_run'])
    m = json.loads((run / 'manifest.json').read_text())
    a = json.loads(Path(p['source_audit']).read_text())
    pub = json.loads(Path(p['source_publication']).read_text())
    if (streaming_hash(run / 'manifest.json') != p['source_manifest_sha256'] or
            streaming_hash(Path(p['source_audit'])) != p['source_audit_sha256'] or
            streaming_hash(Path(p['source_publication'])) != p['source_publication_sha256'] or
            a['status'] != 'completed_and_audited_fresh_query' or a['primary_readouts_audited'] != 3264 or
            a['same_source_replays_audited'] != 1536 or a['actual_donor_query_tensors_audited'] != 9216 or
            a['source_files_verified'] != 222 or any(a['prior_failed_criteria_unchanged'].values()) or
            m['status'] != 'completed' or m['weights_before'] != m['weights_after'] or not m['hooks_restored'] or
            pub['status'] != 'published_remote_assets_roundtrip_byte_verified' or
            pub['audit_sha256'] != p['source_audit_sha256'] or pub['release_url'] != p['source_release'] or
            pub['target_commit'] != p['source_release_target'] or len(pub['remote_assets']) != 3 or
            not all(v['downloaded_bytes_verified'] for v in pub['remote_assets']) or set(m['source_sha256']) != prior_sources()):
        raise ValueError('Complete published and unchanged predecessor required')
    packet, cases, registry, queries, legacy = prior_inputs(Path(p['source_packet']))
    if streaming_hash(Path(p['source_packet']) / 'manifest.json') != p['source_packet_manifest_sha256']:
        raise ValueError('Frozen source packet changed')
    old = json.loads((run / 'plan.json').read_text())
    if any(p[k] != old[k] for k in ('models', 'model_order', 'model_path', 'packages', 'generation',
            'native_loaded_weights_sha256', 'precision_addendum', 'arithmetic_profile', 'cache_origins')):
        raise ValueError('Keep existing models, arithmetic and cache origins')
    refs = {**legacy, 'source_caches': {}, 'full_readouts': {}, 'full_donors': {}}
    for name, state in m['models'].items():
        for source, digest in m['source_sha256'].items():
            if streaming_hash(ROOT / source) != digest or streaming_hash(run / 'sources' / source) != digest:
                raise ValueError('All222predecessor sources must remain unchanged')
        raw = {}
        for kind in ('primary', 'same-source', 'donors'):
            path = run / f'{name}.{kind}.jsonl'
            if streaming_hash(path) != state['results_sha256'][kind]:
                raise ValueError('Predecessor raw bytes changed')
            raw[kind] = [json.loads(line) for line in path.read_text().splitlines()]
        refs['source_caches'][name] = {k: state[k] for k in ('prefixes', 'sources')}
        refs['full_readouts'][name] = [r for r in raw['primary'] if r['reader'] == 'full' and r['condition'] != 'before']
        refs['full_donors'][name] = [r for r in raw['donors'] if r['reader'] == 'full']
        if len(refs['full_readouts'][name]) != 768 or len(refs['full_donors'][name]) != 384:
            raise ValueError('Every after-question Full reference and donor required')
    return cases, registry, queries, refs


def registered_inputs(path=DESTINATION):
    path = Path(path)
    p = read_plan()
    m = json.loads((path / 'manifest.json').read_text())
    if (m['status'] != 'registered' or m['model_inference'] or m['training'] or m['fresh_confirmation'] or
            m['protocol_sha256'] != PROTOCOL_SHA256 or m['counts'] != p['counts'] or
            set(m['source_sha256']) != sources() or len(sources()) != 230 or
            not datetime.fromisoformat(p['registered_utc']) < datetime.fromisoformat(m['created_utc'])):
        raise ValueError('Complete230source packet must precede inference')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / 'sources' / name) != digest:
            raise ValueError('Sealed single-layer source changed')
    for name, digest in m['payload_sha256'].items():
        if streaming_hash(path / name) != digest:
            raise ValueError('Registered input bytes changed')
    values = reference_inputs()
    for name, value in zip(('cases', 'registry', 'queries', 'references'), values):
        if json.loads((path / (name + '.json')).read_text()) != value:
            raise ValueError('Independent reused-input reconstruction differs')
    if json.loads((path / 'plan.json').read_text()) != p:
        raise ValueError('Frozen protocol changed')
    return (m, *values)
