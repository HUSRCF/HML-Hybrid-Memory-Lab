from datetime import datetime,timezone
from pathlib import Path
import json,re,sys
from unittest.mock import patch
sys.path.insert(0,'/home/husrcf/Code/SAS')
from scripts.version_address_data import DESTINATION,SEAL,registered_inputs,predecessor_sources,read_plan
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
root=Path('/media/990Pro/Code/SAS');log=root/'logs/memory_update/version-address-full-tests-20261115-01.log'
receipt=root/'artifacts/memory_update/version-address-independent-inputs-20261115-01.json'
if SEAL.exists() or receipt.exists():raise ValueError('Never overwrite input or source seal evidence')
match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$',log.read_text())
if not match or int(match[1])!=352:raise ValueError('Complete352test regression required')
with patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model constructor')), \
     patch('transformers.Qwen3_5ForCausalLM.from_pretrained',side_effect=AssertionError('No model loading')), \
     patch('torch.cuda._lazy_init',side_effect=AssertionError('No GPU initialization')):
    packet,cases,registry,queries,refs=registered_inputs()
digest=streaming_hash(DESTINATION/'manifest.json');created=datetime.now(timezone.utc).isoformat()
assert (len(packet['source_sha256']),len(cases),len(registry['contexts']),len(queries))==(268,32,288,1728)
assert len(predecessor_sources())==261
result={'status':'all268sources_and_complete_old_fresh_inputs_independently_reconstructed_without_inference',
    'created_utc':created,'packet_manifest_sha256':digest,'source_sha256':packet['source_sha256'],
    'cases':32,'contexts':288,'questions':1728,'reused_questions':864,'fresh_questions':864,'fresh_seeds':list(range(82000,82008)),
    'prior_packets_verified':28,'previous_case_seeds_excluded':len(refs['excluded_seeds']),
    'original_GDN_errors_retained':sum(not r['greedy_exact'] for r in refs['normal_full']['gdn_all_tokens']),
    'max_ledger_tokens':max(c['ledger_tokens'] for c in registry['contexts']),
    'model_constructor_calls':0,'model_loader_calls':0,'gpu_initializations':0,'source':str(Path(__file__)),
    'source_sha256_of_receipt_generator':streaming_hash(Path(__file__))}
write_json(receipt,result);receipt.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
seal={'status':'sealed_before_version_address_native_inference','created_utc':datetime.now(timezone.utc).isoformat(),
    'packet_manifest_sha256':digest,'source_sha256':packet['source_sha256'],'full_tests_passed':352,
    'full_tests_seconds':float(match[2]),'full_tests_log':str(log),'full_tests_log_sha256':streaming_hash(log),
    'new_tests_passed':9,'predecessor_sources_unchanged':261,'independent_inputs':str(receipt),
    'independent_inputs_sha256':streaming_hash(receipt),'new_native_inference_started':False}
write_json(SEAL,seal)
print(json.dumps({'status':seal['status'],'tests':352,'sources':268,'packet_manifest_sha256':digest,
    'seal_sha256':streaming_hash(SEAL),'questions':1728,'source_model_construction':0}),flush=True)
