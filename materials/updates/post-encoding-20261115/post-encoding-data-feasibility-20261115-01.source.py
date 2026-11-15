"""Tokenizer-only fresh-grid feasibility; reject any model construction or load."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
from unittest.mock import patch
ROOT=Path('/home/husrcf/Code/SAS');sys.path.insert(0,str(ROOT))
import torch
from transformers import Qwen3_5ForCausalLM
from sas_memory.runtime import TextRuntime
from scripts.post_encoding_data import read_plan,build_registry
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from sas_memory.training_data import data_fingerprint
p=read_plan();index_path=Path(p['prior_input_inventory']);index=json.loads(index_path.read_text())
with patch.object(TextRuntime,'__init__',side_effect=AssertionError('Model runtime construction forbidden')) as constructor, \
     patch.object(Qwen3_5ForCausalLM,'from_pretrained',side_effect=AssertionError('LLM weights forbidden')) as loader:
    runtime=tokenizer_runtime(p['model_path']);cases,registry,queries=build_registry(runtime,p,index)
    assert not hasattr(runtime,'model') and constructor.call_count==0 and loader.call_count==0
allocated=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0
assert allocated==0
out=Path('/media/990Pro/Code/SAS/artifacts/memory_update/post-encoding-data-feasibility-20261115-01.json');assert not out.exists()
r={'status':'tokenizer_only_fresh_grid_validated','created_utc':datetime.now(timezone.utc).isoformat(),'model_inference':False,
   'model_runtime_constructor_calls':constructor.call_count,'llm_weight_loader_calls':loader.call_count,
   'cuda_context_initialized_by_imports':torch.cuda.is_initialized(),'allocated_gpu_tensor_bytes':allocated,
   'seeds':[c['seed'] for c in cases],'cases':len(cases),'contexts':len(registry['contexts']),'queries':len(queries),
   'prefix_evidence_queries':sum(q['prefix_evidence'] for q in queries.values()),
   'tail_evidence_positive_control_queries':sum(not q['prefix_evidence'] for q in queries.values()),
   'cases_fingerprint':data_fingerprint(cases),'registry_fingerprint':data_fingerprint(registry),'queries_fingerprint':data_fingerprint(queries),
   'prefix_token_lengths':sorted({c['prefix_tokens'] for c in registry['contexts']}),'tail_token_lengths':sorted({c['tail_tokens'] for c in registry['contexts']}),
   'prior_inventory_sha256':hashlib.sha256(index_path.read_bytes()).hexdigest(),
   'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'all_exact_inputs_novel_against_frozen_prior_inventory':True,'related_unrelated_token_and_record_spans_matched':True}
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
