"""CPU-only exact recovery of already-completed physical readouts."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import finalize_version_query as recovery
from scripts import analyze_version_query_finalized as analysis
from scripts.run_version_query_corrected import correction_descriptor
from scripts.version_query_data import sources
from scripts.run_controller_trial import write_json


class TerminalRecoveryTests(unittest.TestCase):
    def test_all250_sources_preserved_and_four_source_import_closure(self):
        descriptor=recovery.recovery_descriptor(False)
        closure=sources()|set(correction_descriptor()['source_sha256'])|set(descriptor['source_sha256'])
        self.assertEqual(len(closure),254)
        for name in descriptor['source_sha256']:
            if not name.endswith('.py'):continue
            for node in ast.walk(ast.parse((recovery.ROOT/name).read_text())):
                if isinstance(node,ast.ImportFrom) and node.module:
                    target=node.module.replace('.','/')+'.py'
                    if (recovery.ROOT/target).is_file():self.assertIn(target,closure)

    def test_real_complete_failed_run_materializes_without_native_runtime(self):
        descriptor=recovery.recovery_descriptor(False);parent=Path(descriptor['parent_run']);before=recovery.inventory(parent)
        with tempfile.TemporaryDirectory() as d, \
             patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model construction')), \
             patch('transformers.Qwen3_5ForCausalLM.from_pretrained',side_effect=AssertionError('No weight loader')), \
             patch('torch.cuda._lazy_init',side_effect=AssertionError('No GPU init')):
            output=Path(d)/'derived';m=recovery.materialize(output,True)
            self.assertEqual(recovery.check_recovery(output,True),descriptor)
            self.assertEqual(m['error'],recovery.ERROR);self.assertEqual(m['status'],'completed')
            self.assertEqual(sum(s['completed_primary'] for s in m['models'].values()),1728)
            for model in ('native','gdn_all_tokens'):
                self.assertEqual((output/f'{model}.jsonl').read_bytes(),(parent/f'{model}.jsonl').read_bytes())
                self.assertEqual(m['primary_same_subset_replay'][model]['exact_readouts'],54)
        self.assertEqual(recovery.inventory(parent),before)

    def test_raw_mutation_and_unregistered_manifest_change_are_rejected(self):
        for kind in ('raw','cache'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                output=Path(d)/'derived';recovery.materialize(output,True)
                if kind=='raw':
                    raw=output/'native.jsonl';raw.write_bytes(raw.read_bytes()+b'\n')
                else:
                    m=json.loads((output/'manifest.json').read_text());m['models']['native']['prefixes']={}
                    write_json(output/'manifest.json',m)
                with self.assertRaises(ValueError):recovery.check_recovery(output,True)

    def test_missing_recovery_annotation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'derived';recovery.materialize(output,True)
            m=json.loads((output/'manifest.json').read_text());del m['terminal_validation_recovery'];write_json(output/'manifest.json',m)
            with self.assertRaisesRegex(ValueError,'completion fields'):recovery.check_recovery(output,True)

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Never overwrite'):recovery.materialize(Path(d),True)

    def test_full_independent_auditor_is_delegated_without_alteration(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'derived';recovery.materialize(output,True)
            with patch.object(analysis,'prior_audit',return_value={'total_source_files_verified':250,'analysis_source_sha256':'unchanged-prior-auditor'}) as delegate:
                result=analysis.audit_run(output,allow_unsealed=True)
            delegate.assert_called_once_with(output)
            self.assertEqual(result['total_source_files_verified'],254);self.assertFalse(result['native_inference_repeated'])
            self.assertEqual(result['prior_analysis_source_sha256'],'unchanged-prior-auditor')


if __name__=='__main__':unittest.main()
