"""Validate complete two-model audit and unchanged model-execution delegation."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import run_query_layer_corrected as wrapper
from scripts.analyze_query_layer_corrected import audit_run
from scripts.query_layer_data import sources
from sas_memory.runtime import TextRuntime


class AuditCorrectionTests(unittest.TestCase):
    def test_complete_actual_two_model_forward_smoke_without_native_inference(self):
        import json
        plan = json.loads(wrapper.CORRECTION.read_text())
        with patch.object(TextRuntime, '__init__', side_effect=AssertionError('No native inference during audit')):
            report = audit_run(Path(plan['allowlisted_pre_correction_run']), True, True)
        self.assertEqual(report['new_primary_readouts_audited'], 288)
        self.assertEqual(report['reference_operational_replays_audited'], 96)
        self.assertEqual(report['single_layer_same_source_replays_audited'], 48)
        self.assertEqual(report['question_only_captures_audited'], 48)
        self.assertEqual(report['total_source_files_verified'], 234)
        self.assertIsNone(report['metrics'])
        self.assertFalse(report['architecture_effectiveness_claim'])

    def test_additional_sources_complete_local_import_closure(self):
        import json
        plan = json.loads(wrapper.CORRECTION.read_text())
        all_sources = sources() | set(plan['additional_sources'])
        self.assertEqual(len(all_sources), 234)
        wrapper.correction_descriptor(False)
        for name in all_sources:
            path = wrapper.ROOT / name
            if path.suffix != '.py': continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                           [v.name for v in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    target = module.replace('.', '/') + '.py'
                    if (wrapper.ROOT / target).is_file(): self.assertIn(target, all_sources, f'{name}: {target}')

    def test_unannotated_subsequent_run_cannot_inherit_forward_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'declare complete'):
                wrapper.check_annotation({'status': 'completed', 'smoke': True}, Path(folder), True)

    def test_wrapper_restores_original_globals_when_execution_fails(self):
        old_gate, old_writer = wrapper.original.checked_gate, wrapper.original.write_json
        with patch.object(wrapper, 'correction_descriptor', return_value={'source_sha256': {}}), \
                patch.object(wrapper.original, 'execute_run', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                wrapper.execute_run(type('Args', (), {'output': Path('/tmp/unused-query-correction-fixture')})())
        self.assertIs(wrapper.original.checked_gate, old_gate)
        self.assertIs(wrapper.original.write_json, old_writer)


if __name__ == '__main__':
    unittest.main()
