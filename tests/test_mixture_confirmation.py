"""Confirmation contracts use only previously observed76000/61000fixtures, never future cohorts."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.mixture_confirmation_data import (build_inputs, check_novelty, model_spec, sources,
                                               training_gate, MODELS, NEW_MODELS)
from scripts.mixture_confirmation_runtime import archive_exact, write_history
from scripts.analyze_mixture_confirmation import (complete_rows, grade, functional_checks,
                                                 verify_history_coverage)
from scripts.prepare_mixture_confirmation import main as prepare
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


def empty_index():
    return {'seeds': [], 'identifiers': [], 'complete_prompt_ids_sha256': [],
            'complete_prefix_ids_sha256': [], 'source_files_sha256': {}}


def record():
    return {'sha256': 'a' * 64, 'logits_sha256': 'b' * 64, 'metadata': {'position': 351},
            'tensors': {str(i): {'sha256': str(i)} for i in range(48)},
            'layouts': {'r': [1, 16, 128, 128]}}


class MixtureConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = json.loads(Path('configs/readout_adapter_training_plan.json').read_text())
        cls.runtime = tokenizer_runtime(p['model_path'])
        with (patch('sas_memory.runtime.TextRuntime.__init__', side_effect=AssertionError('No model')),
              patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU'))):
            cls.inputs = build_inputs(cls.runtime, list(range(76000, 76008)),
                                      list(range(61000, 61008)), empty_index())

    def test_missing_complete_training_audit_stops_before_tokenizer_and_generation(self):
        with (patch('pathlib.Path.is_file', return_value=False),
              patch('scripts.prepare_mixture_confirmation.source_seal', return_value={}),
              patch('scripts.prepare_mixture_confirmation.tokenizer_runtime',
                    side_effect=AssertionError('No tokenizer')) as tokenizer,
              patch('scripts.prepare_mixture_confirmation.build_inputs',
                    side_effect=AssertionError('No future inputs')) as generator):
            with self.assertRaisesRegex(ValueError, 'audit must finish'):
                training_gate()
            with self.assertRaisesRegex(ValueError, 'audit must finish'):
                prepare()
            tokenizer.assert_not_called()
            generator.assert_not_called()

    def test_all_frozen_fixture_questions_and_bindings_are_retained(self):
        self.assertEqual(len(self.inputs['query_order']), 1200)
        self.assertEqual(len(self.inputs['set_queries']), 864)
        self.assertEqual(len(self.inputs['copy_queries']), 336)
        self.assertEqual(len(self.inputs['copy_registry']['bindings']), 432)
        self.assertEqual(len(self.inputs['set_registry']['contexts']) + len(self.inputs['copy_contexts']), 200)
        self.assertEqual(len(sources()), 303)
        for q in self.inputs['set_queries'].values():
            other = self.inputs['set_queries'][q['paired_query_id']]
            self.assertEqual(sum(a != b for a, b in zip(q['input_ids'], other['input_ids'])), 1)

    def test_actual_novelty_collisions_are_fatal_without_resampling(self):
        index = empty_index()
        index['seeds'] = [76000]
        with patch('scripts.natural_load_data.make_cases', side_effect=AssertionError('No resampling')) as generator:
            with self.assertRaisesRegex(ValueError, 'never resample'):
                build_inputs(self.runtime, list(range(76000, 76008)), list(range(61000, 61008)), index)
            generator.assert_not_called()
        q = next(iter(self.inputs['set_queries'].values()))
        for field, value in (('identifiers', q['case_id']),
                             ('complete_prompt_ids_sha256', data_fingerprint(q['input_ids'])),
                             ('complete_prefix_ids_sha256', q['source_key'])):
            index = empty_index()
            index[field] = [value]
            with self.assertRaisesRegex(ValueError, 'never resample'):
                check_novelty(self.inputs, index)

    def test_all_seven_specs_lock_final432_weights_and_parameter_budget(self):
        lock = {'original_training': {'training_run': '/tmp/old', 'files_sha256': {}, 'final_states': {}},
                'mixture_training': {'training_run': '/tmp/new', 'files_sha256': {}, 'final_states': {}}}
        for name in MODELS:
            if name in ('native', 'fixed_gdn'):
                self.assertEqual(model_spec(name, lock)['parameters'], 0)
                continue
            new = name in NEW_MODELS
            part = lock['mixture_training' if new else 'original_training']
            folder = {'gate_set_replay': 'set_replay', 'gate_set_copy_mix': 'set_copy_mix'}.get(name, name)
            path = str(Path(part['training_run']) / folder / 'step-000432/controller.safetensors')
            part['files_sha256'][path], part['final_states'][folder] = 'a' * 64, 'b' * 64
            spec = model_spec(name, lock)
            self.assertEqual(spec['checkpoint'], path)
            self.assertEqual(spec['checkpoint_sha256'], 'a' * 64)
            self.assertEqual(spec['state_sha256'], 'b' * 64)
            self.assertEqual(spec['parameters'], 131072)
            self.assertEqual(spec['rank'], 16 if name == 'query_gate' else 32)
            self.assertEqual(spec['mode'], 'gate' if new else name)
        with self.assertRaises(ValueError):
            model_spec('best_validation_arm', lock)

    def test_streamed_histories_require_all48_fields_unique_context_and_guards(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'histories.jsonl'
            with path.open('w') as handle:
                write_history(handle, 'known_context', record())
            self.assertEqual(verify_history_coverage(path, ['known_context'], {'known_context': 7}),
                             {'known_context': record()})
            for contexts, guards in ((['missing_context'], {'missing_context': 7}),
                                      (['known_context'], {}),
                                      (['known_context'], {'known_context': 6})):
                with self.assertRaises(ValueError):
                    verify_history_coverage(path, contexts, guards)
            path.write_text(path.read_text() * 2)
            with self.assertRaises(ValueError):
                verify_history_coverage(path, ['known_context'], {'known_context': 7})
            for key in record():
                bad = record()
                bad.pop(key)
                with path.open('w') as handle, self.assertRaises(ValueError):
                    write_history(handle, 'known_context', bad)
            bad = record()
            bad['tensors'].pop('47')
            with path.open('w') as handle, self.assertRaises(ValueError):
                write_history(handle, 'known_context', bad)

    def test_old_raw_files_are_byte_preserved_and_never_overwritten(self):
        with TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'old', Path(tmp) / 'archived'
            source.write_bytes(b'{"generated":"176"}\n\n')
            archive_exact(source, target, streaming_hash(source))
            self.assertEqual(source.read_bytes(), target.read_bytes())
            with self.assertRaises(ValueError):
                archive_exact(source, target, streaming_hash(source))
            with self.assertRaises(ValueError):
                archive_exact(source, Path(tmp) / 'wrong', 'f' * 64)

    def test_every_model_raw_order_and_wrong_answers_are_checked(self):
        order = self.inputs['query_order'][:6]
        rows = [{'id': key, 'model': NEW_MODELS[0]} for key in order]
        complete_rows(rows, order, NEW_MODELS[0])
        for bad in (rows[:-1], rows + rows[:1], list(reversed(rows))):
            with self.assertRaises(ValueError):
                complete_rows(bad, order, NEW_MODELS[0])
        with self.assertRaises(ValueError):
            complete_rows(rows, order, NEW_MODELS[1])
        for text in ('176', 'Balance: 522', ''):
            self.assertFalse(grade({'answer': '522', 'generated': text, 'greedy_exact': False,
                                    'answer_nll': 1.2}, '522'))
        with self.assertRaises(ValueError):
            grade({'answer': '522', 'generated': '522', 'greedy_exact': True, 'answer_nll': float('nan')}, '522')

    def test_fresh_checks_reject_hidden_regressions_and_require_base_stream_gains(self):
        names = ('gate_set_copy_mix', 'gate_set_replay', 'gate', 'fixed_gdn')
        summary = {name: {'set_correct': 10, 'copy_joint6_correct': 3,
                         'copy_joint_by_depth': {'1': 1, '2': 1, '3': 1},
                         'copy_joint_by_seed': {str(s): 0 for s in range(8)}} for name in names}
        mix = summary['gate_set_copy_mix']
        mix['copy_joint6_correct'] = 6
        mix['copy_joint_by_depth'] = {'1': 2, '2': 2, '3': 2}
        mix['copy_joint_by_seed'].update({'0': 1, '1': 1, '2': 1})
        protected = {name: {'p'} for name in names}
        joint = {name: {'j'} for name in names}
        self.assertTrue(functional_checks(summary, protected, joint)['all_passed'])
        bad = deepcopy(protected)
        bad['gate_set_copy_mix'] = {'replacement'}
        self.assertFalse(functional_checks(summary, bad, joint)['passed']['set_retention'])
        bad_joint = deepcopy(joint)
        bad_joint['gate_set_copy_mix'] = {'replacement'}
        self.assertFalse(functional_checks(summary, protected, bad_joint)['passed']['strong_copy_reference'])
        bad = deepcopy(summary)
        bad['gate_set_copy_mix']['copy_joint_by_depth']['3'] = 0
        self.assertFalse(functional_checks(bad, protected, joint)['passed']['copy_joint_gain'])
        bad = deepcopy(summary)
        bad['gate_set_copy_mix']['copy_joint_by_seed']['2'] = 0
        self.assertFalse(functional_checks(bad, protected, joint)['passed']['copy_joint_gain'])


if __name__ == '__main__':
    unittest.main()
