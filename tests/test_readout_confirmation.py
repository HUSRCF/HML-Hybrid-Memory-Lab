from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.readout_confirmation_data import (build_inputs, check_novelty, prior_index,
                                               sources, training_gate)
from scripts.analyze_readout_confirmation import grade, complete_rows
from scripts.readout_confirmation_runtime import compare_history
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.prepare_readout_confirmation import main as prepare
from sas_memory.training_data import data_fingerprint


def empty_index():
    return {'seeds': [], 'identifiers': [], 'complete_prompt_ids_sha256': [],
            'complete_prefix_ids_sha256': [], 'source_files_sha256': {}}


class ReadoutConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = json.loads(Path('configs/readout_adapter_training_plan.json').read_text())
        cls.runtime = tokenizer_runtime(p['model_path'])
        old = Path(p['source_packet'])
        cls.old_queries = json.loads((old / 'queries.json').read_text())
        cls.copy_packet = Path('/media/990Pro/Code/SAS/datasets/memory_update/fresh-copy-memory-confirmation-v1')
        # ONLY previously observed76000/61000fixtures,never the prospective88000/89000cohorts.
        with (patch('sas_memory.runtime.TextRuntime.__init__', side_effect=AssertionError('No model construction')),
              patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU initialization'))):
            cls.inputs = build_inputs(cls.runtime, list(range(76000, 76008)),
                                      list(range(61000, 61008)), empty_index())

    def test_missing_full_training_gate_precedes_all_case_generation(self):
        with (patch('pathlib.Path.is_file', return_value=False),
              patch('scripts.prepare_readout_confirmation.source_seal', return_value={}),
              patch('scripts.natural_load_data.make_cases', side_effect=AssertionError('No cases')) as generator):
            with self.assertRaisesRegex(ValueError, 'audit must finish'):
                training_gate()
            with self.assertRaisesRegex(ValueError, 'audit must finish'):
                prepare()
            generator.assert_not_called()

    def test_frozen_set_fixtures_match_every_literal_known_question_and_pair(self):
        queries = self.inputs['set_queries']
        self.assertEqual(len(queries), 864)
        self.assertEqual(len(self.inputs['set_registry']['contexts']), 144)
        for key, query in queries.items():
            self.assertEqual(query, self.old_queries[key])
            other = queries[query['paired_query_id']]
            self.assertEqual(sum(a != b for a, b in zip(query['input_ids'], other['input_ids'])), 1)
            self.assertEqual(len(query['input_ids']) - query['ledger_tokens'], 24)
        self.assertEqual(len(sources()), 289)

    def test_frozen_copy_fixtures_preserve_all_bindings_and_shared_ledgers(self):
        old_cases = json.loads((self.copy_packet / 'cases.json').read_text())
        old_registry = json.loads((self.copy_packet / 'registry.json').read_text())
        self.assertEqual(self.inputs['copy_cases'], old_cases)
        self.assertEqual(self.inputs['copy_registry'], old_registry)
        self.assertEqual(len(self.inputs['copy_contexts']), 56)
        self.assertTrue(all(len(c['query_ids']) == 6 for c in self.inputs['copy_contexts']))
        self.assertEqual(len(self.inputs['query_order']), 1200)
        self.assertEqual(check_novelty(self.inputs, empty_index())['physical_prompts'], 1200)

    def test_novelty_rejects_seed_before_generator_and_all_actual_collision_types(self):
        index = empty_index()
        index['seeds'] = [88000]
        with patch('scripts.natural_load_data.make_cases', side_effect=AssertionError('No resampling')) as generator:
            with self.assertRaisesRegex(ValueError, 'never resample'):
                build_inputs(self.runtime, list(range(88000, 88008)), list(range(89000, 89008)), index)
            generator.assert_not_called()
        q = next(iter(self.inputs['set_queries'].values()))
        for field, value in (('identifiers', q['case_id']), ('complete_prompt_ids_sha256', data_fingerprint(q['input_ids'])),
                             ('complete_prefix_ids_sha256', q['source_key'])):
            index = empty_index()
            index[field] = [value]
            with self.assertRaisesRegex(ValueError, 'never resample'):
                check_novelty(self.inputs, index)
        bad = deepcopy(self.inputs)
        bad['query_order'][-1] = bad['query_order'][0]
        with self.assertRaises(ValueError):
            check_novelty(bad, empty_index())

    def test_actual_inventory_includes_run_training_cases_and_literal_fit_hashes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, runs = root / 'data', root / 'runs'
            packet, run = data / 'registered', runs / 'trained'
            packet.mkdir(parents=True)
            run.mkdir(parents=True)
            q = next(iter(self.inputs['set_queries'].values()))
            (packet / 'manifest.json').write_text('{}')
            (packet / 'examples.json').write_text(json.dumps([{'literal_prompt_ids_sha256': 'f' * 64,
                                                             'seed': q['seed'], 'input_ids': q['input_ids'],
                                                             'ledger_tokens': q['ledger_tokens']}]))
            case = self.inputs['copy_cases'][0]
            (run / 'data.json').write_text(json.dumps({'train': [{'case': case}]}))
            with (patch('sas_memory.runtime.TextRuntime.__init__', side_effect=AssertionError('No model')),
                  patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU'))):
                index = prior_index(self.runtime, data, runs, root / 'excluded')
            self.assertIn(str(run / 'data.json'), index['source_files_sha256'])
            self.assertIn(case['seed'], index['seeds'])
            self.assertIn(q['seed'], index['seeds'])
            self.assertIn('f' * 64, index['complete_prompt_ids_sha256'])
            self.assertIn(q['source_key'], index['complete_prefix_ids_sha256'])
            self.assertEqual(index['all_dataset_and_run_standard_cases_reconstructed'], 1)

    def test_raw_coverage_rejects_missing_duplicate_reordered_and_wrong_models(self):
        order = self.inputs['query_order'][:6]
        rows = [{'id': key, 'model': 'query'} for key in order]
        complete_rows(rows, order, 'query')
        for bad in (rows[:-1], rows + rows[:1], list(reversed(rows))):
            with self.assertRaises(ValueError):
                complete_rows(bad, order, 'query')
        with self.assertRaises(ValueError):
            complete_rows(rows, order, 'gate')

    def test_all_history_fields_are_required_for_frozen_prefix_identity(self):
        original = {'sha256': 'a' * 64, 'logits_sha256': 'b' * 64, 'metadata': {'position': 351},
                    'tensors': {str(i): {'sha256': str(i)} for i in range(48)}, 'layouts': {'r': [1, 16, 128, 128]}}
        compare_history(original, deepcopy(original))
        for key in original:
            changed = deepcopy(original)
            changed[key] = None
            with self.assertRaises(ValueError):
                compare_history(changed, original)

    def test_wrong_numeric_and_malformed_answers_remain_failures_not_accuracy_gates(self):
        for text in ('176', 'The balance is 176.', ''):
            row = {'generated': text, 'answer': '522', 'greedy_exact': False, 'answer_nll': 1.2}
            self.assertFalse(grade(row, '522'))
        row = {'generated': ' 522\n', 'answer': '522', 'greedy_exact': True, 'answer_nll': 1.2}
        self.assertTrue(grade(row, '522'))
        with self.assertRaises(ValueError):
            grade({**row, 'greedy_exact': False}, '522')
        with self.assertRaises(ValueError):
            grade({**row, 'answer_nll': float('nan')}, '522')


if __name__ == '__main__':
    unittest.main()
