from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from scripts.readout_mixture_data import (build_inputs, read_plan, sources, validate_inputs,
                                         exposure_schedule, REGIMES)
from scripts.readout_mixture_runtime import teacher
from scripts.analyze_readout_mixture_training import grade
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.prepare_readout_mixture import main as prepare


class FakeLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(1, 30, 16))
        self.received = None

    def forward(self, input_ids, past_key_values, use_cache, logits_to_keep):
        self.received = input_ids.clone()
        if not use_cache or logits_to_keep != 4:
            raise AssertionError('Four final answer logits required')
        return SimpleNamespace(logits=self.logits[:, :input_ids.shape[1]][:, -4:])


def toy_example(known_length):
    known, answer = [7] * known_length, [1, 2, 3, 4]
    return {'id': 'toy', 'task': 'set' if known_length == 24 else 'copy',
            'ledger_tokens': 351, 'known_ids': known, 'answer_ids': answer,
            'teacher_ids': known + answer[:-1]}


class ReadoutMixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read_plan()
        # Only already observed76000/61000/89000inputs. No160000/161000case construction.
        with (patch('sas_memory.runtime.TextRuntime.__init__', side_effect=AssertionError('No model')),
              patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU')),
              patch('scripts.natural_load_data.make_cases', side_effect=AssertionError('No new SET cases')),
              patch('scripts.fresh_copy_data.make_cases', side_effect=AssertionError('No new COPY cases'))):
            cls.data = build_inputs(tokenizer_runtime(cls.plan['model_path']))

    def test_complete_observed_registry_and_all_exposure_budgets(self):
        d = self.data
        self.assertEqual(len(d['examples']), 1536)
        self.assertEqual(len(d['contexts']), 256)
        self.assertEqual(len(sources()), 296)
        for regime, unique, histories, tokens in (('set_replay', 864, 144, 46656), ('set_copy_mix', 1536, 256, 48816)):
            values = [d['examples'][r['query_id']] for r in d['schedules'][regime]]
            self.assertEqual(len(values), 1728)
            self.assertEqual(len({e['id'] for e in values}), unique)
            self.assertEqual(len({e['context_id'] for e in values}), histories)
            self.assertEqual(sum(len(e['teacher_ids']) for e in values), tokens)
            self.assertEqual(sum(len(e['answer_ids']) for e in values), 6912)
        self.assertEqual(self.plan['confirmation']['set_seeds'], list(range(160000, 160008)))
        self.assertEqual(self.plan['confirmation']['copy_seeds'], list(range(161000, 161008)))
        self.assertEqual(d['set_registry']['contexts'][0]['ledger'], d['set_registry']['contexts'][0]['prefix_text'])

    def test_pairing_preserves_every_set_block_and_explicit_copy_duplicate_weight(self):
        d = self.data
        for block in range(144):
            replay = d['schedules']['set_replay'][block * 12:(block + 1) * 12]
            mix = d['schedules']['set_copy_mix'][block * 12:(block + 1) * 12]
            self.assertEqual([r['query_id'] for r in replay[:6]], [r['query_id'] for r in mix[:6]])
            self.assertEqual([r['query_id'] for r in replay[:6]], [r['query_id'] for r in replay[6:]])
            self.assertEqual([r['query_id'] for r in mix[6:]], d['copy_chunks'][block]['query_ids'])
            self.assertEqual(len({r['context_id'] for r in mix[6:]}), 1)
        copy_exposures = [e for e in d['schedules']['set_copy_mix'] if d['examples'][e['query_id']]['task'] == 'copy']
        counts = Counter(e['query_id'] for e in copy_exposures)
        self.assertEqual((len(copy_exposures), len(counts)), (864, 672))
        self.assertEqual(Counter(counts.values()), {1: 576, 3: 96})
        self.assertEqual(Counter(c['variant'] for c in d['copy_chunks']), {'before': 48, 'control': 48, 'treatment': 48})

    def test_all_copy_snapshot_answers_recomputed_by_independent_plain_program(self):
        root = Path(self.plan['dataset']).parent
        first = json.loads((root / 'fresh-copy-memory-confirmation-v1/cases.json').read_text())
        second = json.loads((root / 'readout-adapter-confirmation-v1/inputs.json').read_text())['copy_cases']
        cases = {c['id']: c for c in first + second}
        for binding in self.data['copy_bindings']:
            c = cases[binding['case_id']]
            variant = c['variants'][binding['variant']]
            current, initial = {}, {}
            for event in c['prefix'] + variant['tail']:
                if event['kind'] == 'SET':
                    value = event['value']
                elif event['kind'] == 'COPY':
                    value = current[event['source']]
                else:
                    self.fail('Unexpected training operation')
                current[event['target']] = value
                initial.setdefault(event['target'], value)
            q = next(q for q in variant['queries'] if q['role'] == binding['role'])
            expected = (initial if binding['role'] == 'target_initial' else current)[q['entity']]
            self.assertEqual(str(expected), self.data['examples'][binding['physical_id']]['answer'])

    def test_exposure_and_target_tampering_are_rejected(self):
        for change in ('missing', 'reorder', 'teacher', 'gold'):
            d = deepcopy(self.data)
            if change == 'missing':
                d['schedules']['set_copy_mix'].pop()
            elif change == 'reorder':
                d['schedules']['set_replay'][:2] = list(reversed(d['schedules']['set_replay'][:2]))
            elif change == 'teacher':
                next(iter(d['examples'].values()))['teacher_ids'].pop()
            else:
                next(iter(d['queries'].values()))['answer'] = '999'
            with self.assertRaises(ValueError):
                validate_inputs(d)
        with self.assertRaises(ValueError):
            exposure_schedule(self.data['set_chunks'], self.data['copy_chunks'], 'unknown')

    def test_variable_teacher_supervises_exactly_four_shifted_targets_and_gradients(self):
        for known in (24, 26, 27):
            torch.manual_seed(1)
            model = FakeLogitModel()
            runtime = SimpleNamespace(model=model, tensor=lambda xs: torch.tensor(xs))
            source = SimpleNamespace(get_seq_length=lambda: 351)
            e = toy_example(known)
            with (patch('scripts.readout_mixture_runtime.regular_cache_clone', return_value=object()),
                  patch('scripts.readout_mixture_runtime.install_teacher_commits') as commit):
                loss, detail = teacher(runtime, None, source, e, True)
            commit.assert_called_once()
            target = torch.tensor(e['answer_ids'])
            expected = F.cross_entropy(model.logits[0, known - 1:known + 3], target)
            self.assertTrue(torch.equal(loss, expected))
            self.assertEqual(model.received.tolist(), [e['teacher_ids']])
            self.assertEqual(detail['cached_teacher_tokens'], known + 3)
            loss.backward()
            self.assertTrue(torch.equal(model.logits.grad[:, :known - 1], torch.zeros_like(model.logits.grad[:, :known - 1])))
            self.assertGreater(model.logits.grad[:, known - 1:known + 3].abs().sum().item(), 0)
            self.assertTrue(torch.equal(model.logits.grad[:, known + 3:], torch.zeros_like(model.logits.grad[:, known + 3:])))

    def test_teacher_rejects_wrong_boundaries_targets_and_keeps_eval_no_grad(self):
        source = SimpleNamespace(get_seq_length=lambda: 351)
        model = FakeLogitModel()
        runtime = SimpleNamespace(model=model, tensor=lambda xs: torch.tensor(xs))
        for kind in ('boundary', 'targets', 'known', 'shift'):
            e = toy_example(26)
            if kind == 'boundary': e['ledger_tokens'] = 350
            elif kind == 'targets': e['answer_ids'] = [1, 2, 3]
            elif kind == 'known': e['known_ids'] = [7] * 25
            else: e['teacher_ids'][0] = 8
            with self.assertRaises(ValueError):
                teacher(runtime, None, source, e, True)
        with (patch('scripts.readout_mixture_runtime.regular_cache_clone', return_value=object()),
              patch('scripts.readout_mixture_runtime.install_teacher_commits') as commit):
            loss, _ = teacher(runtime, None, source, toy_example(27), False)
        self.assertFalse(loss.requires_grad)
        commit.assert_not_called()

    def test_missing_test_seal_precedes_any_packet_build_or_native_runtime(self):
        with (patch('scripts.prepare_readout_mixture.observed_gate', return_value=self.plan),
              patch('scripts.prepare_readout_mixture.verify_seal', side_effect=ValueError('Missing full test seal')),
              patch('scripts.prepare_readout_mixture.build_inputs', side_effect=AssertionError('No input construction')) as builder,
              patch('scripts.prepare_readout_mixture.tokenizer_runtime', side_effect=AssertionError('No tokenizer'))):
            with self.assertRaisesRegex(ValueError, 'full test seal'):
                prepare()
            builder.assert_not_called()

    def test_raw_wrong_or_malformed_answers_are_kept_as_failures(self):
        e = next(iter(self.data['examples'].values()))
        for text in ('', 'The value is ' + e['answer'], '123456'):
            row = {'id': e['id'], 'generated': text, 'answer': e['answer'], 'greedy_exact': False,
                   'cached_teacher_tokens': len(e['teacher_ids']), 'teacher_nll_per_token': 1., 'answer_nll': 1.}
            self.assertFalse(grade(row, e))
            with self.assertRaises(ValueError):
                grade({**row, 'greedy_exact': True}, e)
        good = {'id': e['id'], 'generated': e['answer'] + '\n', 'answer': e['answer'], 'greedy_exact': True,
                'cached_teacher_tokens': len(e['teacher_ids']), 'teacher_nll_per_token': .1, 'answer_nll': .1}
        self.assertTrue(grade(good, e))
        with self.assertRaises(ValueError):
            grade({**good, 'teacher_nll_per_token': float('nan')}, e)


if __name__ == '__main__':
    unittest.main()
