"""Patch one question Query; preserve actual native Query objects elsewhere."""
from contextlib import ExitStack
from scripts.fresh_query_runtime import (QuestionQueryExchange, ArgumentGuard, clone_layout,
    tensor_record, ReaderContext, open_reader, reader_proof)
from scripts.post_encoding_runtime import ATTENTION_LAYERS
from scripts.copy_readout_runtime import readout
from scripts.run_write_locality_load_screen import hook_state


class SingleLayerQueryExchange(QuestionQueryExchange):
    def __init__(self, model, boundary, question_ids, donor, layer):
        if type(layer) is not int or layer not in ATTENTION_LAYERS or donor.reader != 'full':
            raise ValueError('One registered layer and Full-only donor required')
        self.layer = layer
        super().__init__(model, boundary, question_ids, 'replace', donor)

    def wrapped(self, module, query, key, value, attention_mask, **kwargs):
        index = module.layer_idx
        if index not in self.stats or self.current is None:
            raise ValueError('Untracked native eager layer/call')
        past, rows, first = self.current
        stat = self.stats[index]
        if (query.ndim != 4 or query.shape[-2] != rows or key.shape[-2] != past + rows or
                value.shape[-2] != past + rows):
            raise ValueError('Actual native Query/KV boundary differs')
        args = (key, value, attention_mask, kwargs)
        guard, qguard = ArgumentGuard(args), ArgumentGuard(query)
        stat['eager_calls'] += 1
        used = query
        if first:
            if stat['question_calls']:
                raise ValueError('Duplicate question invocation in one layer')
            stat.update(question_calls=1, question_rows=rows, native_query=tensor_record(query))
            self.donor.verify()
            if index == self.layer:
                source = self.donor.queries[index]
                if source.shape != query.shape or source.dtype != query.dtype or source.device != query.device:
                    raise ValueError('Query donor shape/dtype/device differs')
                used = clone_layout(query)
                used.copy_(source)
                if (used.stride() != query.stride() or used.storage_offset() != query.storage_offset() or
                        used.untyped_storage().data_ptr() in
                        (query.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())):
                    raise ValueError('Replacement must preserve native layout with isolated storage')
            stat.update(used_query=tensor_record(used), actual_replacement=index == self.layer,
                        original_query_object_preserved=used is query)
        else:
            stat['passthrough_calls'] += 1
            if used is not query:
                raise ValueError('Decode Query must be the actual original object')
        usedguard = ArgumentGuard(used)
        result = self.original(module, used, key, value, attention_mask, **kwargs)
        stat['non_query_tensor_checks'] += guard.check(args)
        qguard.check(query)
        usedguard.check(used)
        stat['native_query_checks'] += 1
        self.donor.verify()
        return result

    def report(self):
        result = super().report()
        if ([i for i, value in self.stats.items() if value['actual_replacement']] != [self.layer] or
                any(value['original_query_object_preserved'] != (i != self.layer) for i, value in self.stats.items())):
            raise ValueError('Exactly one copied Query and five original Query objects required')
        return {**result, 'selected_layer': self.layer, 'unselected_question_queries_native': True}


def measure_single_layer(runtime, context, cache, query, donor, layer):
    if (donor.reader != 'full' or donor.prompt_sha != query['prompt_ids_sha256'] or
            context['id'] != query['context_id'] or cache.get_seq_length() != query['ledger_tokens']):
        raise ValueError('Registered Full-only same-prompt donor and recipient required')
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    active, eager = hook_state(runtime.model), native.eager_attention_forward
    ids = query['input_ids'][query['ledger_tokens']:]
    rc = ReaderContext('full', query['history_keep_bits'], query['prelude_tokens'])
    with ExitStack() as stack:
        restricted, observer = open_reader(stack, runtime.model, rc)
        exchange = stack.enter_context(SingleLayerQueryExchange(runtime.model, query['ledger_tokens'], ids, donor, layer))
        value = readout(runtime.probe(context['ledger'], cache, query['question'], query['answer'],
            [query['answer']], generate=True, prefill='cached', score_batch=False))
        proof = reader_proof(rc, restricted, observer)
        proof['query_exchange'] = exchange.report()
    if (hook_state(runtime.model) != active or native.eager_attention_forward != eager or
            value['query_tokens'] != len(ids)):
        raise ValueError('Single-layer hook/eager cleanup or question length changed')
    return value, proof
