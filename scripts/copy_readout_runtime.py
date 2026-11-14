"""Actual softmax observations and complete historical-memory preservation proofs."""

from sas_memory.kv_readout import QuestionKVReadout
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_load_screen import hook_state
from scripts.semantic_order_path_common import assert_complete_cache_equal, cache_fingerprint


def readout(row):
    return {k: row[k] for k in ("answer", "answer_nll", "greedy_exact", "generated", "query_tokens")}


class SelectionWeights:
    """Observe kept and suppressed original columns in all six eager layers."""

    def __init__(self, model, masks, prelude):
        if not masks:
            raise ValueError("Nonempty mask mapping required")
        self.masks, self.prelude = masks, prelude
        self.boundary = len(next(iter(masks.values())))
        if not 0 < prelude < self.boundary or any(len(m) != self.boundary or not all(m[:prelude]) or
            not any(m[prelude:]) or all(m) for m in masks.values()):
            raise ValueError("Nonempty retained and suppressed record columns required")
        self.stats = {r: {} for r in masks}
        self.handles = [model.register_forward_pre_hook(self.before, with_kwargs=True)]
        for index, layer in enumerate(model.model.layers):
            if model.config.layer_types[index] == "full_attention":
                for r in masks:
                    self.stats[r][index] = {"prefill_calls": 0, "cached_calls": 0, "query_rows": 0,
                        "max_suppressed_probability": 0., "max_suppressed_mass": 0., "max_kept_record_mass": 0.}
                self.handles.append(layer.self_attn.register_forward_hook(self.after(index)))
        if any(len(s) != 6 for s in self.stats.values()):
            self.close()
            raise ValueError("Expected all six softmax Attention layers")

    def before(self, module, args, kwargs):
        if args:
            raise ValueError("Keyword inputs required")
        cache = kwargs.get("past_key_values")
        self.current = (cache.get_seq_length() if cache is not None else 0, kwargs["input_ids"].shape[1])

    def after(self, index):
        def observe(module, args, result):
            import torch
            weights = result[1]
            past, length = self.current
            if not isinstance(weights, torch.Tensor) or weights.ndim != 4 or weights.shape[-2:] != (length, past + length):
                raise ValueError("Unexpected actual eager softmax layout")
            start = max(0, self.boundary - past)
            if start >= length:
                return
            history = weights[..., start:, :self.boundary]
            for r, keep in self.masks.items():
                blocked = torch.tensor([i for i, bit in enumerate(keep) if not bit], device=weights.device)
                retained = torch.tensor([i for i in range(self.prelude, self.boundary) if keep[i]], device=weights.device)
                suppressed = history.index_select(-1, blocked)
                selected = history.index_select(-1, retained)
                stat = self.stats[r][index]
                stat["prefill_calls" if past == 0 else "cached_calls"] += 1
                stat["query_rows"] += length - start
                stat["max_suppressed_probability"] = max(stat["max_suppressed_probability"], suppressed.abs().max().item())
                stat["max_suppressed_mass"] = max(stat["max_suppressed_mass"], suppressed.float().sum(-1).abs().max().item())
                stat["max_kept_record_mass"] = max(stat["max_kept_record_mass"], selected.float().sum(-1).abs().max().item())
        return observe

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def check_weight_stats(stats, restricted):
    import math
    if set(map(str, stats)) != {str(i) for i in (3, 7, 11, 15, 19, 23)}:
        raise ValueError("Complete six-layer probability observations required")
    for s in stats.values():
        if (s["prefill_calls"] != 1 or s["cached_calls"] < 1 or s["query_rows"] < 1 or
            any(not math.isfinite(s[k]) for k in ("max_suppressed_probability", "max_suppressed_mass", "max_kept_record_mass")) or
            s["max_kept_record_mass"] <= 0 or
            (restricted and (s["max_suppressed_probability"] != 0 or s["max_suppressed_mass"] != 0)) or
            (not restricted and (s["max_suppressed_probability"] <= 0 or s["max_suppressed_mass"] <= 0))):
            raise ValueError("Suppressed/retained actual softmax probabilities failed the prescribed control")


def preserve_history(runtime, ledger, prompt, keep, prefix, prefix_logits, full, prefix_sha):
    import torch
    with QuestionKVReadout(runtime.model, keep) as intervention:
        actual_prefix, logits, count = runtime.advance(ledger)
        assert_complete_cache_equal(prefix, actual_prefix)
        actual_sha = cache_fingerprint(actual_prefix)
        if not torch.equal(logits, prefix_logits) or count != len(keep) or actual_sha != prefix_sha:
            raise ValueError("Restricted mask changed complete prefix memory or logits")
        actual_full, _, _ = runtime.advance(prompt)
        fields = 0
        for expected, actual in zip(full.layers, actual_full.layers):
            for name in ("keys", "values"):
                left, right = getattr(expected, name, None), getattr(actual, name, None)
                if isinstance(left, torch.Tensor):
                    if not isinstance(right, torch.Tensor) or not torch.equal(left[..., :len(keep), :], right[..., :len(keep), :]):
                        raise ValueError("Masked full-question forward changed historical K/V")
                    fields += 1
        if fields != 12:
            raise ValueError("Missing historical key/value slices")
    return {"complete_prefix_memory_exact": True, "prefix_logits_exact": True,
        "prefix_sha256": prefix_sha, "masked_prefix_sha256": actual_sha,
        "historical_kv_fields_exact": fields, "proof_hook_calls": intervention.calls}


def prove_and_measure(runtime, ledger, query, masks, expected):
    import torch
    hooks = hook_state(runtime.model)
    prompt = ledger + runtime.query_suffix(query["question"])
    with SelectionWeights(runtime.model, masks, query["prelude_tokens"]) as observer:
        baseline = measure(runtime, ledger, query)
    if baseline != readout(expected):
        differences = {k: {"expected": expected[k], "actual": v}
                       for k, v in baseline.items() if expected[k] != v}
        raise ValueError(f"Actual unmasked reference readout changed: query={query['id']}, differences={differences}")
    if hook_state(runtime.model) != hooks:
        raise ValueError("Actual unmasked observer cleanup changed")
    for stats in observer.stats.values():
        check_weight_stats(stats, False)
    full, full_logits, _ = runtime.advance(prompt)
    prefix, prefix_logits, count = runtime.advance(ledger)
    if count != query["ledger_tokens"]:
        raise ValueError("Actual source ledger boundary changed")
    prefix_sha = cache_fingerprint(prefix)
    with QuestionKVReadout(runtime.model, [True] * count) as all_history:
        actual_full, logits, _ = runtime.advance(prompt)
        assert_complete_cache_equal(full, actual_full)
        if not torch.equal(logits, full_logits) or measure(runtime, ledger, query) != baseline:
            raise ValueError("All-history mask changed full cache, logits or exact readout")
    proof = {"id": query["id"], "unmasked_reference_exact": True, "unmasked_readout": baseline,
        "unmasked_weights": observer.stats, "all_history_cache_logits_readout_exact": True,
        "all_history_hook_calls": all_history.calls, "regimes": {}}
    results = {}
    for regime, keep in masks.items():
        state = preserve_history(runtime, ledger, prompt, keep, prefix, prefix_logits, full, prefix_sha)
        with QuestionKVReadout(runtime.model, keep) as intervention:
            with SelectionWeights(runtime.model, {regime: keep}, query["prelude_tokens"]) as observer:
                actual = measure(runtime, ledger, query)
        check_weight_stats(observer.stats[regime], True)
        if intervention.calls["prefill"] != 1 or intervention.calls["cached"] < 1 or hook_state(runtime.model) != hooks:
            raise ValueError("Restricted full/cached execution or hook cleanup changed")
        state.update(attention_stats=observer.stats[regime], readout=actual, measure_hook_calls=intervention.calls)
        proof["regimes"][regime], results[regime] = state, actual
    proof["hooks_restored"] = hook_state(runtime.model) == hooks
    return proof, results
