"""Native readout and complete-cache checks for semantic-tail interventions."""

import re

from sas_memory.training_data import data_fingerprint


def selected_pairs(registry, smoke):
    if not smoke:
        return registry["pairs"]
    first = {}
    for pair in registry["pairs"]:
        first.setdefault((pair["cohort"], pair["stratum"]), pair)
    return list(first.values())


def replay_pair_ids(registry, pairs, smoke):
    return {pair["id"] for pair in pairs} if smoke else set(registry["replay_pair_ids"])


def score_worlds(runtime, ledger, cache, query, prefill="cached"):
    worlds = query["world_answers"]
    candidates = list(dict.fromkeys(worlds.values()))
    result = runtime.probe(ledger, cache, query["question"], worlds["left"], candidates,
                           generate=True, prefill=prefill, score_batch=False)
    match = re.fullmatch(r"\s*(-?\d+)\s*", result["generated"])
    scores = {}
    for side, answer in worlds.items():
        score = result["candidate_scores"][answer]
        if score["tokens"] != 3:
            raise ValueError("Expected exactly three gold answer tokens")
        scores[side] = {"answer": answer, "answer_nll": score["nll_mean"],
                        "greedy_exact": bool(match and match.group(1) == answer)}
    return {"generated": result["generated"], "query_tokens": result["query_tokens"], "world_scores": scores}


def same_readout(left, right):
    return (left["generated"] == right["generated"] and left["query_tokens"] == right["query_tokens"] and
            left["world_scores"].keys() == right["world_scores"].keys() and all(
                left["world_scores"][side]["answer"] == right["world_scores"][side]["answer"] and
                left["world_scores"][side]["greedy_exact"] == right["world_scores"][side]["greedy_exact"] and
                abs(left["world_scores"][side]["answer_nll"] - right["world_scores"][side]["answer_nll"]) <= 1e-5
                for side in left["world_scores"]))


def checked_ledger(runtime, case, prefix, query):
    from scripts.run_controller_trial import prompt_for
    ledger, prompt = prompt_for(runtime, case, prefix["cell"], {"question": query["question"]})
    ids = runtime.ids(ledger)
    if (len(ids) != prefix["ledger_tokens"] or data_fingerprint(ids) != prefix["ledger_ids_sha256"] or
        runtime.ids(prompt)[:len(ids)] != ids):
        raise ValueError("Actual ledger-only source differs from frozen registration")
    return ledger


def assert_cache_layout(cache, ledger_tokens):
    from sas_memory.runtime import cache_tensors
    tensors = dict(cache_tensors(cache))
    if len(cache.layers) != 24 or cache.get_seq_length() != ledger_tokens:
        raise ValueError("Incomplete source cache layers or sequence bookkeeping")
    fields = {index: set() for index in range(24)}
    for (index, field, _), tensor in tensors.items():
        fields[index].add(field)
        if tensor.shape[0] != 1:
            raise ValueError("Source cache batch size changed")
        if field in ("keys", "values") and tensor.shape[-2] != ledger_tokens:
            raise ValueError("Source cache contains question/answer tokens or truncated KV")
    if any(fields[index] != ({"keys", "values"} if index % 4 == 3 else {"conv_states", "recurrent_states"})
           for index in fields):
        raise ValueError("Missing recurrent, convolution or KV source tensors")
    return {"layers": 24, "GDN_recurrent_layers": 18, "GDN_convolution_layers": 18,
            "KV_layers": 6, "tensor_count": len(tensors), "sequence_length": ledger_tokens}


def supporting_kv_equal(originals, support_span):
    import torch
    from sas_memory.runtime import cache_tensors
    left, right = [dict(cache_tensors(originals[side])) for side in ("left", "right")]
    start, end = support_span
    checked = []
    for key, tensor in left.items():
        if key[1] in ("keys", "values"):
            if not torch.equal(tensor[..., start:end, :], right[key][..., start:end, :]):
                raise AssertionError("A causally earlier supporting-record KV changed across tail worlds")
            checked.append(list(key))
    if len(checked) != 12 or {key[0] for key in checked} != {3, 7, 11, 15, 19, 23}:
        raise AssertionError("Supporting evidence KV check missed a full-attention layer")
    return {"support_span": support_span, "checked_KV_tensors": checked, "exact": True}


class HistoryWeights:
    """Inspect actual cached question/scoring/generation access at six layers."""

    def __init__(self, model, prelude, boundary, keep=None):
        self.prelude, self.boundary, self.keep = prelude, boundary, keep
        self.stats, self.current = {}, None
        self.handles = [model.register_forward_pre_hook(self.before, with_kwargs=True)]
        for index, layer in enumerate(model.model.layers):
            if model.config.layer_types[index] == "full_attention":
                self.stats[index] = {"prefill_calls": 0, "cached_calls": 0, "query_rows": 0,
                                     "max_retained_mass": 0., "max_suppressed_probability": 0.,
                                     "max_suppressed_mass": 0.}
                self.handles.append(layer.self_attn.register_forward_hook(self.after(index)))

    def before(self, model, args, kwargs):
        if args:
            raise ValueError("Keyword native model calls required")
        cache = kwargs.get("past_key_values")
        self.current = (int(cache.get_seq_length()) if cache is not None else 0, kwargs["input_ids"].shape[1])

    def after(self, index):
        def observe(module, args, result):
            import torch
            weights = result[1]
            past, length = self.current
            if not isinstance(weights, torch.Tensor) or weights.shape[-2:] != (length, past + length):
                raise ValueError("Unexpected eager attention probability layout")
            first = max(0, self.boundary - past)
            if first >= length:
                return
            record = weights[..., first:, self.prelude:self.boundary]
            keep = torch.tensor(self.keep[self.prelude:self.boundary], device=weights.device) if self.keep is not None else torch.ones(record.shape[-1], dtype=torch.bool, device=weights.device)
            if not keep.any():
                raise ValueError("Vacuous retained evidence observation")
            stat = self.stats[index]
            stat["prefill_calls" if past == 0 else "cached_calls"] += 1
            stat["query_rows"] += length - first
            stat["max_retained_mass"] = max(stat["max_retained_mass"], record[..., keep].float().sum(-1).max().item())
            if (~keep).any():
                suppressed = record[..., ~keep]
                stat["max_suppressed_probability"] = max(stat["max_suppressed_probability"], suppressed.abs().max().item())
                stat["max_suppressed_mass"] = max(stat["max_suppressed_mass"], suppressed.float().sum(-1).abs().max().item())
        return observe

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def validate(self):
        if set(self.stats) != {3, 7, 11, 15, 19, 23} or any(
            stat["prefill_calls"] != 0 or stat["cached_calls"] < 1 or stat["query_rows"] < 1 or
            stat["max_retained_mass"] <= 0 or stat["max_suppressed_probability"] != 0 or
            stat["max_suppressed_mass"] != 0 for stat in self.stats.values()):
            raise ValueError("Actual cached history access or oracle suppression safety failed")
