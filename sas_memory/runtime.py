"""Local text-only Qwen runtime with explicit, isolated mixed-cache forks."""

from __future__ import annotations

import copy
from collections.abc import Mapping
import re
import time

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM


def fork_cache(cache, repeats: int = 1):
    """Clone KV, recurrent, convolution, and bookkeeping state before a probe.

    Transformers 5.14's linear cache does not implement batch_repeat_interleave;
    repeat its state containers explicitly instead of silently repeating only KV.
    """
    result = copy.deepcopy(cache)
    if repeats != 1:
        for layer in result.layers:
            for field in ("keys", "values"):
                tensor = getattr(layer, field, None)
                if isinstance(tensor, torch.Tensor):
                    setattr(layer, field, tensor.repeat_interleave(repeats, dim=0))
            for field in ("recurrent_states", "conv_states"):
                states = getattr(layer, field, None)
                if states is not None:
                    for index, tensor in state_items(states):
                        if isinstance(tensor, torch.Tensor):
                            states[index] = tensor.repeat_interleave(repeats, dim=0)
    return result


def cache_tensors(cache):
    for layer_index, layer in enumerate(cache.layers):
        for field in ("keys", "values"):
            tensor = getattr(layer, field, None)
            if isinstance(tensor, torch.Tensor):
                yield (layer_index, field, 0), tensor
        for field in ("recurrent_states", "conv_states"):
            for index, tensor in state_items(getattr(layer, field, ())):
                if isinstance(tensor, torch.Tensor):
                    yield (layer_index, field, index), tensor


def assert_cache_equal(left, right):
    left_tensors, right_tensors = dict(cache_tensors(left)), dict(cache_tensors(right))
    if left_tensors.keys() != right_tensors.keys():
        raise AssertionError("Cache structure changed")
    for key in left_tensors:
        if not torch.equal(left_tensors[key], right_tensors[key]):
            raise AssertionError(f"Probe modified its source cache: {key}")


class TextRuntime:
    def __init__(self, path: str, model_kind: str, backend: str = "auto"):
        self.path, self.model_kind, self.backend = path, model_kind, backend
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
        self.model, loading = Qwen3_5ForCausalLM.from_pretrained(
            path, config=config, dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="eager", local_files_only=True, output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Text checkpoint did not load completely: {loading}")
        self.model.eval().requires_grad_(False)
        if backend == "reference":
            from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
            for layer in self.model.model.layers:
                if hasattr(layer, "linear_attn"):
                    gdn = layer.linear_attn
                    gdn.chunk_gated_delta_rule = implementation.torch_chunk_gated_delta_rule
                    gdn.recurrent_gated_delta_rule = implementation.torch_recurrent_gated_delta_rule
        if model_kind == "it":
            marker = "SAS_UNIQUE_CONTENT_MARKER"
            template = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": marker}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            self.prompt_start, self.prompt_end = template.split(marker)
        else:
            self.prompt_start, self.prompt_end = "", ""

    def ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def continuation_ids(self, prefix: str, suffix: str) -> list[int]:
        prefix_ids, whole_ids = self.ids(prefix), self.ids(prefix + suffix)
        if whole_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Tokenization crosses the cached segment boundary; choose another boundary")
        return whole_ids[len(prefix_ids):]

    def tensor(self, ids):
        return torch.tensor(ids, dtype=torch.long, device=self.model.device)

    @torch.inference_mode()
    def advance(self, text: str, cache=None, prefix_text: str = ""):
        ids = self.continuation_ids(prefix_text, text) if cache is not None else self.ids(text)
        if not ids:
            raise ValueError("Cannot advance an empty token sequence")
        result = self.model(
            input_ids=self.tensor([ids]), past_key_values=fork_cache(cache) if cache is not None else None,
            use_cache=True, logits_to_keep=1,
        )
        return result.past_key_values, result.logits[:, -1].float(), len(ids)

    @torch.inference_mode()
    def candidate_scores(self, prompt: str, query_cache, first_logits, candidates: list[str], batch: bool = True):
        groups: dict[int, list] = {}
        for candidate in candidates:
            ids = self.continuation_ids(prompt, candidate)
            if not ids:
                raise ValueError("Empty candidate")
            groups.setdefault(len(ids), []).append((candidate, ids))
        scores = {}
        first_logp = first_logits.log_softmax(-1)
        for length, group in groups.items():
            chunks = [group] if batch else [[item] for item in group]
            for chunk in chunks:
                ids = self.tensor([item[1] for item in chunk])
                losses = -first_logp[0, ids[:, 0]]
                if length > 1:
                    result = self.model(
                        input_ids=ids[:, :-1], past_key_values=fork_cache(query_cache, len(chunk)),
                        use_cache=True, logits_to_keep=length - 1,
                    )
                    logp = result.logits.float().log_softmax(-1)
                    losses += -logp.gather(-1, ids[:, 1:, None]).squeeze(-1).sum(-1)
                for (candidate, token_ids), loss in zip(chunk, losses.tolist()):
                    scores[candidate] = {"nll_sum": loss, "tokens": len(token_ids), "nll_mean": loss / len(token_ids)}
        return scores

    @torch.inference_mode()
    def greedy_answer(self, cache, first_logits, max_tokens: int = 12) -> str:
        work = fork_cache(cache)
        logits, generated = first_logits, []
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for _ in range(max_tokens):
            token = logits.argmax(-1).item()
            if token in eos:
                break
            generated.append(token)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if "\n" in text and text.strip():
                break
            result = self.model(input_ids=self.tensor([[token]]), past_key_values=work, use_cache=True, logits_to_keep=1)
            work, logits = result.past_key_values, result.logits[:, -1].float()
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def query_suffix(self, question: str) -> str:
        return f"Question: {question}\nAnswer:" + self.prompt_end

    @torch.inference_mode()
    def probe(self, ledger: str, cache, question: str, answer: str, candidates: list[str], generate: bool = True,
              prefill: str = "full", score_batch: bool = False):
        suffix = self.query_suffix(question)
        prompt = ledger + suffix
        if prefill == "full":
            query_cache, logits, _ = self.advance(prompt)
            count = len(self.continuation_ids(ledger, suffix))
        elif prefill == "cached":
            query_cache, logits, count = self.advance(suffix, cache, ledger)
        else:
            raise ValueError(prefill)
        scores = self.candidate_scores(prompt, query_cache, logits, candidates, batch=score_batch)
        ranked = min(scores, key=lambda key: scores[key]["nll_sum"])
        text = self.greedy_answer(query_cache, logits) if generate else None
        match = re.fullmatch(r"\s*(-?\d+)\s*", text) if text is not None else None
        return {
            "answer": answer, "answer_nll": scores[answer]["nll_mean"],
            "candidate_prediction": ranked, "candidate_correct": ranked == answer,
            "candidate_scores": scores, "generated": text,
            "greedy_exact": bool(match and match.group(1) == answer) if generate else None,
            "query_tokens": count,
        }

    @torch.inference_mode()
    def verify(self, ledger: str, cache, question: str, candidates: list[str]) -> dict:
        """Check the machinery before interpreting any model errors as memory errors."""
        saved = fork_cache(cache)
        suffix = self.query_suffix(question)
        query_cache, logits, _ = self.advance(suffix, cache, ledger)
        assert_cache_equal(cache, saved)
        full = self.model(input_ids=self.tensor([self.ids(ledger + suffix)]), use_cache=True, logits_to_keep=1)
        full_logits = full.logits[:, -1].float()
        max_logit_error = (full_logits - logits).abs().max().item()
        # BF16 chunkwise versus split-prefill can differ slightly; compare answer
        # scores as well as logits, and fail instead of silently accepting a bad fork.
        ids = self.tensor([self.continuation_ids(ledger + suffix, c)[0] for c in candidates])
        logp_error = (full_logits.log_softmax(-1)[0, ids] - logits.log_softmax(-1)[0, ids]).abs().max().item()
        prompt = ledger + suffix
        batched = self.candidate_scores(prompt, query_cache, logits, candidates, batch=True)
        serial = self.candidate_scores(prompt, query_cache, logits, candidates, batch=False)
        full_scores = self.candidate_scores(prompt, full.past_key_values, full_logits, candidates, batch=False)
        full_score_error = max(abs(full_scores[c]["nll_mean"] - serial[c]["nll_mean"]) for c in candidates)
        batch_error = max(abs(batched[c]["nll_mean"] - serial[c]["nll_mean"]) for c in candidates)
        assert_cache_equal(cache, saved)
        return {"source_cache_unchanged": True, "cached_full_max_logit_error": max_logit_error,
                "cached_full_first_answer_logp_error": logp_error, "batched_serial_nll_error": batch_error,
                "cached_full_answer_nll_error": full_score_error,
                "cached_full_candidate_prediction_match": min(full_scores, key=lambda c: full_scores[c]["nll_sum"]) == min(serial, key=lambda c: serial[c]["nll_sum"]),
                "cached_prefill_within_tolerance": logp_error <= 0.15 and full_score_error <= 0.15,
                "batch_scoring_within_tolerance": batch_error <= 0.05,
                "tolerances": {"first_answer_logp": 0.15, "answer_nll_mean": 0.15, "batch_nll_mean": 0.05},
                "checked_candidates": len(candidates)}

def state_items(states):
    return states.items() if isinstance(states, Mapping) else enumerate(states)

