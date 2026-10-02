"""Token-ID-neutral packing, context curriculum and text-distillation helpers.

No tokenizer is created or modified here. The caller owns corpus processing,
teacher answer verification, sampling budgets and the actual training loop.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import random

import torch


def _ids(values, name, *, labels=False):
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype not in {torch.int32, torch.int64}:
            raise ValueError(f"{name} must be a 1D integer tensor or sequence")
        values = values.tolist()
    else:
        values = list(values)
    if not values or any(type(token) is not int or (token < 0 and not (labels and token == -100)) for token in values):
        raise ValueError(f"{name} must contain nonnegative token IDs{' or -100' if labels else ''}")
    return values


def _sample(sample, *, vocab_size=None):
    ids = _ids(sample["input_ids"], "input_ids")
    labels = _ids(sample.get("labels", ids), "labels", labels=True)
    if len(ids) != len(labels):
        raise ValueError("Unshifted labels must have the same length as input_ids")
    if vocab_size is not None and (max(ids) >= vocab_size or max(labels) >= vocab_size):
        raise ValueError("Sample token IDs exceed the existing model vocabulary")
    return {"input_ids": ids, "labels": labels}


def best_fit_pack(
    samples: Iterable[Mapping], *, seq_length: int, pad_token_id: int,
    vocab_size: int | None = None, device=None,
) -> dict[str, torch.Tensor]:
    """Best-fit decreasing packing of whole, already-tokenized samples.

    Run on a bounded pool (e.g. a collator's examples), not the entire corpus.
    Each row is padded to seq_length and has local segment IDs 0, 1, ...;
    padding has segment ID -1. Positions restart at zero for every sample.
    Original assistant-only labels are preserved, without shifting. The model
    must reset its recurrent/convolution state and mask cross-segment targets.
    Oversized samples raise rather than cutting a reasoning/answer sequence.
    """
    if type(seq_length) is not int or seq_length < 2:
        raise ValueError("seq_length must be an integer >= 2")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative existing token ID")
    if vocab_size is not None and (type(vocab_size) is not int or vocab_size <= pad_token_id):
        raise ValueError("vocab_size must exceed pad_token_id")
    normalized = [_sample(sample, vocab_size=vocab_size) for sample in samples]
    if not normalized:
        raise ValueError("Packing requires at least one sample")
    for index, sample in enumerate(normalized):
        if len(sample["input_ids"]) > seq_length:
            raise ValueError(f"Sample {index} has {len(sample['input_ids'])} tokens, exceeding seq_length={seq_length}; it was not truncated")
    rows, remaining = [], []
    for sample in sorted(normalized, key=lambda item: len(item["input_ids"]), reverse=True):
        length = len(sample["input_ids"])
        candidates = [(room - length, row) for row, room in enumerate(remaining) if room >= length]
        if candidates:
            _, row = min(candidates)
            rows[row].append(sample)
            remaining[row] -= length
        else:
            rows.append([sample])
            remaining.append(seq_length - length)
    shape = (len(rows), seq_length)
    batch = dict(
        input_ids=torch.full(shape, pad_token_id, dtype=torch.long),
        labels=torch.full(shape, -100, dtype=torch.long),
        attention_mask=torch.zeros(shape, dtype=torch.bool),
        segment_ids=torch.full(shape, -1, dtype=torch.long),
        position_ids=torch.zeros(shape, dtype=torch.long),
    )
    for row, items in enumerate(rows):
        offset = 0
        for segment, sample in enumerate(items):
            length = len(sample["input_ids"])
            location = slice(offset, offset + length)
            batch["input_ids"][row, location] = torch.tensor(sample["input_ids"])
            batch["labels"][row, location] = torch.tensor(sample["labels"])
            batch["attention_mask"][row, location] = True
            batch["segment_ids"][row, location] = segment
            batch["position_ids"][row, location] = torch.arange(length)
            offset += length
    return {name: tensor.to(device) for name, tensor in batch.items()} if device is not None else batch


@dataclass(frozen=True)
class PackedCollator:
    """DataLoader collator: batch size is the sample pool, packed rows vary."""
    seq_length: int
    pad_token_id: int
    vocab_size: int | None = None

    def __call__(self, samples):
        return best_fit_pack(samples, seq_length=self.seq_length, pad_token_id=self.pad_token_id, vocab_size=self.vocab_size)


@dataclass(frozen=True)
class ContextStage:
    seq_length: int
    tokens: int

    def __post_init__(self):
        if type(self.seq_length) is not int or self.seq_length < 2:
            raise ValueError("Stage seq_length must be an integer >= 2")
        if type(self.tokens) is not int or self.tokens <= 0:
            raise ValueError("Stage token budget must be a positive integer")


class ContextCurriculum:
    """Choose context by consumed nonpadding tokens; never starts training.

    Pass explicit budgets: ContextCurriculum([ContextStage(2048, budget_2k),
    ContextStage(4096, budget_4k), ContextStage(10240, budget_10k)]).
    Save processed_tokens in the user's training checkpoint. After exhaustion
    stage_for_tokens returns the last stage; finished() tells the loop to stop.
    """
    def __init__(self, stages: Sequence[ContextStage], *, max_position_embeddings=10240):
        self.stages = tuple(stages)
        if not self.stages or any(not isinstance(stage, ContextStage) for stage in self.stages):
            raise ValueError("stages must be a nonempty sequence of ContextStage")
        if type(max_position_embeddings) is not int or max_position_embeddings < 2:
            raise ValueError("max_position_embeddings must be an integer >= 2")
        lengths = [stage.seq_length for stage in self.stages]
        if any(b <= a for a, b in zip(lengths, lengths[1:])):
            raise ValueError("Context lengths must strictly increase")
        if lengths[-1] > max_position_embeddings:
            raise ValueError("A stage exceeds the model context limit")
        self.total_tokens = sum(stage.tokens for stage in self.stages)

    @staticmethod
    def _validate_progress(processed_tokens):
        if type(processed_tokens) is not int or processed_tokens < 0:
            raise ValueError("processed_tokens must be a nonnegative integer")

    def stage_for_tokens(self, processed_tokens: int) -> ContextStage:
        self._validate_progress(processed_tokens)
        boundary = 0
        for stage in self.stages:
            boundary += stage.tokens
            if processed_tokens < boundary:
                return stage
        return self.stages[-1]

    def seq_length(self, processed_tokens: int) -> int:
        return self.stage_for_tokens(processed_tokens).seq_length

    def finished(self, processed_tokens: int) -> bool:
        self._validate_progress(processed_tokens)
        return processed_tokens >= self.total_tokens


def encode_distillation_records(template, records: Iterable[Mapping], *, require_verified=True, max_sample_tokens=None):
    """Encode verified teacher TEXT with the existing ChatTemplate/tokenizer.

    Record fields: messages, answer, optional reasoning, answer_verified.
    A caller may set require_verified=False for separately verified data.
    The helper performs no teacher inference and no cross-tokenizer logit KL.
    Length limits reject complete examples rather than silently truncating.
    """
    if max_sample_tokens is not None and (type(max_sample_tokens) is not int or max_sample_tokens < 2):
        raise ValueError("max_sample_tokens must be an integer >= 2")
    encoded = []
    for index, record in enumerate(records):
        if require_verified and record.get("answer_verified") is not True:
            raise ValueError(f"Teacher sample {index} needs answer_verified=True after external answer checking")
        answer, reasoning = record["answer"], record.get("reasoning")
        if not isinstance(answer, str) or (reasoning is not None and not isinstance(reasoning, str)):
            raise ValueError("Teacher answer and optional reasoning must be text")
        sample = template.encode_sft(record["messages"], answer, reasoning=reasoning)
        if max_sample_tokens is not None and len(sample["input_ids"]) > max_sample_tokens:
            raise ValueError(f"Teacher sample {index} exceeds max_sample_tokens; reasoning and answer were not truncated")
        encoded.append(sample)
    return encoded


def mix_sft_examples(
    direct_examples: Sequence[Mapping], reasoning_examples: Sequence[Mapping], *,
    reasoning_fraction=0.5, total_samples: int | None = None, seed=17,
):
    """Deterministic text-SFT mixing; resamples each pool with replacement.

    The returned finite pool preserves unshifted assistant-only labels. The
    teacher's answer quality and desired language/task mix remain caller-owned.
    """
    if not math.isfinite(reasoning_fraction) or not 0 <= reasoning_fraction <= 1:
        raise ValueError("reasoning_fraction must be in [0, 1]")
    if total_samples is None:
        total_samples = len(direct_examples) + len(reasoning_examples)
    if type(total_samples) is not int or total_samples < 1:
        raise ValueError("total_samples must be a positive integer")
    n_reasoning = int(total_samples * reasoning_fraction + 0.5)
    n_direct = total_samples - n_reasoning
    if (n_reasoning and not reasoning_examples) or (n_direct and not direct_examples):
        raise ValueError("Each requested SFT mode needs a nonempty example pool")
    rng = random.Random(seed)
    result = [_sample(rng.choice(direct_examples)) for _ in range(n_direct)]
    result += [_sample(rng.choice(reasoning_examples)) for _ in range(n_reasoning)]
    rng.shuffle(result)
    return result


def build_retrieval_probe(
    *, needle_ids, distractor_ids, query_ids, answer_ids, seq_length: int,
    depth=0.5, prefix_ids=(),
):
    """Tokenized long-context retrieval probe with exact total length.

    Supply a verified needle/query/answer in the existing vocabulary. Only
    distractor filler is repeated/cropped; the needle, query and answer remain
    complete. Labels supervise the answer only. This prepares a sample, not an
    accuracy score: compare generated answers or answer-only loss externally.
    """
    if type(seq_length) is not int or seq_length < 2 or not math.isfinite(depth) or not 0 <= depth <= 1:
        raise ValueError("seq_length must be >= 2 and depth must be in [0, 1]")
    needle, distractor = _ids(needle_ids, "needle_ids"), _ids(distractor_ids, "distractor_ids")
    query, answer = _ids(query_ids, "query_ids"), _ids(answer_ids, "answer_ids")
    prefix = _ids(prefix_ids, "prefix_ids") if len(prefix_ids) else []
    room = seq_length - len(prefix) - len(needle) - len(query) - len(answer)
    if room < 0:
        raise ValueError("Complete retrieval sample exceeds seq_length; it was not truncated")
    filler = (distractor * ((room + len(distractor) - 1) // len(distractor)))[:room]
    split = int(room * depth)
    prompt = prefix + filler[:split] + needle + filler[split:] + query
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}
