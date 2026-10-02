"""Tokenizer-neutral SFT formatting and bounded, optional visible reasoning.

This defines an output protocol, not a pretrained capability. Train on examples
from BOTH modes using encode_sft(). Reserve all SPECIAL_TOKENS inside the 32K
vocabulary and ensure each is a single token before pretraining starts.
"""

from contextlib import nullcontext
from dataclasses import dataclass, fields
import math
from typing import Callable

import torch


SPECIAL_TOKENS = (
    "<|pad|>", "<|bos|>", "<|eos|>", "<|im_start|>", "<|im_end|>",
    "<|think_on|>", "<|think_off|>", "<think>", "</think>", "<answer>", "</answer>",
)


def _encode(tokenizer, text):
    result = tokenizer.encode(text, add_special_tokens=False)
    return list(result.ids if hasattr(result, "ids") else result)


@dataclass(frozen=True)
class ChatTokens:
    pad: int
    bos: int
    eos: int
    im_start: int
    im_end: int
    think_on: int
    think_off: int
    think_start: int
    think_end: int
    answer_start: int
    answer_end: int

    @classmethod
    def from_tokenizer(cls, tokenizer):
        encoded = [_encode(tokenizer, token) for token in SPECIAL_TOKENS]
        if any(len(ids) != 1 for ids in encoded):
            raise ValueError("Every reserved chat marker must encode to exactly one token")
        ids = [tokens[0] for tokens in encoded]
        if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise ValueError("Reserved token IDs must be distinct and nonnegative")
        return cls(*ids)

    @property
    def all_ids(self):
        return tuple(getattr(self, field.name) for field in fields(self))


class ChatTemplate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = ChatTokens.from_tokenizer(tokenizer)

    def encode_prompt(self, messages, *, enable_thinking=False):
        """messages: list of {role: system/user/assistant, content: plain text}."""
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("The prompt must end with a user message")
        t = self.tokens
        result = [t.bos]
        for message in messages:
            role = message["role"]
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role: {role}")
            result += [t.im_start] + _encode(self.tokenizer, role + "\n")
            # Historical assistant messages are answers, without previous think.
            if role == "assistant":
                result += [t.think_off, t.answer_start]
            result += _encode(self.tokenizer, message["content"])
            if role == "assistant":
                result += [t.answer_end]
            result += [t.im_end]
        result += [t.im_start] + _encode(self.tokenizer, "assistant\n")
        result += [t.think_on, t.think_start] if enable_thinking else [t.think_off, t.answer_start]
        return result

    def encode_sft(self, messages, answer, *, reasoning=None):
        """Unshifted labels: only the new assistant completion contributes to CE.

        reasoning=None selects direct-answer mode. Supply a short verified
        explanation for think mode. Do NOT shift these labels a second time.
        """
        prompt = self.encode_prompt(messages, enable_thinking=reasoning is not None)
        t = self.tokens
        completion = []
        if reasoning is not None:
            completion += _encode(self.tokenizer, reasoning) + [t.think_end, t.answer_start]
        completion += _encode(self.tokenizer, answer) + [t.answer_end, t.im_end, t.eos]
        return {"input_ids": prompt + completion, "labels": [-100] * len(prompt) + completion}


@dataclass
class ChatResult:
    answer: str
    reasoning: str
    generated_ids: list[int]
    finish_reason: str
    think_truncated: bool


def _sample(logits, temperature, top_k, top_p):
    scores = logits.float().clone()
    if temperature == 0:
        return int(scores.argmax())
    scores /= temperature
    if top_k:
        cutoff = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores.masked_fill_(scores < cutoff, -torch.inf)
    if top_p < 1:
        ordered, index = scores.sort(descending=True)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[index[remove]] = -torch.inf
    return int(torch.multinomial(scores.softmax(-1), 1))


@torch.inference_mode()
def generate_chat(
    model, template: ChatTemplate, messages, *, enable_thinking=False,
    max_new_tokens=256, max_think_tokens=128,
    temperature=0.8, top_k=40, top_p=0.95,
    on_token: Callable[[int, str], None] | None = None,
):
    """Single-conversation generation with cache and a hard reasoning budget.

    Control tokens count toward max_new_tokens AND the model context limit.
    on_token receives (id, 'think'/'answer'/'control'); decode accumulated token
    IDs when displaying text, as individual byte-level BPE tokens can be partial.
    The controller forces </think><answer> on budget exhaustion; this ensures
    termination of the think section, not correctness of the resulting answer.
    """
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(max_think_tokens) is not int or max_think_tokens < 0:
        raise ValueError("max_think_tokens must be a nonnegative integer")
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be finite and >= 0; top_p must be in (0, 1]")
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    device = next(model.parameters()).device
    prompt = template.encode_prompt(messages, enable_thinking=enable_thinking)
    t = template.tokens
    if max((*prompt, *t.all_ids)) >= model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds model vocabulary; include special tokens in vocab_size")
    available = model.config.max_position_embeddings - len(prompt)
    limit = min(max_new_tokens, available)
    if limit < (3 if enable_thinking else 1):
        raise ValueError("Not enough context/generation budget for this mode")
    think_budget = min(max_think_tokens, limit - 3)
    stage = "think" if enable_thinking else "answer"
    generated, thought, answer = [], [], []
    truncated, finish = False, "length"
    was_training = model.training
    model.eval()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    try:
        with autocast:
            result = model(torch.tensor([prompt], device=device), use_cache=True, logits_to_keep=1)
            for index in range(limit):
                channel = stage
                if stage == "answer_start":
                    token, channel = t.answer_start, "control"
                elif stage == "think" and len(thought) >= think_budget:
                    token, channel, truncated = t.think_end, "control", True
                else:
                    scores = result.logits[0, -1].float().clone()
                    allowed_end = t.think_end if stage == "think" else t.answer_end
                    banned = [i for i in t.all_ids if i != allowed_end]
                    scores[banned] = -torch.inf
                    token = _sample(scores, temperature, top_k, top_p)
                    if token == allowed_end:
                        channel = "control"
                generated.append(token)
                if on_token is not None:
                    on_token(token, channel)
                if stage == "think":
                    if token == t.think_end:
                        stage = "answer_start"
                    else:
                        thought.append(token)
                elif stage == "answer_start":
                    stage = "answer"
                elif token == t.answer_end:
                    finish = "stop"
                    break
                else:
                    answer.append(token)
                if index + 1 < limit:
                    result = model(torch.tensor([[token]], device=device), cache=result.cache, use_cache=True, logits_to_keep=1)
    finally:
        model.train(was_training)
    return ChatResult(
        answer=template.tokenizer.decode(answer, skip_special_tokens=False),
        reasoning=template.tokenizer.decode(thought, skip_special_tokens=False),
        generated_ids=generated, finish_reason=finish, think_truncated=truncated,
    )
