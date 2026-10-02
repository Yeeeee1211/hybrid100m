import copy
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from hybrid100m import HybridConfig, HybridForCausalLM
from hybrid100m.chat import ChatTemplate, SPECIAL_TOKENS, generate_chat
from hybrid100m.model import DeltaCache, KVCache, RotaryEmbedding, apply_rope, chunked_cross_entropy
from hybrid100m.ops import gated_delta, recurrent_delta_reference


def small_config(**kwargs):
    values = dict(
        vocab_size=160, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_heads=2, linear_head_dim=8, linear_backend="reference", sdpa_backend="math",
        max_position_embeddings=64, gradient_checkpointing=False, loss_chunk_size=5,
        linear_attention="gated_delta", rope_type="default", mtp_num_layers=0, loss_backend="chunked",
    )
    values.update(kwargs)
    return HybridConfig(**values)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def test_default_parameter_count_and_weight_tying(self):
        with torch.device("meta"):
            model = HybridForCausalLM()
        self.assertEqual(model.num_backbone_parameters(), 100_162_108)
        self.assertGreater(model.num_parameters(), model.num_backbone_parameters())
        self.assertIs(model.embedding.weight, model.lm_head.weight)
        self.assertEqual(model.config.layer_types, ("linear", "linear", "linear", "full") * 5)

    def test_reference_matches_independent_matrix_update_and_gradients(self):
        q, k, v = [torch.randn(2, 5, 2, 3, requires_grad=True) for _ in range(3)]
        g = (-torch.rand(2, 5, 2)).requires_grad_()
        beta = torch.rand(2, 5, 2, requires_grad=True)
        initial = torch.randn(2, 2, 3, 3, requires_grad=True)
        actual, state = recurrent_delta_reference(q, k, v, g, beta, initial)
        qn = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        kn = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        h = initial
        expected = []
        eye = torch.eye(3)
        for i in range(5):
            ki = kn[:, i]
            write = torch.einsum("bhk,bhv->bhkv", ki, v[:, i])
            erase = eye - beta[:, i, :, None, None] * torch.einsum("bhk,bhl->bhkl", ki, ki)
            h = g[:, i, :, None, None].exp() * (erase @ h) + beta[:, i, :, None, None] * write
            expected.append(torch.einsum("bhk,bhkv->bhv", qn[:, i] / math.sqrt(3), h))
        expected = torch.stack(expected, dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(state, h, atol=1e-6, rtol=1e-5)
        inputs = (q, k, v, g, beta, initial)
        a = torch.autograd.grad(actual.square().sum() + state.square().sum(), inputs, retain_graph=True)
        b = torch.autograd.grad(expected.square().sum() + h.square().sum(), inputs)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, atol=3e-6, rtol=1e-4)

    def test_causality(self):
        model = HybridForCausalLM(small_config()).eval()
        x = torch.randint(0, 160, (2, 10))
        changed = x.clone()
        changed[:, 5:] = (changed[:, 5:] + 13) % 160
        with torch.no_grad():
            a, b = model(x).logits, model(changed).logits
        torch.testing.assert_close(a[:, :5], b[:, :5], atol=1e-7, rtol=1e-6)

    def test_cache_matches_full_for_single_tokens_and_multi_token_chunks(self):
        model = HybridForCausalLM(small_config()).eval()
        x = torch.randint(0, 160, (2, 11))
        with torch.no_grad():
            expected = model(x).logits
            for chunks in ([1] * 11, [3, 1, 4, 3]):
                cache, outputs, offset = None, [], 0
                for length in chunks:
                    output = model(x[:, offset:offset + length], use_cache=True, cache=cache)
                    cache = output.cache
                    outputs.append(output.logits)
                    offset += length
                torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=2e-6, rtol=2e-5)
                self.assertEqual(cache.seen_tokens, 11)
                for state in cache.layers:
                    if isinstance(state, DeltaCache):
                        self.assertEqual(state.state.dtype, torch.float32)
                        self.assertEqual(state.conv.shape[-1], 3)
                    else:
                        self.assertIsInstance(state, KVCache)
                        self.assertEqual(state.key.shape[1], 2)

    def test_cache_is_not_mutated_when_branching(self):
        model = HybridForCausalLM(small_config()).eval()
        with torch.no_grad():
            prefix = model(torch.tensor([[1, 3, 5]]), use_cache=True).cache
            saved = copy.deepcopy(prefix)
            a = model(torch.tensor([[7, 9]]), cache=prefix, use_cache=True).logits
            model(torch.tensor([[8]]), cache=prefix, use_cache=True)
            b = model(torch.tensor([[7, 9]]), cache=prefix, use_cache=True).logits
        torch.testing.assert_close(a, b)
        for a, b in zip(prefix.layers, saved.layers):
            if isinstance(a, DeltaCache):
                torch.testing.assert_close(a.state, b.state)
                torch.testing.assert_close(a.conv, b.conv)
            else:
                torch.testing.assert_close(a.key, b.key)

    def test_checkpointed_training_matches_dense_ce_and_gradients(self):
        a = HybridForCausalLM(small_config(gradient_checkpointing=True))
        b = HybridForCausalLM(small_config())
        b.load_state_dict(a.state_dict())
        ids = torch.randint(0, 160, (2, 7))
        labels = ids.clone()
        labels[:, :3] = -100
        loss_a = a(ids, labels=labels).loss
        logits = b(ids).logits
        loss_b = F.cross_entropy(logits[:, :-1].reshape(-1, 160), labels[:, 1:].reshape(-1), ignore_index=-100)
        loss_a.backward()
        loss_b.backward()
        torch.testing.assert_close(loss_a, loss_b)
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(p.grad.isfinite().all()), name)
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-4, msg=name)

    def test_right_padding_matches_independent_sequences_and_loss(self):
        model = HybridForCausalLM(small_config()).eval()
        ids = torch.randint(0, 160, (2, 8))
        valid = torch.tensor([[1] * 5 + [0] * 3, [1] * 8])
        output = model(ids, labels=ids, attention_mask=valid, return_logits=True)
        expected_sum = 0
        for row, length in enumerate((5, 8)):
            standalone = model(ids[row:row + 1, :length]).logits
            torch.testing.assert_close(output.logits[row, :length], standalone[0], atol=1e-6, rtol=1e-5)
            expected_sum += F.cross_entropy(standalone[0, :-1], ids[row, 1:length], reduction="sum")
        torch.testing.assert_close(output.loss, expected_sum / 11)
        altered = ids.clone()
        altered[0, 5:] = (altered[0, 5:] + 1) % 160
        torch.testing.assert_close(output.loss, model(altered, labels=altered, attention_mask=valid).loss)

    def test_all_ignored_loss_is_finite_zero(self):
        model = HybridForCausalLM(small_config())
        ids = torch.randint(0, 160, (1, 5))
        out = model(ids, labels=torch.full_like(ids, -100))
        self.assertEqual(out.loss.item(), 0)
        out.loss.backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters()))

    def test_bf16_autocast_forward_backward(self):
        model = HybridForCausalLM(small_config(gradient_checkpointing=True))
        ids = torch.randint(0, 160, (1, 6))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(ids, labels=ids)
        self.assertTrue(bool(out.loss.isfinite()))
        out.loss.backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters()))

    def test_rope_at_10k_and_bf16_module_conversion(self):
        rope = RotaryEmbedding(HybridConfig(rope_type="default"))
        p = torch.tensor([0, 10000, 10239])
        expected = rope(p)
        actual = rope.to(torch.bfloat16)(p)
        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        x = torch.randn(1, 2, 3, 64)
        y = apply_rope(x, *actual)
        torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1))
        self.assertEqual(actual[0].dtype, torch.float32)

    def test_invalid_padding_cache_and_context_are_rejected(self):
        model = HybridForCausalLM(small_config())
        x = torch.tensor([[1, 2, 3]])
        for mask in ([[0, 1, 1]], [[1, 0, 1]], [[0, 0, 0]], [[1, 2, 1]]):
            with self.assertRaises(ValueError):
                model(x, attention_mask=torch.tensor(mask))
        with self.assertRaises(ValueError):
            model(x, use_cache=True)
        with self.assertRaises(ValueError):
            model(x.repeat(1, 22))
        with self.assertRaises(ValueError):
            model(x, labels=x.to(torch.int32))
        model.eval()
        with torch.no_grad(), self.assertRaises(ValueError):
            model(x, use_cache=True, attention_mask=torch.tensor([[1, 1, 0]]))

    def test_fla_fails_explicitly_on_cpu(self):
        model = HybridForCausalLM(small_config(linear_backend="fla"))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            model(torch.tensor([[1, 2]]))

    def test_config_rejects_invalid_architecture(self):
        for kw in ({"num_hidden_layers": 15}, {"num_key_value_heads": 3}, {"rotary_fraction": 0.01}, {"rope_theta": 1}):
            with self.assertRaises(ValueError):
                HybridConfig(**kw)

    def test_training_never_uses_forward_only_recurrent_kernel(self):
        # Verify dispatch without requiring an installed Triton runtime.
        class FakeTensor:
            device = SimpleNamespace(type="cuda")
            dtype = torch.bfloat16
            shape = (1, 1, 4, 128)
            def contiguous(self): return self
            def float(self): return self
        x = FakeTensor()
        with patch("hybrid100m.ops._fla_ops") as ops:
            chunk, recurrent = unittest.mock.Mock(), unittest.mock.Mock()
            chunk.return_value = recurrent.return_value = (x, x)
            ops.return_value = chunk, recurrent
            gated_delta(x, x, x, x, x, backend="fla")
            chunk.assert_called_once()
            recurrent.assert_not_called()
            with torch.no_grad():
                gated_delta(x, x, x, x, x, backend="fla")
            recurrent.assert_called_once()


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text in SPECIAL_TOKENS:
            return [SPECIAL_TOKENS.index(text)]
        return [ord(c) + 16 for c in text]
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i - 16) for i in ids)


class EndlessModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(vocab_size=160, max_position_embeddings=128)
    def forward(self, ids, **kwargs):
        logits = torch.full((1, 1, 160), -100.0)
        logits[0, 0, ord("x") + 16] = 10.0
        return SimpleNamespace(logits=logits, cache=None)


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.template = ChatTemplate(FakeTokenizer())
        self.messages = [{"role": "user", "content": "2+3?"}]

    def test_sft_prompt_is_identical_to_generation_and_masks_only_prompt(self):
        for reasoning in (None, "2+3=5"):
            item = self.template.encode_sft(self.messages, "5", reasoning=reasoning)
            prompt = self.template.encode_prompt(self.messages, enable_thinking=reasoning is not None)
            self.assertEqual(item["input_ids"][:len(prompt)], prompt)
            self.assertEqual(item["labels"][:len(prompt)], [-100] * len(prompt))
            self.assertEqual(item["labels"][len(prompt):], item["input_ids"][len(prompt):])

    def test_reasoning_budget_forces_transition(self):
        model = EndlessModel()
        calls = []
        out = generate_chat(model, self.template, self.messages, enable_thinking=True,
                            max_new_tokens=8, max_think_tokens=2, temperature=0,
                            on_token=lambda token, channel: calls.append(channel))
        self.assertEqual(out.reasoning, "xx")
        self.assertEqual(out.answer, "xxxx")
        self.assertEqual(len(out.generated_ids), 8)
        self.assertTrue(out.think_truncated)
        self.assertEqual(calls, ["think", "think", "control", "control"] + ["answer"] * 4)
        self.assertTrue(model.training)

    def test_non_thinking_mode_has_no_reasoning(self):
        out = generate_chat(EndlessModel(), self.template, self.messages, max_new_tokens=3, temperature=0)
        self.assertEqual(out.reasoning, "")
        self.assertEqual(out.answer, "xxx")
        self.assertFalse(out.think_truncated)

    def test_context_budget_is_enforced(self):
        model = EndlessModel()
        prompt = self.template.encode_prompt(self.messages)
        model.config.max_position_embeddings = len(prompt) + 2
        out = generate_chat(model, self.template, self.messages, max_new_tokens=20, temperature=0)
        self.assertEqual(len(out.generated_ids), 2)


if __name__ == "__main__":
    unittest.main()
