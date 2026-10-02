"""Independent CPU correctness checks for the combined P1 architecture."""

import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hybrid100m import HybridConfig, HybridForCausalLM, build_optimizer
from hybrid100m.data import PackedCollator
from hybrid100m.model import DeltaCache, KVCache, RotaryEmbedding
from hybrid100m.losses import cross_entropy_stats


def p1_config(**overrides):
    settings = dict(
        vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_heads=2, linear_head_dim=8, linear_gate_rank=8,
        linear_attention="kda", linear_backend="reference", sdpa_backend="math",
        rope_type="yarn", rope_original_max_position_embeddings=16, rope_factor=4.0,
        max_position_embeddings=64, gradient_checkpointing=False,
        loss_backend="chunked", loss_chunk_size=3, mtp_num_layers=1, mtp_loss_weight=0.1,
    )
    settings.update(overrides)
    return HybridConfig(**settings)


def documents():
    return [
        {"input_ids": list(range(10, 17)), "labels": [-100, -100, 12, 13, 14, 15, 16]},
        {"input_ids": [30, 31], "labels": [30, 31]},
        {"input_ids": [40], "labels": [40]},
        {"input_ids": list(range(50, 55)), "labels": [-100, -100, -100, 53, 54]},
        {"input_ids": [70, 71, 72], "labels": [-100] * 3},
    ]


class P1ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(29)

    def test_packing_matches_individual_logits_losses_counts_and_gradients(self):
        packed_model = HybridForCausalLM(p1_config(gradient_checkpointing=True))
        independent = HybridForCausalLM(p1_config())
        independent.load_state_dict(packed_model.state_dict())
        docs = documents()
        batch = PackedCollator(8, 0, 96)(docs)
        self.assertEqual(batch["input_ids"].shape, (3, 8))
        self.assertTrue(bool((~batch["attention_mask"]).any()))
        actual = packed_model(**batch, return_logits=True, compute_mtp=True)
        expected = [independent(torch.tensor([doc["input_ids"]]), labels=torch.tensor([doc["labels"]]),
                                return_logits=True, compute_mtp=True) for doc in docs]
        by_tokens = {tuple(doc["input_ids"]): result for doc, result in zip(docs, expected)}
        for row in range(batch["input_ids"].shape[0]):
            valid = batch["attention_mask"][row]
            segments = batch["segment_ids"][row]
            for segment in segments[valid].unique():
                selection = valid & segments.eq(segment)
                tokens = tuple(batch["input_ids"][row, selection].tolist())
                torch.testing.assert_close(actual.logits[row, selection], by_tokens[tokens].logits[0], atol=2e-7, rtol=2e-6)
        ntp_sum = sum(result.loss_sum for result in expected)
        ntp_count = sum(result.num_loss_tokens for result in expected)
        mtp_sum = sum(result.mtp_loss_sums[0] for result in expected)
        mtp_count = sum(result.mtp_num_tokens[0] for result in expected)
        self.assertEqual(ntp_count.item(), 8)
        self.assertEqual(mtp_count.item(), 7)
        torch.testing.assert_close(actual.loss_sum, ntp_sum)
        torch.testing.assert_close(actual.num_loss_tokens, ntp_count)
        torch.testing.assert_close(actual.mtp_loss_sums[0], mtp_sum)
        torch.testing.assert_close(actual.mtp_num_tokens[0], mtp_count)
        expected_loss = ntp_sum / ntp_count + packed_model.config.mtp_loss_weight * mtp_sum / mtp_count
        torch.testing.assert_close(actual.loss, expected_loss)
        actual.loss.backward()
        expected_loss.backward()
        for (name, a), (_, b) in zip(packed_model.named_parameters(), independent.named_parameters()):
            self.assertIsNotNone(a.grad, name)
            self.assertIsNotNone(b.grad, name)
            self.assertTrue(bool(a.grad.isfinite().all()), name)
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=5e-4, msg=name)

    def test_later_document_cannot_observe_changed_earlier_document(self):
        model = HybridForCausalLM(p1_config()).eval()
        batch = PackedCollator(12, 0, 96)([
            {"input_ids": [11, 12, 13, 14, 15]}, {"input_ids": [21, 22, 23, 24]},
        ])
        inputs = {key: value for key, value in batch.items() if key != "labels"}
        changed = {key: value.clone() for key, value in inputs.items()}
        first = inputs["segment_ids"].eq(0)
        changed["input_ids"][first] = (changed["input_ids"][first] + 19) % 96
        with torch.no_grad():
            before = model(**inputs).logits
            after = model(**changed).logits
        second = inputs["segment_ids"].eq(1)
        torch.testing.assert_close(before[second], after[second], atol=0, rtol=0)

    def test_cross_boundary_predictions_are_excluded_from_ntp_and_mtp(self):
        model = HybridForCausalLM(p1_config())
        batch = PackedCollator(7, 0, 96)([{"input_ids": [10, 11]}, {"input_ids": [20, 21]}, {"input_ids": [30]}])
        result = model(**batch, compute_mtp=True)
        self.assertEqual(result.num_loss_tokens.item(), 2)
        self.assertEqual(result.mtp_num_tokens[0].item(), 0)
        self.assertEqual(result.mtp_loss_sums[0].item(), 0)
        result.loss.backward()
        self.assertTrue(all(param.grad is not None and bool(param.grad.isfinite().all()) for param in model.parameters()))

    def test_mtp_observes_next_token_and_predicts_following_token(self):
        model = HybridForCausalLM(p1_config())
        ids = torch.tensor([[10, 20, 30, 40, 50, 60]])
        next_embeddings, ce_targets = [], []
        hook = model.mtp[0].register_forward_pre_hook(lambda module, args: next_embeddings.append(args[1].detach().clone()))
        def record_ce(hidden, weight, targets, *args, **kwargs):
            ce_targets.append(targets.detach().clone())
            return cross_entropy_stats(hidden, weight, targets, *args, **kwargs)
        try:
            with patch("hybrid100m.model.cross_entropy_stats", side_effect=record_ce):
                result = model(ids, labels=ids, compute_mtp=True)
        finally:
            hook.remove()
        self.assertEqual(len(next_embeddings), 1)
        torch.testing.assert_close(next_embeddings[0], model.embedding(ids[:, 1:]))
        self.assertEqual(ce_targets[0].tolist(), [[20, 30, 40, 50, 60]])
        self.assertEqual(ce_targets[1].tolist(), [[30, 40, 50, 60]])
        self.assertEqual(result.mtp_num_tokens[0].item(), 4)

    def test_disabling_mtp_preserves_main_logits_and_enabled_mtp_has_gradients(self):
        model = HybridForCausalLM(p1_config())
        ids = torch.tensor([[3, 8, 11, 14, 22, 35]])
        enabled = model(ids, labels=ids, compute_mtp=True, return_logits=True)
        disabled = model(ids, labels=ids, compute_mtp=False, return_logits=True)
        torch.testing.assert_close(enabled.logits, disabled.logits, atol=0, rtol=0)
        torch.testing.assert_close(disabled.loss, disabled.ntp_loss, atol=0, rtol=0)
        self.assertIsNone(disabled.mtp_loss)
        enabled.loss.backward()
        for name, param in model.mtp.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(bool(param.grad.isfinite().all()), name)
        self.assertGreater(model.mtp[0].fuse.weight.grad.abs().sum().item(), 0)

    def test_kda_cache_chunks_and_branching_match_full_sequence(self):
        model = HybridForCausalLM(p1_config()).eval()
        ids = torch.randint(0, 96, (2, 12))
        with torch.no_grad():
            full = model(ids).logits
            cache, chunks, offset = None, [], 0
            for length in (3, 1, 5, 3):
                result = model(ids[:, offset:offset + length], use_cache=True, cache=cache)
                cache = result.cache
                chunks.append(result.logits)
                offset += length
            torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=2e-6, rtol=2e-5)
            prefix = model(ids[:, :4], use_cache=True).cache
            saved = copy.deepcopy(prefix)
            branch_a = model(ids[:, 4:7], cache=prefix, use_cache=True).logits
            model((ids[:, 4:6] + 1) % 96, cache=prefix, use_cache=True)
            branch_b = model(ids[:, 4:7], cache=prefix, use_cache=True).logits
            torch.testing.assert_close(branch_a, branch_b, atol=0, rtol=0)
            for original, copied in zip(prefix.layers, saved.layers):
                if isinstance(original, DeltaCache):
                    torch.testing.assert_close(original.state, copied.state, atol=0, rtol=0)
                    torch.testing.assert_close(original.conv, copied.conv, atol=0, rtol=0)
                else:
                    self.assertIsInstance(original, KVCache)
                    torch.testing.assert_close(original.key, copied.key, atol=0, rtol=0)
                    torch.testing.assert_close(original.value, copied.value, atol=0, rtol=0)

    def test_yarn_matches_independent_scalar_formula_and_static_prefix(self):
        cfg = p1_config(head_dim=64, rope_original_max_position_embeddings=128, rope_factor=5.0, max_position_embeddings=640)
        rope = RotaryEmbedding(cfg)
        positions = torch.tensor([0, 17, 127, 256, 639])
        cos, sin = rope(positions)
        lower = max(math.floor(64 * math.log(128 / (2 * math.pi * 32)) / (2 * math.log(10000))), 0)
        upper = min(math.ceil(64 * math.log(128 / (2 * math.pi)) / (2 * math.log(10000))), 63)
        scale = 1 + 0.1 * math.log(5)
        frequencies = []
        for index in range(32):
            blend = max(0, min(1, (index - lower) / max(upper - lower, 0.001)))
            original = 10000 ** (-2 * index / 64)
            frequencies.append(original * (1 - blend) + original / 5 * blend)
        phases = [[position * frequency for frequency in frequencies] * 2 for position in positions.tolist()]
        phases = torch.tensor(phases, dtype=torch.float32)[None, None]
        # The scalar oracle forms frequencies in Python FP64; the model forms
        # them in FP32, so long-position phase rounding needs a small tolerance.
        torch.testing.assert_close(cos, phases.cos() * scale, atol=3e-5, rtol=1e-4)
        torch.testing.assert_close(sin, phases.sin() * scale, atol=3e-5, rtol=1e-4)
        short_cos, short_sin = rope(torch.arange(20))
        long_cos, long_sin = rope(torch.arange(640))
        torch.testing.assert_close(short_cos, long_cos[..., :20, :], atol=0, rtol=0)
        torch.testing.assert_close(short_sin, long_sin[..., :20, :], atol=0, rtol=0)
        bf16_cos, bf16_sin = rope.to(torch.bfloat16)(positions)
        torch.testing.assert_close(cos, bf16_cos, atol=0, rtol=0)
        torch.testing.assert_close(sin, bf16_sin, atol=0, rtol=0)

    def test_old_config_load_retains_original_algorithms(self):
        values = asdict(p1_config())
        for name in ("linear_attention", "linear_gate_rank", "rope_type", "rope_factor",
                     "rope_original_max_position_embeddings", "rope_beta_fast", "rope_beta_slow",
                     "loss_backend", "mtp_num_layers", "mtp_loss_weight"):
            values.pop(name)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            loaded = HybridConfig.load(path)
        self.assertEqual(loaded.linear_attention, "gated_delta")
        self.assertEqual(loaded.rope_type, "default")
        self.assertEqual(loaded.mtp_num_layers, 0)
        self.assertEqual(loaded.loss_backend, "chunked")
        model = HybridForCausalLM(loaded)
        self.assertFalse(model.mtp)

    def test_invalid_segment_spans_and_positions_are_rejected(self):
        model = HybridForCausalLM(p1_config())
        ids = torch.tensor([[10, 11, 12, 13]])
        for segments in ([[0, 1, 0, 0]], [[0, -1, 1, 1]], [[-1, 0, 0, 0]]):
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                model(ids, segment_ids=torch.tensor(segments))
        valid_segments = torch.tensor([[0, 0, 1, 1]])
        for positions in ([[0, 1, 2, 3]], [[0, 1, 0, -1]], [[0, 1, 0, 64]]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                model(ids, segment_ids=valid_segments, position_ids=torch.tensor(positions))
        with self.assertRaises(ValueError):
            model(ids, segment_ids=valid_segments.float())
        with self.assertRaises(ValueError):
            model(ids, position_ids=torch.tensor([0, 1, 2, 3]))
        with self.assertRaises(ValueError):
            model(ids, segment_ids=torch.tensor([[0, 0, 1, -2]]), attention_mask=torch.tensor([[1, 1, 1, 0]]))

    def test_bf16_checkpointed_packing_and_muon_step_keep_fp32_parameters(self):
        cfg = p1_config(gradient_checkpointing=True)
        checked = HybridForCausalLM(cfg)
        plain = HybridForCausalLM(replace(cfg, gradient_checkpointing=False))
        plain.load_state_dict(checked.state_dict())
        batch = PackedCollator(8, 0, 96)(documents())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            left = checked(**batch)
            right = plain(**batch)
        torch.testing.assert_close(left.loss, right.loss, rtol=0, atol=0)
        left.loss.backward()
        right.loss.backward()
        for (name, a), (_, b) in zip(checked.named_parameters(), plain.named_parameters()):
            self.assertEqual(a.dtype, torch.float32, name)
            self.assertIsNotNone(a.grad, name)
            self.assertTrue(bool(a.grad.isfinite().all()), name)
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
        saved = checked.mtp[0].fuse.weight.detach().clone()
        build_optimizer(checked).step()
        self.assertFalse(torch.equal(saved, checked.mtp[0].fuse.weight))
        self.assertTrue(all(p.dtype == torch.float32 and bool(p.isfinite().all()) for p in checked.parameters()))

    def test_serial_two_depths_exclude_short_documents_and_all_ignored_gradients(self):
        model = HybridForCausalLM(p1_config(mtp_num_layers=2))
        batch = PackedCollator(8, 0, 96)(documents())
        output = model(**batch)
        self.assertEqual([n.item() for n in output.mtp_num_tokens], [7, 6])
        expected = output.ntp_loss + model.config.mtp_loss_weight * sum(
            s / n.clamp_min(1) for s, n in zip(output.mtp_loss_sums, output.mtp_num_tokens)) / 2
        torch.testing.assert_close(output.loss, expected)
        batch["labels"].fill_(-100)
        ignored = model(**batch)
        self.assertEqual(ignored.loss.item(), 0)
        ignored.loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(p.grad.eq(0).all()), name)
        with self.assertRaises(ValueError):
            model(torch.tensor([[3, 5]]), labels=torch.tensor([[3, 96]]))


if __name__ == "__main__":
    unittest.main()
