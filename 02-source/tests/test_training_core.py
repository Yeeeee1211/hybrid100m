import copy
import math
import unittest

import torch
from torch import nn

from hybrid100m.chat import ChatTemplate, SPECIAL_TOKENS
from hybrid100m.data import (
    ContextCurriculum, ContextStage, PackedCollator, best_fit_pack,
    build_retrieval_probe, encode_distillation_records, mix_sft_examples,
)
from hybrid100m.optim import build_optimizer, newton_schulz_orthogonalize


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 4)
        self.hidden = nn.Linear(4, 6)
        self.norm = nn.LayerNorm(4)
        self.conv = nn.Conv1d(4, 4, 3)
        self.A_log = nn.Parameter(torch.zeros(2))
        self.dt_bias = nn.Parameter(torch.zeros(2, 2))
        self.lm_head = nn.Linear(4, 11, bias=False)
        self.lm_head.weight = self.embedding.weight


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(37)

    def test_grouping_deduplicates_embedding_and_keeps_constants_out_of_muon(self):
        model = TinyModel()
        optimizer = build_optimizer(model)
        grouped = [param for group in optimizer.param_groups for param in group["params"]]
        self.assertEqual(len(grouped), len(list(model.parameters())))
        self.assertEqual(len({id(param) for param in grouped}), len(grouped))
        by_id = {id(param): group for group in optimizer.param_groups for param in group["params"]}
        self.assertEqual(by_id[id(model.hidden.weight)]["algorithm"], "muon")
        self.assertEqual(by_id[id(model.embedding.weight)]["algorithm"], "adamw")
        self.assertEqual(by_id[id(model.conv.weight)]["algorithm"], "adamw")
        for param in (model.hidden.bias, model.norm.weight, model.A_log, model.dt_bias):
            self.assertEqual(by_id[id(param)]["algorithm"], "adamw")
            self.assertEqual(by_id[id(param)]["weight_decay"], 0)

    def test_muon_scaled_update_and_zero_matrix(self):
        model = TinyModel()
        optimizer = build_optimizer(model, muon_lr=0.01, weight_decay=0.1)
        param = model.hidden.weight
        before = param.detach().clone()
        param.grad = torch.randn_like(param)
        expected = before * (1 - 0.01 * 0.1) - 0.01 * 0.2 * math.sqrt(6) * newton_schulz_orthogonalize(param.grad)
        optimizer.step()
        # The first Nesterov factor is scalar and disappears in normalization.
        torch.testing.assert_close(param, expected, rtol=3e-6, atol=3e-7)
        zero = torch.zeros(3, 7)
        torch.testing.assert_close(newton_schulz_orthogonalize(zero), zero)
        x = torch.randn(7, 3)
        torch.testing.assert_close(newton_schulz_orthogonalize(x), newton_schulz_orthogonalize(x.mT).mT)

    def test_adamw_matches_torch_and_scheduler_accepts_all_groups(self):
        model = TinyModel()
        optimizer = build_optimizer(model, adamw_lr=0.002, weight_decay=0.1)
        param = model.embedding.weight
        expected = nn.Parameter(param.detach().clone())
        reference = torch.optim.AdamW([expected], lr=0.002, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, foreach=False)
        for _ in range(3):
            param.grad = torch.randn_like(param)
            expected.grad = param.grad.clone()
            optimizer.step()
            reference.step()
            torch.testing.assert_close(param, expected)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.00015)

    def test_state_dict_resume_matches_uninterrupted_updates(self):
        model = TinyModel()
        optimizer = build_optimizer(model)
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()
        resumed_model = copy.deepcopy(model)
        resumed = build_optimizer(resumed_model)
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for _ in range(2):
            for a, b in zip(model.parameters(), resumed_model.parameters()):
                a.grad = torch.randn_like(a)
                b.grad = a.grad.clone()
            optimizer.step()
            resumed.step()
            for a, b in zip(model.parameters(), resumed_model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_nonfinite_rejects_entire_step_before_mutation(self):
        model = TinyModel()
        optimizer = build_optimizer(model)
        before = [param.detach().clone() for param in model.parameters()]
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        model.dt_bias.grad[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            optimizer.step()
        self.assertFalse(optimizer.state)
        for param, original in zip(model.parameters(), before):
            torch.testing.assert_close(param, original)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text in SPECIAL_TOKENS:
            return [SPECIAL_TOKENS.index(text)]
        return [ord(char) + 16 for char in text]


class DataTests(unittest.TestCase):
    def test_best_fit_preserves_complete_samples_and_segment_positions(self):
        samples = [dict(input_ids=[token] * length, labels=[-100] + [token] * (length - 1)) for token, length in ((11, 6), (12, 4), (13, 3), (14, 3))]
        batch = PackedCollator(seq_length=10, pad_token_id=0)(samples)
        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["segment_ids"].tolist(), [[0] * 6 + [1] * 4, [0] * 3 + [1] * 3 + [-1] * 4])
        self.assertEqual(batch["position_ids"][0].tolist(), list(range(6)) + list(range(4)))
        self.assertEqual(batch["labels"][0].tolist(), samples[0]["labels"] + samples[1]["labels"])
        self.assertEqual(batch["attention_mask"].sum().item(), 16)
        self.assertTrue(bool((batch["labels"][~batch["attention_mask"]] == -100).all()))

    def test_packing_rejects_oversized_or_incompatible_tokens(self):
        with self.assertRaisesRegex(ValueError, "not truncated"):
            best_fit_pack([{"input_ids": [1] * 11}], seq_length=10, pad_token_id=0)
        with self.assertRaises(ValueError):
            best_fit_pack([{"input_ids": [1, 2], "labels": [1]}], seq_length=10, pad_token_id=0)
        with self.assertRaises(ValueError):
            best_fit_pack([{"input_ids": [1, 10]}], seq_length=10, pad_token_id=0, vocab_size=10)

    def test_context_stage_boundaries_use_consumed_tokens(self):
        curriculum = ContextCurriculum([ContextStage(2048, 100), ContextStage(4096, 200), ContextStage(10240, 300)])
        for tokens, length in ((0, 2048), (99, 2048), (100, 4096), (299, 4096), (300, 10240), (600, 10240)):
            self.assertEqual(curriculum.seq_length(tokens), length)
        self.assertFalse(curriculum.finished(599))
        self.assertTrue(curriculum.finished(600))
        with self.assertRaises(ValueError):
            ContextCurriculum([ContextStage(4096, 100), ContextStage(2048, 100)])

    def test_teacher_sft_preserves_template_and_mixes_modes(self):
        template = ChatTemplate(FakeTokenizer())
        messages = [{"role": "user", "content": "2+3?"}]
        records = [dict(messages=messages, answer="5", reasoning="2+3=5", answer_verified=True)]
        reasoning = encode_distillation_records(template, records)
        self.assertEqual(reasoning[0], template.encode_sft(messages, "5", reasoning="2+3=5"))
        direct = [template.encode_sft(messages, "5")]
        mixed = mix_sft_examples(direct, reasoning, reasoning_fraction=0.4, total_samples=10, seed=19)
        self.assertEqual(sum(sample == reasoning[0] for sample in mixed), 4)
        self.assertEqual(mixed, mix_sft_examples(direct, reasoning, reasoning_fraction=0.4, total_samples=10, seed=19))
        with self.assertRaises(ValueError):
            encode_distillation_records(template, [dict(messages=messages, answer="5")])
        with self.assertRaisesRegex(ValueError, "not truncated"):
            encode_distillation_records(template, records, max_sample_tokens=2)

    def test_retrieval_probe_only_crops_distractors_and_supervises_answer(self):
        probe = build_retrieval_probe(needle_ids=[5, 6, 7], distractor_ids=[1, 2], query_ids=[8, 9], answer_ids=[10, 11], prefix_ids=[3], seq_length=20, depth=0.5)
        self.assertEqual(len(probe["input_ids"]), 20)
        self.assertEqual(probe["input_ids"][-4:], [8, 9, 10, 11])
        self.assertEqual(probe["input_ids"][7:10], [5, 6, 7])
        self.assertEqual(probe["labels"], [-100] * 18 + [10, 11])
        with self.assertRaisesRegex(ValueError, "not truncated"):
            build_retrieval_probe(needle_ids=[5, 6, 7], distractor_ids=[1], query_ids=[8], answer_ids=[9], seq_length=4)


if __name__ == "__main__":
    unittest.main()
