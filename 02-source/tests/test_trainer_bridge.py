"""Check existing trainer/core bridges without starting a training run."""

from contextlib import nullcontext
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import train
import train_sft
from hybrid100m.optim import MuonAdamW


class DispatchModel(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.config = SimpleNamespace(linear_attention=attention)
        self.linear = nn.Linear(4, 4)


class StatisticModel(nn.Module):
    """Analytic per-target objective, intentionally with unequal NTP/MTP counts."""
    def __init__(self, mtp_depths=2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(mtp_num_layers=mtp_depths, mtp_loss_weight=0.3)
        self.forward_calls = 0

    def forward(self, ids, *, labels, attention_mask, compute_mtp):
        self.forward_calls += 1
        # The artificial per-token cost is its ID for NTP, and squared ID for
        # MTP; the oracle is analytic and independent of model loss mechanics.
        totals, counts = [], []
        for offset in range(1, self.config.mtp_num_layers + 2 if compute_mtp else 2):
            y = labels[:, offset:]
            mask = attention_mask[:, offset:] & (y != -100)
            values = y[mask].float()
            coefficient = values.sum() if offset == 1 else values.square().sum()
            totals.append(self.weight * coefficient)
            counts.append(mask.sum())
        return SimpleNamespace(loss_sum=totals[0], num_loss_tokens=counts[0],
                               mtp_loss_sums=tuple(totals[1:]), mtp_num_tokens=tuple(counts[1:]))


def samples():
    # Nonignored label[0] exercises the old off-by-one counting error. The
    # third row has no targets and must not change any objective denominator.
    rows = [([10, 11], [10, 11]), ([20, 21, 22, 23, 24], [-100, -100, 22, 23, 24]), ([30], [30])]
    return [(np.asarray(ids, dtype=np.int32), np.asarray(labels, dtype=np.int32)) for ids, labels in rows]


class TrainerBridgeTests(unittest.TestCase):
    def test_optimizer_dispatch_and_legacy_state_resume(self):
        for module in (train, train_sft):
            with self.subTest(module=module.__name__):
                new = module.build_optimizer(DispatchModel("kda"), 3e-4, 0.1)
                self.assertIsInstance(new, MuonAdamW)
                self.assertTrue(any(group["algorithm"] == "muon" for group in new.param_groups))
                old_model = DispatchModel("gated_delta")
                old = module.build_optimizer(old_model, 3e-4, 0.1)
                self.assertIsInstance(old, torch.optim.AdamW)
                self.assertEqual(len(old.param_groups), 2)
                for param in old_model.parameters():
                    param.grad = torch.ones_like(param)
                old.step()
                restored_model = copy.deepcopy(old_model)
                restored = module.build_optimizer(restored_model, 3e-4, 0.1)
                restored.load_state_dict(copy.deepcopy(old.state_dict()))
                for a, b in zip(old_model.parameters(), restored_model.parameters()):
                    a.grad = torch.full_like(a, 2)
                    b.grad = a.grad.clone()
                old.step()
                restored.step()
                for a, b in zip(old_model.parameters(), restored_model.parameters()):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_sft_accumulation_uses_each_objectives_own_target_count(self):
        model = StatisticModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        args = SimpleNamespace(accum=3, grad_clip=1e6, log_every=1000)
        # NTP [11,22,23,24] -> 80/4; MTP depth1 [22,23,24] ->
        # 1589/3; depth2 [23,24] -> 1105/2. Not weighted by NTP counts.
        expected = 80 / 4 + 0.3 * (1589 / 3 + 1105 / 2) / 2
        with patch("train_sft.torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()):
            loss, steps = train_sft.train_epoch(model, samples(), [[0], [1], [2]], optimizer, "cpu", 0,
                                                args, lambda step: 0.01, 0, "test")
        self.assertEqual(steps, 1)
        self.assertEqual(model.forward_calls, 3)
        self.assertAlmostEqual(loss, expected, places=5)
        self.assertAlmostEqual(model.weight.item(), 1 - 0.01 * expected, places=6)
        self.assertEqual(train_sft.group_token_counts(samples(), [[0], [1], [2]]), [1, 3, 0])

    def test_sft_evaluation_is_invariant_to_microbatch_partition(self):
        model = StatisticModel()
        expected = 80 / 4 + 0.3 * (1589 / 3 + 1105 / 2) / 2
        with patch("train_sft.torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()):
            separate = train_sft.evaluate(model, samples(), [[0], [1], [2]], "cpu", 0)
            combined = train_sft.evaluate(model, samples(), [[0, 1, 2]], "cpu", 0)
        self.assertAlmostEqual(separate, expected, places=5)
        self.assertAlmostEqual(combined, expected, places=5)
        self.assertTrue(model.training)

    def test_legacy_sft_without_mtp_uses_ntp_target_count(self):
        model = StatisticModel(mtp_depths=0)
        with patch("train_sft.torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()):
            loss = train_sft.evaluate(model, samples(), [[0], [1], [2]], "cpu", 0)
        self.assertEqual(loss, 20)


if __name__ == "__main__":
    unittest.main()
