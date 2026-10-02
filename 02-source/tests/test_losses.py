"""Numerical and dispatch checks for filtered CE and strict CUDA CCE."""

import unittest
from unittest.mock import Mock, patch
from types import ModuleType
import warnings

import torch
from torch.nn import functional as F

from hybrid100m import losses
from hybrid100m.losses import chunked_cross_entropy, cross_entropy_stats


class FilteredLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        torch.set_num_threads(2)

    def tensors(self):
        hidden = torch.randn(2, 6, 8, requires_grad=True)
        weight = torch.randn(19, 8, requires_grad=True)
        targets = torch.tensor([[-100, -100, 4, 7, 2, -100], [-100, 13, -100, 3, -100, 8]])
        return hidden, weight, targets

    def test_loss_statistics_and_gradients_match_dense_ce(self):
        for size in (1, 4, 128):
            hidden, weight, targets = self.tensors()
            mean, total, count = cross_entropy_stats(hidden, weight, targets, size, "chunked")
            dense = F.cross_entropy(F.linear(hidden, weight).float().reshape(-1, 19),
                                    targets.reshape(-1), ignore_index=-100, reduction="sum")
            self.assertEqual(count.dtype, torch.int64)
            self.assertEqual(count.item(), 6)
            torch.testing.assert_close(total, dense, atol=1e-5, rtol=1e-6)
            torch.testing.assert_close(mean, dense / count, atol=1e-6, rtol=1e-6)
            left = torch.autograd.grad(mean, (hidden, weight), retain_graph=True)
            right = torch.autograd.grad(dense / count, (hidden, weight))
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, atol=2e-7, rtol=1e-5)
            self.assertEqual(torch.count_nonzero(left[0][targets.eq(-100)]).item(), 0)

    def test_only_supervised_rows_reach_projection(self):
        hidden, weight, targets = self.tensors()
        with patch("hybrid100m.losses.F.linear", wraps=F.linear) as linear, torch.no_grad():
            cross_entropy_stats(hidden, weight, targets, 4, "chunked")
        self.assertEqual([call.args[0].shape[0] for call in linear.call_args_list], [4, 2])
        self.assertEqual(sum(call.args[0].shape[0] for call in linear.call_args_list), 6)

    def test_all_ignored_produces_zero_grad_without_projection_or_package(self):
        for backend in ("chunked", "auto", "cce"):
            hidden, weight, targets = self.tensors()
            targets.fill_(-100)
            with patch("hybrid100m.losses.F.linear") as linear, patch("hybrid100m.losses._load_cce") as loader:
                mean, total, count = cross_entropy_stats(hidden, weight, targets, 4, backend)
                mean.backward()
            self.assertEqual((mean.item(), total.item(), count.item()), (0, 0, 0))
            linear.assert_not_called()
            loader.assert_not_called()
            self.assertIsNotNone(hidden.grad)
            self.assertIsNotNone(weight.grad)
            self.assertEqual(torch.count_nonzero(hidden.grad).item(), 0)
            self.assertEqual(torch.count_nonzero(weight.grad).item(), 0)

    def test_cpu_cce_and_auto_are_chunked_correctness_paths(self):
        hidden, weight, targets = self.tensors()
        expected = chunked_cross_entropy(hidden, weight, targets, 4)
        with patch("hybrid100m.losses._load_cce", side_effect=AssertionError("CPU must not load CUDA CCE")):
            for backend in ("cce", "auto"):
                actual, _, _ = cross_entropy_stats(hidden, weight, targets, 4, backend)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_noncontiguous_inputs_and_empty_token_axes(self):
        hidden, weight, targets = self.tensors()
        sliced = hidden[:, ::2]
        actual = chunked_cross_entropy(sliced, weight, targets[:, ::2], 4)
        expected = F.cross_entropy(F.linear(sliced, weight).float().reshape(-1, 19),
                                   targets[:, ::2].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(actual, expected)
        mean, _, count = cross_entropy_stats(hidden[:, :0], weight, targets[:, :0], 4)
        mean.backward()
        self.assertEqual(mean.item(), 0)
        self.assertEqual(count.item(), 0)

    def test_cpu_bf16_autocast_retains_fp32_master_gradients(self):
        hidden, weight, targets = self.tensors()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mean, total, count = cross_entropy_stats(hidden, weight, targets, 4)
            dense = F.cross_entropy(F.linear(hidden, weight).float().reshape(-1, 19),
                                    targets.reshape(-1), ignore_index=-100, reduction="sum") / count
        torch.testing.assert_close(mean, dense, atol=0.02, rtol=0.005)
        mean.backward()
        self.assertEqual(total.dtype, torch.float32)
        self.assertEqual(hidden.grad.dtype, torch.float32)
        self.assertEqual(weight.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(weight.grad).all())

    def test_invalid_arguments_are_rejected(self):
        hidden, weight, targets = self.tensors()
        for chunk in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                cross_entropy_stats(hidden, weight, targets, chunk)
        with self.assertRaises(ValueError):
            cross_entropy_stats(hidden, weight, targets, 4, "unknown")
        with self.assertRaises(ValueError):
            cross_entropy_stats(hidden, weight[:, :3], targets, 4)
        with self.assertRaises(ValueError):
            cross_entropy_stats(hidden, weight, targets.float(), 4)


class CCEDispatchTests(unittest.TestCase):
    def test_loader_rejects_old_cce_api_without_fp32_accumulation(self):
        module = ModuleType("cut_cross_entropy")
        def old_cce(hidden, weight, targets, reduction="mean", filter_eps="auto"):
            raise AssertionError("Old CCE API must never execute")
        module.linear_cross_entropy = old_cce
        losses._load_cce.cache_clear()
        try:
            with patch.dict("sys.modules", {"cut_cross_entropy": module}):
                with self.assertRaisesRegex(RuntimeError, "FP32 accumulation API"):
                    losses._load_cce()
        finally:
            losses._load_cce.cache_clear()

    def test_autocast_explicitly_casts_inputs_and_passes_exact_gradient_options(self):
        hidden = torch.randn(5, 8, requires_grad=True)
        weight = torch.randn(19, 8, requires_grad=True)
        targets = torch.tensor([3, 4, 2, 1, 17])
        fn = Mock(side_effect=lambda h, w, y, **kwargs: h.float().sum() + w.float().sum())
        with patch("hybrid100m.losses.torch.is_autocast_enabled", return_value=True), \
             patch("hybrid100m.losses.torch.get_autocast_dtype", return_value=torch.bfloat16):
            total = losses._cce_sum(hidden, weight, targets, fn)
        for value in fn.call_args.args[:2]:
            self.assertEqual(value.dtype, torch.bfloat16)
            self.assertTrue(value.is_contiguous())
        self.assertEqual(fn.call_args.kwargs, {
            "reduction": "sum", "impl": "cce", "filter_eps": None,
            "accum_e_fp32": True, "accum_c_fp32": True,
        })
        total.backward()
        self.assertEqual(hidden.dtype, torch.float32)
        self.assertEqual(weight.dtype, torch.float32)
        self.assertEqual(weight.grad.dtype, torch.float32)

    def test_fp32_cce_without_autocast_is_rejected(self):
        with patch("hybrid100m.losses.torch.is_autocast_enabled", return_value=False):
            with self.assertRaises(RuntimeError):
                losses._cce_sum(torch.ones(2, 4), torch.ones(9, 4), torch.tensor([1, 2]), Mock())

    def test_auto_fallback_only_for_absent_cce_package(self):
        with patch("hybrid100m.losses._load_cce", side_effect=losses._CCENotInstalled("missing")):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                self.assertIsNone(losses._resolve_cuda_backend("auto"))
            self.assertEqual(len(recorded), 1)
            self.assertIn("checkpointed CE", str(recorded[0].message))
            with self.assertRaises(RuntimeError):
                losses._resolve_cuda_backend("cce")
        with patch("hybrid100m.losses._load_cce", side_effect=ModuleNotFoundError("missing Triton", name="triton")):
            with self.assertRaises(ModuleNotFoundError):
                losses._resolve_cuda_backend("auto")

    def test_kernel_runtime_failures_are_not_swallowed(self):
        hidden = torch.ones(2, 4, dtype=torch.bfloat16)
        weight = torch.ones(9, 4, dtype=torch.bfloat16)
        fn = Mock(side_effect=RuntimeError("kernel compilation failed"))
        with patch("hybrid100m.losses.torch.is_autocast_enabled", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "kernel compilation failed"):
                losses._cce_sum(hidden, weight, torch.tensor([1, 2]), fn)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable")
class CCECUDAComparisonTests(unittest.TestCase):
    def test_fused_matches_dense_loss_and_fp32_parameter_gradients(self):
        torch.manual_seed(59)
        hidden = torch.randn(2, 37, 64, device="cuda", requires_grad=True)
        weight = torch.randn(256, 64, device="cuda", requires_grad=True)
        targets = torch.randint(0, 256, (2, 37), device="cuda")
        targets[:, :9] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual, total, count = cross_entropy_stats(hidden, weight, targets, 17, "cce")
            expected = F.cross_entropy(F.linear(hidden, weight).float().reshape(-1, 256),
                                       targets.reshape(-1), ignore_index=-100)
        self.assertEqual(count.item(), 56)
        torch.testing.assert_close(total / count, actual, atol=0, rtol=0)
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.002)
        left = torch.autograd.grad(actual, (hidden, weight), retain_graph=True)
        right = torch.autograd.grad(expected, (hidden, weight))
        for a, b in zip(left, right):
            self.assertEqual(a.dtype, torch.float32)
            torch.testing.assert_close(a, b, atol=0.001, rtol=0.05)
        self.assertEqual(torch.count_nonzero(left[0][targets.eq(-100)]).item(), 0)


if __name__ == "__main__":
    unittest.main()
