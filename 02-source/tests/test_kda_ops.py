"""KDA recurrence, packing isolation, and safe training dispatch checks."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hybrid100m.ops import gated_delta, kda, recurrent_delta_reference, recurrent_kda_reference


class KDAReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def inputs(self, b=2, t=5, h=2, kd=3, vd=4):
        q, k = [torch.randn(b, t, h, kd, requires_grad=True) for _ in range(2)]
        v = torch.randn(b, t, h, vd, requires_grad=True)
        g = (-torch.rand(b, t, h, kd)).requires_grad_()
        beta = torch.rand(b, t, h, requires_grad=True)
        initial = torch.randn(b, h, kd, vd, requires_grad=True)
        return q, k, v, g, beta, initial

    def test_matrix_recurrence_outputs_final_state_and_all_gradients(self):
        inputs = self.inputs()
        q, k, v, g, beta, initial = inputs
        actual, state = recurrent_kda_reference(*inputs)
        # Independent formulation with explicit diagonal decay and erase matrices.
        qn = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        kn = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        memory, expected = initial, []
        for i in range(q.shape[1]):
            key = kn[:, i]
            erase = torch.eye(q.shape[-1]) - beta[:, i, :, None, None] * torch.einsum(
                "bhk,bhl->bhkl", key, key,
            )
            memory = erase @ torch.diag_embed(g[:, i].exp()) @ memory + beta[:, i, :, None, None] * torch.einsum(
                "bhk,bhv->bhkv", key, v[:, i],
            )
            expected.append(torch.einsum("bhk,bhkv->bhv", qn[:, i] / math.sqrt(q.shape[-1]), memory))
        expected = torch.stack(expected, dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(state, memory, atol=1e-6, rtol=1e-5)
        left = torch.autograd.grad(actual.square().sum() + state.square().sum(), inputs, retain_graph=True)
        right = torch.autograd.grad(expected.square().sum() + memory.square().sum(), inputs)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-4)

    def test_headwise_decay_reduces_exactly_to_gdn(self):
        q, key, v, _, beta, initial = self.inputs()
        scalar = (-torch.rand(q.shape[:3])).requires_grad_()
        a, sa = recurrent_kda_reference(q, key, v, scalar[..., None].expand_as(q), beta, initial)
        b, sb = recurrent_delta_reference(q, key, v, scalar, beta, initial)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(sa, sb, atol=0, rtol=0)
        left = torch.autograd.grad(a.square().sum(), scalar, retain_graph=True)[0]
        right = torch.autograd.grad(b.square().sum(), scalar)[0]
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    def test_packed_matches_independent_sequences_and_gradients(self):
        q, key, v, g, beta, _ = self.inputs(b=1, t=7)
        initial = torch.randn(3, 2, 3, 4, requires_grad=True)
        cu = torch.tensor([0, 2, 6, 7], dtype=torch.int32)
        for oracle, gate in ((recurrent_kda_reference, g), (recurrent_delta_reference, g[..., 0])):
            packed, states = oracle(q, key, v, gate, beta, initial, cu_seqlens=cu)
            pieces, final = [], []
            for i, (start, end) in enumerate(zip(cu.tolist(), cu.tolist()[1:])):
                out, state = oracle(q[:, start:end], key[:, start:end], v[:, start:end],
                                    gate[:, start:end], beta[:, start:end], initial[i:i + 1])
                pieces.append(out)
                final.append(state)
            expected, expected_state = torch.cat(pieces, dim=1), torch.cat(final)
            torch.testing.assert_close(packed, expected, atol=0, rtol=0)
            torch.testing.assert_close(states, expected_state, atol=0, rtol=0)
            inputs = (q, key, v, g, beta, initial)
            left = torch.autograd.grad(packed.square().sum() + states.square().sum(), inputs, retain_graph=True)
            right = torch.autograd.grad(expected.square().sum() + expected_state.square().sum(), inputs, retain_graph=True)
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_packed_second_document_does_not_read_first_document(self):
        q, key, v, g, beta, _ = self.inputs(b=1, t=7)
        cu = torch.tensor([0, 3, 7])
        a, _ = recurrent_kda_reference(q, key, v, g, beta, cu_seqlens=cu)
        changed = v.detach().clone()
        changed[:, :3] += 1000
        b, _ = recurrent_kda_reference(q, key, changed, g, beta, cu_seqlens=cu)
        torch.testing.assert_close(a[:, 3:], b[:, 3:], atol=0, rtol=0)
        grad = torch.autograd.grad(a[:, 3:].sum(), v)[0]
        self.assertEqual(torch.count_nonzero(grad[:, :3]).item(), 0)

    def test_bf16_rounding_state_dtype_and_cache_continuation(self):
        q, key, v, g, beta, initial = self.inputs(b=1, t=7)
        q, key, v, beta = (x.to(torch.bfloat16) for x in (q, key, v, beta))
        full, full_state = recurrent_kda_reference(q, key, v, g, beta, initial)
        first, cache = recurrent_kda_reference(q[:, :3], key[:, :3], v[:, :3], g[:, :3], beta[:, :3], initial)
        saved = cache.clone()
        last, state = recurrent_kda_reference(q[:, 3:], key[:, 3:], v[:, 3:], g[:, 3:], beta[:, 3:], cache)
        torch.testing.assert_close(cache, saved, atol=0, rtol=0)
        torch.testing.assert_close(torch.cat((first, last), dim=1), full, atol=0, rtol=0)
        torch.testing.assert_close(state, full_state, atol=0, rtol=0)
        self.assertEqual(full.dtype, torch.bfloat16)
        self.assertEqual(state.dtype, torch.float32)

    def test_reference_rejects_invalid_boundaries_and_state_counts(self):
        q, key, v, g, beta, initial = self.inputs(b=1, t=5)
        for cu in (torch.tensor([1, 5]), torch.tensor([0, 3, 3, 5]), torch.tensor([0, 6]),
                   torch.tensor([0.0, 5.0]), torch.tensor([[0, 5]])):
            with self.assertRaises(ValueError):
                recurrent_kda_reference(q, key, v, g, beta, cu_seqlens=cu)
        with self.assertRaises(ValueError):
            recurrent_kda_reference(q, key, v, g, beta, initial, cu_seqlens=torch.tensor([0, 2, 5]))

    def test_adapter_reference_and_strict_cuda_requirement(self):
        q, key, v, g, beta, initial = self.inputs()
        expected, state = recurrent_kda_reference(q, key, v, g, beta, initial)
        actual, cache = kda(q, key, v, g, beta, backend="reference", initial_state=initial, output_final_state=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(cache, state, atol=0, rtol=0)
        self.assertIsNone(kda(q, key, v, g, beta, backend="reference")[1])
        with self.assertRaises(RuntimeError):
            kda(q, key, v, g, beta, backend="fla")
        with self.assertRaises(ValueError):
            kda(q, key, v, g, beta, backend="unknown")


class KDADispatchTests(unittest.TestCase):
    def test_training_uses_chunk_even_for_one_token(self):
        class FakeTensor:
            device = SimpleNamespace(type="cuda")
            dtype = torch.bfloat16
            shape = (1, 1, 4, 128)
            def contiguous(self): return self
            def float(self): return self
        x = FakeTensor()
        with patch("hybrid100m.ops._fla_kda_ops") as ops:
            chunk, recurrent = Mock(return_value=(x, x)), Mock(return_value=(x, x))
            ops.return_value = chunk, recurrent
            kda(x, x, x, x, x, backend="fla", output_final_state=True)
            chunk.assert_called_once()
            recurrent.assert_not_called()
            self.assertTrue(chunk.call_args.kwargs["use_qk_l2norm_in_kernel"])
            self.assertEqual(chunk.call_args.kwargs["scale"], 128 ** -0.5)
            with torch.no_grad():
                kda(x, x, x, x, x, backend="fla")
            recurrent.assert_called_once()
            x.shape = (1, 3, 4, 128)
            with torch.no_grad():
                kda(x, x, x, x, x, backend="fla")
            self.assertEqual(chunk.call_count, 2)
            self.assertEqual(recurrent.call_count, 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable")
class KDACUDATests(unittest.TestCase):
    def test_chunk_matches_reference_with_packing_and_backward(self):
        torch.manual_seed(43)
        # Boundaries are deliberately not multiples of the 64-token chunk size.
        q, key, v = [torch.randn(1, 79, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                     for _ in range(3)]
        g = (-torch.rand(1, 79, 2, 32, device="cuda") * 0.1).requires_grad_()
        beta = torch.rand(1, 79, 2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        initial = torch.randn(2, 2, 32, 32, device="cuda", requires_grad=True)
        cu = torch.tensor([0, 11, 79], device="cuda", dtype=torch.long)
        inputs = (q, key, v, g, beta, initial)
        actual, state = kda(q, key, v, g, beta, backend="fla", initial_state=initial,
                            output_final_state=True, cu_seqlens=cu)
        expected, expected_state = recurrent_kda_reference(*inputs, cu_seqlens=cu)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.04)
        torch.testing.assert_close(state, expected_state, atol=0.02, rtol=0.04)
        left = torch.autograd.grad(actual.float().square().mean() + state.square().mean(), inputs, retain_graph=True)
        right = torch.autograd.grad(expected.float().square().mean() + expected_state.square().mean(), inputs)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=0.001, rtol=0.08)


if __name__ == "__main__":
    unittest.main()
