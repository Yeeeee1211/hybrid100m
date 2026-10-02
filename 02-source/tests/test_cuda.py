"""Isolated synthetic validation on the user's CUDA host, without a trainer.

HYBRID100M_FULL_GPU_TEST=1 also runs a 100M backbone + MTP forward/backward
and one real Muon + AdamW update, then backward with resident optimizer state.
HYBRID100M_GPU_SEQ_LEN (default 10240) and HYBRID100M_GPU_BATCH_SIZE (default 1)
allow checking the actual RTX 4090 memory budget before launching real training.
HYBRID100M_GPU_LOSS_CHUNK_SIZE optionally overrides the config's CE chunk size
for memory comparisons; it does not alter the saved config or model architecture.
HYBRID100M_GPU_LOSS_BACKEND can select cce or chunked (default: config value).
Missing FLA, CCE or Flash on a CUDA machine is a FAILURE when selected, not a
skip or silent backend fallback. No corpus is consumed and no checkpoint saved.
"""

import os
from dataclasses import replace
from pathlib import Path
import time
import unittest

import torch

from hybrid100m import HybridConfig, HybridForCausalLM
from hybrid100m.data import PackedCollator
from hybrid100m.ops import gated_delta, recurrent_delta_reference
from hybrid100m.optim import build_optimizer


def _loss_backend(default):
    backend = os.getenv("HYBRID100M_GPU_LOSS_BACKEND", default)
    if backend not in {"cce", "chunked"}:
        raise ValueError("HYBRID100M_GPU_LOSS_BACKEND must be cce or chunked")
    return backend


@unittest.skipUnless(torch.cuda.is_available(), "CUDA host required")
class CUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        if props.major < 8 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This validation requires native CUDA BF16 on Ampere or newer")
        print(f"\nGPU={props.name}, SM={props.major}.{props.minor}, "
              f"VRAM={props.total_memory / 2**30:.2f} GiB, "
              f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")

    def setUp(self):
        torch.manual_seed(19)

    def test_fla_forward_state_and_backward_match_reference(self):
        shape = (2, 79, 2, 32)  # crosses the kernel's usual 64-token chunk boundary
        source = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.25 for _ in range(3)]
        source += [-torch.rand(shape[:3], device="cuda") * 0.1,
                   torch.rand(shape[:3], device="cuda", dtype=torch.bfloat16)]
        state = torch.randn(2, 2, 32, 32, device="cuda") * 0.1
        source.append(state)
        actual_inputs = [x.detach().clone().requires_grad_() for x in source]
        expected_inputs = [x.detach().clone().requires_grad_() for x in source]
        actual, actual_state = gated_delta(
            *actual_inputs[:5], initial_state=actual_inputs[5], backend="fla", output_final_state=True,
        )
        expected, expected_state = recurrent_delta_reference(*expected_inputs)
        self.assertEqual(actual_state.dtype, torch.float32)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.002, rtol=0.04)
        torch.testing.assert_close(actual_state, expected_state, atol=0.006, rtol=0.04)
        weights = torch.randn_like(actual.float())
        a_loss = (actual.float() * weights).sum() + actual_state.square().sum() * 0.1
        e_loss = (expected.float() * weights).sum() + expected_state.square().sum() * 0.1
        a_grad = torch.autograd.grad(a_loss, actual_inputs)
        e_grad = torch.autograd.grad(e_loss, expected_inputs)
        for a, e in zip(a_grad, e_grad):
            self.assertTrue(bool(a.isfinite().all()))
            # BF16 accumulation order differs across chunk and recurrence kernels.
            relative_rms = (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-6)
            self.assertLess(float(relative_rms), 0.08)

    def test_kda_yarn_mtp_training_padding_packing_and_cached_chunks(self):
        cfg = HybridConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            linear_num_heads=2, linear_head_dim=128, sdpa_backend="flash", loss_chunk_size=32,
            linear_attention="kda", linear_gate_rank=32, linear_backend="fla",
            rope_type="yarn", mtp_num_layers=1, loss_backend=_loss_backend("cce"),
        )
        model = HybridForCausalLM(cfg).cuda()
        ids = torch.randint(0, cfg.vocab_size, (2, 131), device="cuda")
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[0, 90:] = False
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(ids, labels=ids, attention_mask=mask).loss
        loss.backward()
        self.assertTrue(bool(loss.isfinite()))
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters()))
        model.zero_grad(set_to_none=True)
        # Two complete samples in one row; positions and every attention/state
        # boundary must reset at the second segment, without a dense T^2 mask.
        samples = [dict(input_ids=ids[0, :67].tolist()), dict(input_ids=ids[1, :64].tolist())]
        packed = {name: tensor.cuda() for name, tensor in PackedCollator(133, 0, cfg.vocab_size)(samples).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            packed_loss = model(**packed).loss
        packed_loss.backward()
        self.assertTrue(bool(packed_loss.isfinite()))
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters()))
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            packed_inputs = {name: tensor for name, tensor in packed.items() if name != "labels"}
            packed_logits = model(**packed_inputs).logits
            first = model(ids[0:1, :67]).logits
            second = model(ids[1:2, :64]).logits
            torch.testing.assert_close(packed_logits[0, :67].float(), first[0].float(), atol=0.01, rtol=0.05)
            torch.testing.assert_close(packed_logits[0, 67:131].float(), second[0].float(), atol=0.01, rtol=0.05)
            altered = {name: tensor.clone() for name, tensor in packed_inputs.items()}
            altered["input_ids"][0, :67] = (altered["input_ids"][0, :67] + 1) % cfg.vocab_size
            altered_logits = model(**altered).logits
            torch.testing.assert_close(packed_logits[0, 67:131].float(), altered_logits[0, 67:131].float(), atol=0.01, rtol=0.05)
            expected = model(ids).logits
            padded = model(ids, attention_mask=mask).logits
            torch.testing.assert_close(expected[0, :90].float(), padded[0, :90].float(), atol=0.01, rtol=0.05)
            cache, pieces, start = None, [], 0
            for count in (65, 1, 65):
                result = model(ids[:, start:start + count], cache=cache, use_cache=True)
                pieces.append(result.logits)
                cache = result.cache
                start += count
            actual = torch.cat(pieces, dim=1)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.025, rtol=0.05)

    @unittest.skipUnless(os.getenv("HYBRID100M_FULL_GPU_TEST") == "1", "Set HYBRID100M_FULL_GPU_TEST=1 for full model memory check")
    def test_full_100m_mtp_muon_training_memory(self):
        cfg = HybridConfig.load(Path(__file__).resolve().parents[1] / "config.json")
        cfg = replace(
            cfg, loss_chunk_size=int(os.getenv("HYBRID100M_GPU_LOSS_CHUNK_SIZE", str(cfg.loss_chunk_size))),
            loss_backend=_loss_backend(cfg.loss_backend),
        )
        length = int(os.getenv("HYBRID100M_GPU_SEQ_LEN", "10240"))
        batch = int(os.getenv("HYBRID100M_GPU_BATCH_SIZE", "1"))
        if not 2 <= length <= cfg.max_position_embeddings or batch < 1:
            raise ValueError("Require 2 <= GPU_SEQ_LEN <= context limit and GPU_BATCH_SIZE >= 1")
        model = HybridForCausalLM(cfg).cuda()
        backbone_parameters = model.num_parameters(include_mtp=False)
        total_parameters = model.num_parameters()
        self.assertEqual(backbone_parameters, 100_162_108)
        # One disposable synthetic update allocates Muon momentum and AdamW
        # moments, which a forward/backward-only check misses.
        optimizer = build_optimizer(model, muon_lr=1e-4, adamw_lr=1e-4)
        ids = torch.randint(0, cfg.vocab_size, (batch, length), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(ids, labels=ids)
        result.loss.backward()
        self.assertTrue(bool(result.loss.isfinite()))
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        # A second forward/backward sees resident moments AND retained gradients,
        # approximating the memory overlap of gradient accumulation. No second
        # optimizer update is needed. This is still only a synthetic smoke check.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            second = model(ids, labels=ids)
        second.loss.backward()
        self.assertTrue(bool(second.loss.isfinite()))
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters()))
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"\nbackbone_parameters={backbone_parameters:,}, "
              f"mtp_parameters={total_parameters - backbone_parameters:,}, "
              f"total_training_parameters={total_parameters:,}, batch={batch}, tokens={length}, "
              f"linear_attention={cfg.linear_attention}, rope={cfg.rope_type}, mtp_layers={cfg.mtp_num_layers}, "
              f"loss_backend={cfg.loss_backend}, loss_chunk={cfg.loss_chunk_size}: "
              f"loss={result.loss.item():.4f}, "
              f"elapsed={time.perf_counter() - start:.2f}s, "
              f"peak_allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
              f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB, "
              f"free_after={free / 2**30:.2f}/{total / 2**30:.2f} GiB "
              "(FP32 Muon momentum/AdamW states + resident gradients; includes cold kernel compilation)")


if __name__ == "__main__":
    unittest.main()
