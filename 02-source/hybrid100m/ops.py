"""Gated delta rule and KDA: differentiable oracles and strict FLA adapters.

All inputs use [batch, time, heads, dimension]. The state uses [B, H, K, V].
The reference recurrence deliberately avoids a [T, T] tensor and preserves
autograd, but its Python time loop is ONLY intended for small correctness tests.
"""

from functools import lru_cache
import torch


def _validate_reference_inputs(q, k, v, g, beta, initial_state, cu_seqlens, channel_decay):
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("q/k/v must use [B,T,H,D] with equal batch, time, and heads")
    if q.shape[1] < 1 or min(q.shape[0], q.shape[2], q.shape[3], v.shape[3]) < 1:
        raise ValueError("batch, sequence length, heads, and head dimensions must be positive")
    expected_gate = q.shape if channel_decay else q.shape[:3]
    if g.shape != expected_gate or beta.shape != q.shape[:3]:
        raise ValueError("g must be [B,T,H,K] for KDA or [B,T,H] for GDN; beta must be [B,T,H]")
    if any(x.device != q.device for x in (k, v, g, beta)):
        raise ValueError("all recurrence inputs must be on the same device")
    n = q.shape[0]
    bounds = None
    if cu_seqlens is not None:
        _validate_packed_metadata(q, cu_seqlens)
        bounds = cu_seqlens.detach().cpu().tolist()
        if bounds[0] != 0 or bounds[-1] != q.shape[1] or any(
            right <= left for left, right in zip(bounds, bounds[1:])
        ):
            raise ValueError("cu_seqlens must increase strictly from 0 to the total sequence length")
        n = len(bounds) - 1
    if initial_state is not None:
        if initial_state.shape != (n, q.shape[2], q.shape[3], v.shape[3]):
            raise ValueError("initial_state must be [N,H,K,V], with one state per sequence")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
    return bounds


def _validate_packed_metadata(q, cu_seqlens):
    # The model validates boundary values once. Avoid repeated GPU-to-CPU
    # synchronization in every layer; the reference oracle checks the values too.
    if q.shape[0] != 1:
        raise ValueError("cu_seqlens requires flattened inputs with batch size 1")
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional cumulative-length tensor")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must contain integer cumulative lengths")


def _recurrent_reference(q, k, v, g, beta, initial_state, cu_seqlens, *, channel_decay):
    bounds = _validate_reference_inputs(q, k, v, g, beta, initial_state, cu_seqlens, channel_decay)
    if bounds is not None:
        outputs, states = [], []
        # Each document starts from its own independent state. Concatenating
        # these differentiable recurrences exactly matches independent samples.
        for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
            output, state = _recurrent_reference(
                q[:, start:end], k[:, start:end], v[:, start:end], g[:, start:end], beta[:, start:end],
                None if initial_state is None else initial_state[i:i + 1], None,
                channel_decay=channel_decay,
            )
            outputs.append(output)
            states.append(state)
        return torch.cat(outputs, dim=1), torch.cat(states, dim=0)
    dtype = q.dtype
    with torch.autocast(device_type=q.device.type, enabled=False):
        qf, kf = q.float(), k.float()
        qf = (qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)).to(dtype).float()
        kf = (kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)).to(dtype).float()
        qf = qf * (q.shape[-1] ** -0.5)
        vf, gf, bf = v.float(), g.float(), beta.float()
        b, t, h, kd = q.shape
        state = (
            torch.zeros(b, h, kd, v.shape[-1], device=q.device, dtype=torch.float32)
            if initial_state is None else initial_state.float()
        )
        outputs = []
        for i in range(t):
            decay = gf[:, i].exp()
            state = state * (decay[..., None] if channel_decay else decay[..., None, None])
            predicted = (state * kf[:, i, :, :, None]).sum(-2)
            correction = bf[:, i, :, None] * (vf[:, i] - predicted)
            state = state + kf[:, i, :, :, None] * correction[..., None, :]
            outputs.append((state * qf[:, i, :, :, None]).sum(-2))
        return torch.stack(outputs, dim=1).to(dtype), state


def recurrent_delta_reference(q, k, v, g, beta, initial_state=None, cu_seqlens=None):
    """L2-normalized gated delta rule, FP32 state, differentiable in all inputs.

    S_bar = exp(g_t) * S_(t-1)
    S_t = S_bar + k_t outer [beta_t * (v_t - k_t @ S_bar)]
    y_t = (q_t / sqrt(K)) @ S_t

    L2 normalization rounds back to input dtype to match FLA's kernel contract.
    g is log decay (not a sigmoid, and not a cumulative sum).
    """
    return _recurrent_reference(q, k, v, g, beta, initial_state, cu_seqlens, channel_decay=False)


def recurrent_kda_reference(q, k, v, g, beta, initial_state=None, cu_seqlens=None):
    """Kimi Delta Attention with per-key-channel log decay and FP32 state.

    S_bar = diag(exp(g_t)) @ S_(t-1)
    S_t = (I - beta_t * k_t outer k_t) @ S_bar + beta_t * k_t outer v_t
    y_t = (q_t / sqrt(K)) @ S_t

    g is [B,T,H,K], not a headwise scalar. Q/K normalization and rounding
    match the GDN oracle. Packed inputs [1,total,H,D] use cu_seqlens to reset
    state at every document boundary; final states are returned as [N,H,K,V].
    """
    return _recurrent_reference(q, k, v, g, beta, initial_state, cu_seqlens, channel_decay=True)


@lru_cache(maxsize=1)
def _fla_ops():
    try:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except ImportError as exc:
        raise RuntimeError(
            "FLA backend requested but unavailable. Install a CUDA-compatible "
            "PyTorch/Triton and flash-linear-attention on the training server. "
            "Use linear_backend='reference' only for small correctness tests."
        ) from exc
    return chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


@lru_cache(maxsize=1)
def _fla_kda_ops():
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
    except ImportError as exc:
        raise RuntimeError(
            "KDA backend requested but unavailable. Install CUDA-compatible "
            "PyTorch/Triton and flash-linear-attention>=0.4 with fla.ops.kda. "
            "Use linear_backend='reference' only for small correctness tests."
        ) from exc
    return chunk_kda, fused_recurrent_kda


def _fla_dispatch(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, ops):
    if q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError("FLA requires CUDA BF16/FP16 activations; use torch.autocast('cuda', dtype=torch.bfloat16)")
    if cu_seqlens is not None:
        _validate_packed_metadata(q, cu_seqlens)
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int64).contiguous()
        if initial_state is not None and initial_state.shape[0] != cu_seqlens.numel() - 1:
            raise ValueError("packed initial_state must provide one state per sequence")
    chunk, recurrent = ops()
    # Fused recurrent is forward-only: NEVER select it when autograd is enabled,
    # even for a one-token training batch or model.eval() with gradients enabled.
    fn = recurrent if q.shape[1] == 1 and not torch.is_grad_enabled() else chunk
    out, state = fn(
        q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
        g=g.float().contiguous(), beta=beta.contiguous(),
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state.float().contiguous() if initial_state is not None else None,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        **({"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}),
    )
    return out, state.float() if state is not None else None


def gated_delta(q, k, v, g, beta, *, backend, initial_state=None, output_final_state=False, cu_seqlens=None):
    if backend == "reference":
        out, state = recurrent_delta_reference(q, k, v, g, beta, initial_state, cu_seqlens)
        return out, state if output_final_state else None
    if backend != "fla":
        raise ValueError(f"Unknown delta backend: {backend}")
    return _fla_dispatch(
        q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, _fla_ops,
    )


def kda(q, k, v, g, beta, *, backend, initial_state=None, output_final_state=False, cu_seqlens=None):
    """Strict KDA adapter; training always uses the differentiable chunk kernel."""
    if backend == "reference":
        out, state = recurrent_kda_reference(q, k, v, g, beta, initial_state, cu_seqlens)
        return out, state if output_final_state else None
    if backend != "fla":
        raise ValueError(f"Unknown KDA backend: {backend}")
    return _fla_dispatch(
        q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, _fla_kda_ops,
    )
