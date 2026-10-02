"""Filtered language-model loss with checkpointed or fused projection/CE.

Only supervised token positions enter the LM projection. ``chunked`` is the
portable oracle; CUDA ``cce`` avoids materializing vocabulary-wide logits.
All-ignored batches produce a differentiable zero with zero parameter grads.
CUDA CCE requires the official Apple implementation with FP32 gradient
accumulation options: the CUDA extra pins commit
b7a02791b234e187b524fb1dba6a812d521b203a (25.9.3), rather than the older PyPI
25.1.1 package. See pyproject.toml for the reproducible install source.
"""

from functools import lru_cache
from inspect import signature
import warnings

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class _CCENotInstalled(ImportError):
    pass


@lru_cache(maxsize=1)
def _load_cce():
    try:
        from cut_cross_entropy import linear_cross_entropy
    except ModuleNotFoundError as exc:
        # A broken installed package (e.g. missing Triton) is not an acceptable
        # reason to silently replace the selected training kernel.
        if exc.name != "cut_cross_entropy":
            raise
        raise _CCENotInstalled("cut-cross-entropy is not installed") from exc
    required = {"filter_eps", "accum_e_fp32", "accum_c_fp32"}
    if not required.issubset(signature(linear_cross_entropy).parameters):
        raise RuntimeError(
            "Installed cut-cross-entropy lacks the required FP32 accumulation API. "
            "Install the official Apple commit pinned by pip install -e '.[cuda]'."
        )
    return linear_cross_entropy


def _resolve_cuda_backend(backend):
    if backend == "chunked":
        return None
    try:
        return _load_cce()
    except _CCENotInstalled as exc:
        if backend == "cce":
            raise RuntimeError(
                "loss_backend='cce' requires cut-cross-entropy on the CUDA server. "
                "Install the project's CUDA dependencies, or select loss_backend='chunked'."
            ) from exc
        warnings.warn(
            "cut-cross-entropy is not installed; loss_backend='auto' is using "
            "filtered checkpointed CE. Install cut-cross-entropy to enable fused CE.",
            RuntimeWarning, stacklevel=3,
        )
        return None


def _ce_sum(hidden, weight, targets):
    return F.cross_entropy(F.linear(hidden, weight).float(), targets, reduction="sum")


def _chunked_sum(hidden, weight, targets, chunk_size):
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_size):
        args = (hidden[start:start + chunk_size], weight, targets[start:start + chunk_size])
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            part = checkpoint(_ce_sum, *args, use_reentrant=False)
        else:
            part = _ce_sum(*args)
        total = total + part
    return total


def _cce_sum(hidden, weight, targets, linear_cross_entropy):
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        if dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError("CUDA CCE requires BF16/FP16 autocast")
        # Fused Triton ops do not apply F.linear's autocast policy themselves.
        # These differentiable casts keep master parameters and their grads FP32.
        hidden = hidden.to(dtype=dtype)
        weight = weight.to(dtype=dtype)
    elif hidden.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != hidden.dtype:
        raise RuntimeError(
            "CUDA CCE requires BF16/FP16 hidden activations and matching weights; "
            "keep parameters FP32 and use torch.autocast('cuda', dtype=torch.bfloat16)."
        )
    return linear_cross_entropy(
        hidden.contiguous(), weight.contiguous(), targets.contiguous(),
        reduction="sum", impl="cce", filter_eps=None,
        accum_e_fp32=True, accum_c_fp32=True,
    ).float()


def cross_entropy_stats(hidden, weight, targets, chunk_size, backend="chunked"):
    """Return supervised-token CE ``(mean, sum, count)`` without prompt logits.

    hidden is [..., D], weight is [vocab, D], and targets is [...] with -100 for
    ignored positions. CPU always uses the chunked correctness path. On CUDA,
    ``auto`` falls back only when the CCE package is absent; other errors remain
    visible. ``cce`` explicitly requires the package. ``count`` is an int64
    scalar tensor on the same device, allowing correct token-weighted metrics.
    """
    if backend not in ("chunked", "cce", "auto"):
        raise ValueError(f"Unknown loss backend: {backend}")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden must be [...,D] and weight must be [vocab,D]")
    if weight.shape[0] < 1 or weight.shape[1] < 1 or targets.shape != hidden.shape[:-1]:
        raise ValueError("targets must match hidden token dimensions and weight must be nonempty")
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must contain integer token IDs or -100")
    if weight.device != hidden.device or targets.device != hidden.device:
        raise ValueError("hidden, weight, and targets must be on the same device")
    flat_targets = targets.reshape(-1)
    supervised = flat_targets.ne(-100)
    count = supervised.sum()
    y = flat_targets[supervised].long()
    if y.numel() == 0:
        # Keep both computation paths attached so .backward() returns zero grads
        # even for entirely ignored SFT batches. Do not compute any LM logits.
        total = hidden.sum(dtype=torch.float32) * 0.0 + weight.reshape(-1)[0].float() * 0.0
        return total, total, count
    h = hidden.reshape(-1, hidden.shape[-1])[supervised]
    fn = _resolve_cuda_backend(backend) if hidden.device.type == "cuda" else None
    total = _chunked_sum(h, weight, y, chunk_size) if fn is None else _cce_sum(h, weight, y, fn)
    return total / count, total, count


def chunked_cross_entropy(hidden, weight, targets, chunk_size):
    """Backward-compatible mean CE wrapper with filtered checkpointed logits."""
    return cross_entropy_stats(hidden, weight, targets, chunk_size, backend="chunked")[0]
