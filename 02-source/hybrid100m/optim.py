"""Single-GPU Muon + AdamW, with ordinary PyTorch optimizer checkpoints.

Muon is used only for hidden-layer two-dimensional weights. Embeddings/output
heads, vectors and convolution weights retain AdamW. The update scale follows
Moonshot's RMS matching (0.2 * sqrt(max(rows, columns))), rather than the
original fan-out/fan-in adjustment. This is Muon with QK-Norm in the model;
it does not implement the MLA-specific QK-Clip procedure from Kimi K2.
"""

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer


def _positive_finite(value, name, *, allow_zero=False):
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


@torch.no_grad()
def newton_schulz_orthogonalize(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz approximation; zero matrices yield zero updates.

    The smaller dimension is placed on the left to reduce Gram-matrix work.
    Five iterations are an approximation to the polar factor, not an exact
    SVD. Normalization is FP32; CUDA iterations use BF16 and CPU uses FP32.
    """
    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError("Muon orthogonalization requires a floating-point 2D matrix")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if matrix.device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Muon CUDA orthogonalization requires native BF16 support")
    with torch.autocast(matrix.device.type, enabled=False):
        x = matrix.float()
        transpose = x.shape[0] > x.shape[1]
        if transpose:
            x = x.mT
        x = x / (torch.linalg.vector_norm(x) + 1e-7)
        if matrix.device.type == "cuda":
            x = x.to(torch.bfloat16)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
        return (x.mT if transpose else x).to(matrix.dtype)


class MuonAdamW(Optimizer):
    """Muon and bias-corrected AdamW groups in one serializable optimizer.

    No optimizer-level gradient accumulation or loss scaling is performed.
    Call after backward (and optional clip_grad_norm_). Parameters and state
    remain FP32 under BF16 autocast; converting model parameters to BF16 is
    deliberately rejected. ``check_nonfinite`` rejects an entire step before
    updating any parameter. ``lr`` in every group supports standard schedulers.
    """
    def __init__(
        self, params: Iterable, *, lr=3e-4, weight_decay=0.1,
        momentum=0.95, nesterov=True, ns_steps=5, betas=(0.9, 0.95),
        eps=1e-8, check_nonfinite=True,
    ):
        _positive_finite(lr, "lr", allow_zero=True)
        _positive_finite(weight_decay, "weight_decay", allow_zero=True)
        _positive_finite(eps, "eps")
        if not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if len(betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in betas):
            raise ValueError("betas must be two finite numbers in [0, 1)")
        if type(ns_steps) is not int or ns_steps < 1:
            raise ValueError("ns_steps must be a positive integer")
        defaults = dict(
            lr=lr, initial_lr=lr, weight_decay=weight_decay, momentum=momentum,
            nesterov=nesterov, ns_steps=ns_steps, betas=betas, eps=eps,
            algorithm="adamw", check_nonfinite=check_nonfinite,
        )
        super().__init__(params, defaults)
        seen = set()
        for group in self.param_groups:
            if group["algorithm"] not in {"muon", "adamw"}:
                raise ValueError("algorithm must be muon or adamw")
            _positive_finite(group["lr"], "group lr", allow_zero=True)
            _positive_finite(group["weight_decay"], "group weight_decay", allow_zero=True)
            for param in group["params"]:
                if id(param) in seen:
                    raise ValueError("A parameter must occur in exactly one optimizer group")
                seen.add(id(param))
                if param.dtype != torch.float32:
                    raise ValueError("Keep model parameters in FP32; use BF16 autocast for forward")
                if group["algorithm"] == "muon" and param.ndim != 2:
                    raise ValueError("Muon groups may contain only 2D matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Check before mutation, aggregated per device to avoid one host sync
        # for every tensor. No quadratic attention scores are materialized.
        checks = {}
        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("MuonAdamW requires dense gradients")
                if group["check_nonfinite"]:
                    checks.setdefault(grad.device, []).append(torch.isfinite(grad).all())
        for device_checks in checks.values():
            if not bool(torch.stack(device_checks).all()):
                raise FloatingPointError("Non-finite gradient: no optimizer parameters were updated")
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.float()
                state = self.state[param]
                if group["algorithm"] == "muon":
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(param)
                        state["step"] = 0
                    state["step"] += 1
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(grad)
                    update = grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    update = newton_schulz_orthogonalize(update, group["ns_steps"])
                    scale = 0.2 * math.sqrt(max(param.shape))
                    param.mul_(1 - lr * wd).add_(update, alpha=-lr * scale)
                else:
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)
                    state["step"] += 1
                    b1, b2 = group["betas"]
                    state["exp_avg"].mul_(b1).add_(grad, alpha=1 - b1)
                    state["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    step = state["step"]
                    denominator = state["exp_avg_sq"].sqrt().div_(math.sqrt(1 - b2 ** step)).add_(group["eps"])
                    param.mul_(1 - lr * wd).addcdiv_(state["exp_avg"], denominator, value=-lr / (1 - b1 ** step))
        return loss


def build_optimizer(
    model: nn.Module, *, muon_lr=3e-4, adamw_lr=3e-4, weight_decay=0.1,
    momentum=0.95, nesterov=True, ns_steps=5, betas=(0.9, 0.95),
    eps=1e-8, check_nonfinite=True,
) -> MuonAdamW:
    """Deduplicate tied weights and create Muon/AdamW parameter groups.

    Embedding weights and the exposed ``lm_head`` always use AdamW. Every
    one-dimensional/vector parameter (including normalization, decay and time
    constants) has zero weight decay. Remaining hidden-layer 2D weights use
    Muon; higher-dimensional convolution kernels use decayed AdamW.
    """
    excluded = {id(module.weight) for module in model.modules() if isinstance(module, nn.Embedding)}
    head = getattr(model, "lm_head", None)
    if isinstance(head, nn.Module):
        excluded.update(id(param) for param in head.parameters())
    buckets = {("muon", True): [], ("adamw", True): [], ("adamw", False): []}
    names = {key: [] for key in buckets}
    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        # Channel-wise time/decay constants may be represented as 2D arrays
        # by KDA; their matrix shape does not make them hidden projections.
        special_constant = name.rsplit(".", 1)[-1] in {"A_log", "dt_bias"}
        decay = param.ndim > 1 and not special_constant
        algorithm = "muon" if param.ndim == 2 and id(param) not in excluded and not special_constant else "adamw"
        key = (algorithm, decay)
        buckets[key].append(param)
        names[key].append(name)
    groups = [dict(
        params=params, param_names=names[key], algorithm=key[0],
        lr=muon_lr if key[0] == "muon" else adamw_lr,
        initial_lr=muon_lr if key[0] == "muon" else adamw_lr,
        weight_decay=weight_decay if key[1] else 0.0,
    ) for key, params in buckets.items() if params]
    if not groups:
        raise ValueError("The model has no trainable parameters")
    return MuonAdamW(
        groups, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
        betas=betas, eps=eps, check_nonfinite=check_nonfinite,
    )
