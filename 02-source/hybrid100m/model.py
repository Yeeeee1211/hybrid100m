"""Causal hybrid decoder, isolated packing, serial MTP, and inference cache.

Training: keep parameters in FP32 and use BF16 autocast. No trainer is included.
Padding: training supports right padding only. Explicit segment_ids isolate packed
documents in attention, convolution, recurrent state and both prediction losses.
Cached inference uses equal-length, unpadded sequences; EOS alone does not reset state.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import HybridConfig
from .ops import gated_delta, kda
from .losses import cross_entropy_stats, chunked_cross_entropy


@dataclass(frozen=True)
class DeltaCache:
    conv: torch.Tensor
    state: torch.Tensor


@dataclass(frozen=True)
class KVCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class HybridCache:
    layers: tuple[DeltaCache | KVCache, ...]
    seen_tokens: int


@dataclass
class CausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor | None
    cache: HybridCache | None
    ntp_loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    loss_sum: torch.Tensor | None = None
    num_loss_tokens: torch.Tensor | None = None
    mtp_loss_sums: tuple[torch.Tensor, ...] = ()
    mtp_num_tokens: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class PackedLayout:
    """Compact B=1 layout. All lengths are positive and sum to the packed time."""
    lengths: tuple[int, ...]
    cu_seqlens: torch.Tensor
    document_ids: torch.Tensor

    @classmethod
    def from_lengths(cls, lengths, device):
        lengths = tuple(lengths)
        if not lengths or any(n <= 0 for n in lengths):
            raise ValueError("Packed sequences must have positive lengths")
        sizes = torch.tensor(lengths, device=device, dtype=torch.int32)
        cu = torch.cat((sizes.new_zeros(1), sizes.cumsum(0, dtype=torch.int32)))
        docs = torch.repeat_interleave(torch.arange(len(lengths), device=device), sizes.long())
        return cls(lengths, cu, docs)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = int(config.head_dim * config.rotary_fraction)
        self.theta = config.rope_theta
        self.rope_type = config.rope_type
        self.factor = config.rope_factor
        self.original_length = config.rope_original_max_position_embeddings
        self.beta_fast = config.rope_beta_fast
        self.beta_slow = config.rope_beta_slow

    def forward(self, positions):
        # Recompute FP32 frequencies: .to(bfloat16) must not permanently round a
        # stored inverse-frequency buffer, especially at large absolute positions.
        with torch.autocast(device_type=positions.device.type, enabled=False):
            exponent = torch.arange(0, self.dim, 2, device=positions.device, dtype=torch.float32) / self.dim
            inv = self.theta ** -exponent
            scale = 1.0
            if self.rope_type == "yarn":
                def correction(rotations):
                    return self.dim * math.log(self.original_length / (2 * math.pi * rotations)) / (2 * math.log(self.theta))
                low = max(math.floor(correction(self.beta_fast)), 0)
                high = min(math.ceil(correction(self.beta_slow)), self.dim - 1)
                ramp = ((torch.arange(self.dim // 2, device=positions.device, dtype=torch.float32) - low)
                        / max(high - low, 0.001)).clamp(0, 1)
                inv = inv * (1 - ramp) + (inv / self.factor) * ramp
                scale = 1 + 0.1 * math.log(self.factor)
            pos = positions[None] if positions.ndim == 1 else positions
            if pos.ndim != 2:
                raise ValueError("RoPE positions must be [time] or [batch,time]")
            phase = pos.float()[..., None] * inv
            phase = torch.cat((phase, phase), dim=-1)
            return (phase.cos() * scale)[:, None], (phase.sin() * scale)[:, None]


def apply_rope(x, cos, sin):
    dim = cos.shape[-1]
    a, b = x[..., :dim].float().chunk(2, dim=-1)
    rotated = torch.cat((-b, a), dim=-1)
    head = x[..., :dim].float() * cos + rotated * sin
    return torch.cat((head.to(x.dtype), x[..., dim:]), dim=-1)


class GatedDeltaNet(nn.Module):
    is_kda = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.heads = config.linear_num_heads
        self.dim = config.linear_head_dim
        self.width = self.heads * self.dim
        self.qkv_proj = nn.Linear(d, self.width * 3, bias=False)
        if self.is_kda:
            self.a_down = nn.Linear(d, config.linear_gate_rank, bias=False)
            self.a_up = nn.Linear(config.linear_gate_rank, self.width, bias=False)
            self.z_down = nn.Linear(d, config.linear_gate_rank, bias=False)
            self.z_up = nn.Linear(config.linear_gate_rank, self.width, bias=False)
        else:
            self.z_proj = nn.Linear(d, self.width, bias=False)
            self.a_proj = nn.Linear(d, self.heads, bias=False)
        self.b_proj = nn.Linear(d, self.heads, bias=False)
        self.conv = nn.Conv1d(
            self.width * 3, self.width * 3, config.conv_kernel_size,
            groups=self.width * 3, bias=False,
        )
        self.A_log = nn.Parameter(torch.empty(self.heads))
        self.dt_bias = nn.Parameter(torch.empty(self.width if self.is_kda else self.heads))
        self.out_norm = RMSNorm(self.dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.width, d, bias=False)

    def forward(self, x, cos, sin, past, use_cache, valid, layout=None):
        batch, length, _ = x.shape
        raw = self.qkv_proj(x).transpose(1, 2)
        if valid is not None:
            raw = raw * valid[:, None].to(raw.dtype)
        keep = self.config.conv_kernel_size - 1
        extended = None
        if layout is None:
            history = raw.new_zeros(batch, raw.shape[1], keep) if past is None else past.conv
            extended = torch.cat((history, raw), dim=-1)
            convolved = F.silu(F.conv1d(extended, self.conv.weight, groups=raw.shape[1]))
        else:
            # A four-tap depthwise convolution with no history across documents.
            # Avoid a global quadratic attention mask or one Python loop per token.
            convolved = torch.zeros_like(raw)
            weights = self.conv.weight[:, 0].to(raw.dtype)
            docs = layout.document_ids
            for lag in range(min(self.config.conv_kernel_size, length)):
                shifted = F.pad(raw[:, :, :length - lag], (lag, 0)) if lag else raw
                same_doc = torch.ones(length, device=x.device, dtype=torch.bool)
                if lag:
                    same_doc[:lag] = False
                    same_doc[lag:] = docs[lag:] == docs[:-lag]
                convolved = convolved + shifted * same_doc[None, None] * weights[:, -1 - lag][None, :, None]
            convolved = F.silu(convolved)
        q, k, v = (part.reshape(batch, length, self.heads, self.dim) for part in convolved.transpose(1, 2).chunk(3, dim=-1))
        # These exponentials and the recurrent state remain FP32 under autocast.
        if self.is_kda:
            a = self.a_up(self.a_down(x)).float().view(batch, length, self.heads, self.dim)
            g = -self.A_log.float().exp()[None, None, :, None] * F.softplus(a + self.dt_bias.float().view(self.heads, self.dim))
        else:
            a = self.a_proj(x).float()
            g = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())
        beta = self.b_proj(x).float().sigmoid().to(q.dtype)
        if valid is not None:
            g = g.masked_fill(~valid.view(batch, length, *([1] * (g.ndim - 2))), 0.0)
            beta = beta.masked_fill(~valid[..., None], 0.0)
        operator = kda if self.is_kda else gated_delta
        output, state = operator(
            q, k, v, g, beta,
            backend=self.config.linear_backend,
            initial_state=None if past is None else past.state,
            output_final_state=use_cache,
            cu_seqlens=None if layout is None else layout.cu_seqlens,
        )
        z = (self.z_up(self.z_down(x)) if self.is_kda else self.z_proj(x)).view(batch, length, self.heads, self.dim)
        gate = z.float().sigmoid() if self.is_kda else F.silu(z.float())
        output = (self.out_norm(output).float() * gate).to(x.dtype)
        output = self.out_proj(output.reshape(batch, length, self.width))
        new_cache = None
        if use_cache:
            conv_state = extended[:, :, -keep:].clone() if keep else extended[:, :, :0].clone()
            new_cache = DeltaCache(conv_state, state)
        return output, new_cache


class GatedKDA(GatedDeltaNet):
    """Per-key-channel forgetting with low-rank decay and sigmoid output gates."""
    is_kda = True


_NATIVE_GQA = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")


class GatedGQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = config.head_dim
        d = config.hidden_size
        self.q_gate_proj = nn.Linear(d, 2 * self.q_heads * self.dim, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.dim, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.dim, bias=False)
        self.q_norm = RMSNorm(self.dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.q_heads * self.dim, d, bias=False)

    def forward(self, x, cos, sin, past, use_cache, valid, layout=None):
        b, t, _ = x.shape
        # Gate and Q are interleaved PER HEAD in the projection, explicitly.
        q, gate = self.q_gate_proj(x).view(b, t, self.q_heads, 2 * self.dim).chunk(2, dim=-1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.kv_heads, self.dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        offset = 0 if past is None else past.key.shape[2]
        if past is not None:
            k, v = torch.cat((past.key, k), dim=2), torch.cat((past.value, v), dim=2)
        new_cache = KVCache(k, v) if use_cache else None
        groups = self.q_heads // self.kv_heads
        kwargs = {}
        if _NATIVE_GQA:
            kwargs["enable_gqa"] = groups > 1
        elif groups > 1:
            if x.is_cuda:
                raise RuntimeError("CUDA GQA requires a PyTorch build with SDPA enable_gqa support")
            # CPU correctness compatibility with older PyTorch; KV CACHE stays small.
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        mask = None
        causal = offset == 0
        if offset and t > 1:
            # SDPA is_causal uses top-left alignment on rectangular matrices.
            # Cached multi-token chunks need absolute (bottom-right) alignment.
            if x.is_cuda and _NATIVE_GQA:
                from torch.nn.attention.bias import causal_lower_right
                mask = causal_lower_right(t, k.shape[2])
            else:
                keys = torch.arange(k.shape[2], device=x.device)
                queries = offset + torch.arange(t, device=x.device)
                mask = keys[None, :] <= queries[:, None]
        # offset > 0, t == 1: all cached keys are visible; no causal mask needed.
        context = nullcontext()
        if self.config.sdpa_backend != "auto":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            which = SDPBackend.FLASH_ATTENTION if self.config.sdpa_backend == "flash" else SDPBackend.MATH
            context = sdpa_kernel(which)
        with context:
            if layout is None:
                output = F.scaled_dot_product_attention(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    attn_mask=mask, is_causal=causal, dropout_p=0.0, **kwargs,
                )
            else:
                # Separate native Flash SDPA calls retain linear memory and GQA;
                # each document sees only its own causal prefix.
                parts, start = [], 0
                for size in layout.lengths:
                    end = start + size
                    parts.append(F.scaled_dot_product_attention(
                        q[:, :, start:end].contiguous(), k[:, :, start:end].contiguous(),
                        v[:, :, start:end].contiguous(), is_causal=True, dropout_p=0.0, **kwargs,
                    ))
                    start = end
                output = torch.cat(parts, dim=2)
        output = output.transpose(1, 2) * gate.sigmoid()
        return self.out_proj(output.reshape(b, t, -1)), new_cache


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class DecoderBlock(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        linear = GatedKDA if config.linear_attention == "kda" else GatedDeltaNet
        self.mixer = GatedGQA(config) if kind == "full" else linear(config)
        self.ffn = SwiGLU(config)

    def forward(self, x, cos, sin, past=None, use_cache=False, valid=None, layout=None):
        mixed, cache = self.mixer(self.attn_norm(x), cos, sin, past, use_cache, valid, layout)
        x = x + mixed
        x = x + self.ffn(self.ffn_norm(x))
        if valid is not None:
            x = x * valid[..., None].to(x.dtype)
        return x, cache


class MTPBlock(nn.Module):
    """Serial next-next-token predictor; embedding and LM head live in the parent."""
    def __init__(self, config):
        super().__init__()
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.token_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fuse = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.block = DecoderBlock(config, "full")
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, next_embedding, cos, sin, layout):
        fused = self.fuse(torch.cat((self.hidden_norm(hidden), self.token_norm(next_embedding)), dim=-1))
        hidden, _ = self.block(fused, cos, sin, layout=layout)
        return self.final_norm(hidden)


class HybridForCausalLM(nn.Module):
    def __init__(self, config: HybridConfig | None = None):
        super().__init__()
        self.config = config or HybridConfig()
        c = self.config
        self.embedding = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(DecoderBlock(c, kind) for kind in c.layer_types)
        self.final_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.rope = RotaryEmbedding(c)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.mtp = nn.ModuleList(MTPBlock(c) for _ in range(c.mtp_num_layers))
        self.apply(self._init_module)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        # Scale residual branch outputs for a stable from-scratch starting point.
        with torch.no_grad():
            for layer in (*self.layers, *(module.block for module in self.mtp)):
                layer.mixer.out_proj.weight.div_(math.sqrt(2 * c.num_hidden_layers))
                layer.ffn.down.weight.div_(math.sqrt(2 * c.num_hidden_layers))

    def _init_module(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, GatedDeltaNet):
            with torch.no_grad():
                module.A_log.uniform_(0.0, math.log(16.0))
                dt = torch.empty_like(module.dt_bias).uniform_(math.log(0.001), math.log(0.1)).exp()
                module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def num_parameters(self, include_mtp=True):
        return sum(p.numel() for name, p in self.named_parameters() if include_mtp or not name.startswith("mtp."))

    def num_backbone_parameters(self):
        return self.num_parameters(include_mtp=False)

    def _mtp_losses(self, hidden, embeddings, labels, positions, lengths):
        """At depth k, fuse h_i^(k-1) with e_(i+k) and predict token i+k+1.

        Slice each document BEFORE compacting; no MTP input, attention or target
        ever crosses a boundary. All depths reuse the parent's embedding/head.
        """
        originals = list(embeddings.split(lengths, dim=1))
        label_docs = list(labels.split(lengths, dim=1))
        position_docs = list(positions.split(lengths, dim=1))
        previous = list(hidden.split(lengths, dim=1))
        means, sums, counts = [], [], []
        for depth, module in enumerate(self.mtp, start=1):
            active = [i for i, size in enumerate(lengths) if size > depth]
            if not active:
                zero = hidden.sum() * 0 + self.lm_head.weight.reshape(-1)[0] * 0
                for param in module.parameters():
                    zero = zero + param.reshape(-1)[0] * 0
                means.append(zero)
                sums.append(zero)
                counts.append(labels.new_zeros(()))
                continue
            sizes = tuple(lengths[i] - depth for i in active)
            layout = PackedLayout.from_lengths(sizes, hidden.device)
            old = torch.cat([previous[i][:, :-1] for i in active], dim=1)
            nxt = torch.cat([originals[i][:, depth:] for i in active], dim=1)
            pos = torch.cat([position_docs[i][:, depth:] for i in active], dim=1)
            cos, sin = self.rope(pos)
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                prediction = checkpoint(module, old, nxt, cos, sin, layout, use_reentrant=False)
            else:
                prediction = module(old, nxt, cos, sin, layout)
            chunks = prediction.split(sizes, dim=1)
            for index, chunk in zip(active, chunks):
                previous[index] = chunk
            h = torch.cat([chunk[:, :-1] for chunk in chunks], dim=1)
            y = torch.cat([label_docs[i][:, depth + 1:] for i in active], dim=1)
            mean, total, count = cross_entropy_stats(h, self.lm_head.weight, y,
                                                   self.config.loss_chunk_size, self.config.loss_backend)
            means.append(mean)
            sums.append(total)
            counts.append(count)
        return torch.stack(means).mean(), tuple(sums), tuple(counts)

    def forward(
        self, input_ids, *, labels=None, attention_mask=None, segment_ids=None, position_ids=None,
        cache: HybridCache | None = None, use_cache=False,
        return_logits: bool | None = None, logits_to_keep=0, compute_mtp: bool | None = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, time] tensor")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        b, t = input_ids.shape
        if cache is not None and not use_cache:
            raise ValueError("Supplying cache requires use_cache=True")
        if use_cache and (self.training or torch.is_grad_enabled() or labels is not None):
            raise ValueError("Cache is inference-only: use model.eval() and torch.no_grad()/inference_mode(), without labels")
        if use_cache and (segment_ids is not None or position_ids is not None):
            raise ValueError("Cached inference derives contiguous positions and does not accept packing")
        offset = 0 if cache is None else cache.seen_tokens
        if offset + t > self.config.max_position_embeddings:
            raise ValueError("Input + cached tokens exceeds max_position_embeddings (including generated tokens)")
        if cache is not None:
            if len(cache.layers) != len(self.layers) or offset <= 0:
                raise ValueError("Invalid hybrid cache")
            for kind, state in zip(self.config.layer_types, cache.layers):
                if kind == "full":
                    if not isinstance(state, KVCache) or state.key.shape[:3] != (b, self.config.num_key_value_heads, offset):
                        raise ValueError("KV cache shape/batch does not match this input")
                elif not isinstance(state, DeltaCache) or state.state.shape[0] != b:
                    raise ValueError("Delta cache shape/batch does not match this input")
        valid = None
        if segment_ids is not None and attention_mask is None:
            attention_mask = segment_ids.ge(0)
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask must be boolean or contain only 0/1")
            valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
                raise ValueError("Only nonempty right-padded sequences are supported; no left/interior padding")
            if use_cache and not bool(valid.all()):
                raise ValueError("Cached inference requires unpadded, equal-length sequences")
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer")
        if labels is not None and (labels.shape != input_ids.shape or labels.dtype != torch.long or labels.device != input_ids.device):
            raise ValueError("labels must be int64 [batch,time] aligned with input_ids on the same device; mask with -100")
        if labels is not None and bool(((labels != -100) & ((labels < 0) | (labels >= self.config.vocab_size))).any()):
            raise ValueError("labels must be -100 or an existing vocabulary token ID")
        if compute_mtp is None:
            compute_mtp = self.training and labels is not None and bool(self.mtp) and self.config.mtp_loss_weight > 0
        if compute_mtp and (labels is None or not self.mtp):
            raise ValueError("compute_mtp requires labels and configured MTP layers")
        positions = torch.arange(offset, offset + t, device=input_ids.device)[None].expand(b, -1)
        layout, restore_indices = None, None
        if segment_ids is not None:
            if (segment_ids.shape != input_ids.shape or segment_ids.dtype not in (torch.int32, torch.int64)
                    or segment_ids.device != input_ids.device):
                raise ValueError("segment_ids must be integer [batch,time] on the input device")
            if not bool((segment_ids.ge(0) == valid).all()) or bool((segment_ids[~valid] != -1).any()):
                raise ValueError("Valid tokens need nonnegative segment IDs; padding needs -1")
            # One host read validates boundaries once; kernels never synchronize
            # cu_seqlens values once per layer. IDs need not be globally unique.
            rows = segment_ids.detach().cpu().tolist()
            spans, lengths = [], []
            local_positions = torch.zeros_like(input_ids)
            for row, ids in enumerate(rows):
                seen, start = set(), 0
                while start < t and ids[start] >= 0:
                    doc = ids[start]
                    if doc in seen:
                        raise ValueError("Each segment ID must form one contiguous span per row")
                    seen.add(doc)
                    end = start + 1
                    while end < t and ids[end] == doc:
                        end += 1
                    spans.extend(range(row * t + start, row * t + end))
                    lengths.append(end - start)
                    local_positions[row, start:end] = torch.arange(end - start, device=input_ids.device)
                    start = end
            positions = local_positions
            restore_indices = torch.tensor(spans, device=input_ids.device, dtype=torch.long)
            layout = PackedLayout.from_lengths(lengths, input_ids.device)
        if position_ids is not None:
            if (position_ids.shape != input_ids.shape or position_ids.dtype not in (torch.int32, torch.int64)
                    or position_ids.device != input_ids.device):
                raise ValueError("position_ids must be integer [batch,time] on the input device")
            used = position_ids if valid is None else position_ids[valid]
            if bool(((used < 0) | (used >= self.config.max_position_embeddings)).any()):
                raise ValueError("position_ids exceed the supported context")
            if layout is not None and not bool((position_ids[valid] == positions[valid]).all()):
                raise ValueError("Packed position_ids must restart from zero in each document")
            positions = position_ids
        work_ids, work_labels = input_ids, labels
        if layout is not None:
            work_ids = input_ids.reshape(-1).index_select(0, restore_indices)[None]
            positions = positions.reshape(-1).index_select(0, restore_indices)[None]
            if labels is not None:
                work_labels = labels.reshape(-1).index_select(0, restore_indices)[None]
        work_valid = None if layout is not None else valid
        cos, sin = self.rope(positions)
        embeddings = self.embedding(work_ids)
        x = embeddings
        if work_valid is not None:
            x = x * work_valid[..., None].to(x.dtype)
        states = []
        for index, layer in enumerate(self.layers):
            past = None if cache is None else cache.layers[index]
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x, state = checkpoint(layer, x, cos, sin, None, False, work_valid, layout, use_reentrant=False)
            else:
                x, state = layer(x, cos, sin, past, use_cache, work_valid, layout)
            if use_cache:
                states.append(state)
        x = self.final_norm(x)
        loss = ntp_loss = mtp_loss = loss_sum = num_loss_tokens = None
        mtp_sums, mtp_counts = (), ()
        if labels is not None:
            targets = work_labels[:, 1:]
            if work_valid is not None:
                targets = targets.masked_fill(~(work_valid[:, 1:] & work_valid[:, :-1]), -100)
            if layout is not None:
                targets = targets.masked_fill((layout.document_ids[1:] != layout.document_ids[:-1])[None], -100)
            ntp_loss, loss_sum, num_loss_tokens = cross_entropy_stats(
                x[:, :-1], self.lm_head.weight, targets, self.config.loss_chunk_size, self.config.loss_backend)
            loss = ntp_loss
            if compute_mtp:
                if layout is not None:
                    h, emb, y, pos, lengths = x, embeddings, work_labels, positions, layout.lengths
                else:
                    lengths = tuple([t] * b) if valid is None else tuple(valid.sum(1).detach().cpu().tolist())
                    h = torch.cat([x[i:i + 1, :n] for i, n in enumerate(lengths)], dim=1)
                    emb = torch.cat([embeddings[i:i + 1, :n] for i, n in enumerate(lengths)], dim=1)
                    y = torch.cat([labels[i:i + 1, :n] for i, n in enumerate(lengths)], dim=1)
                    pos = torch.cat([positions[i:i + 1, :n] for i, n in enumerate(lengths)], dim=1)
                mtp_loss, mtp_sums, mtp_counts = self._mtp_losses(h, emb, y, pos, lengths)
                loss = loss + self.config.mtp_loss_weight * mtp_loss
        if return_logits is None:
            return_logits = labels is None
        logits = None
        if return_logits:
            if layout is not None:
                restored = x.new_zeros(b * t, x.shape[-1]).index_copy(0, restore_indices, x[0])
                x = restored.view(b, t, -1)
            logits = self.lm_head(x[:, -logits_to_keep:] if logits_to_keep else x)
        new_cache = HybridCache(tuple(states), offset + t) if use_cache else None
        return CausalLMOutput(loss=loss, logits=logits, cache=new_cache, ntp_loss=ntp_loss,
                              mtp_loss=mtp_loss, loss_sum=loss_sum, num_loss_tokens=num_loss_tokens,
                              mtp_loss_sums=mtp_sums, mtp_num_tokens=mtp_counts)
