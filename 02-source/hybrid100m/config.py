"""Architecture configuration; vocabulary size INCLUDES every special token."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class HybridConfig:
    vocab_size: int = 32768
    hidden_size: int = 512
    intermediate_size: int = 1920
    num_hidden_layers: int = 20
    full_attention_interval: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 64
    linear_num_heads: int = 4
    linear_head_dim: int = 128
    linear_attention: str = "kda"
    linear_gate_rank: int = 128
    conv_kernel_size: int = 4
    max_position_embeddings: int = 10240
    rope_theta: float = 10000.0
    rotary_fraction: float = 1.0
    rope_type: str = "yarn"
    rope_factor: float = 5.0
    rope_original_max_position_embeddings: int = 2048
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    linear_backend: str = "fla"
    # RTX 4090 / 24GB training: fail explicitly rather than selecting quadratic
    # math attention on a long sequence when the Flash kernel is unavailable.
    sdpa_backend: str = "flash"
    gradient_checkpointing: bool = True
    # Conservative 24GB baseline. Benchmark 256/512 on the target GPU before
    # increasing this; larger chunks trade temporary logit memory for fewer calls.
    loss_chunk_size: int = 128
    loss_backend: str = "cce"
    mtp_num_layers: int = 1
    mtp_loss_weight: float = 0.1

    def __post_init__(self):
        for name in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "full_attention_interval", "num_attention_heads", "num_key_value_heads",
            "head_dim", "linear_num_heads", "linear_head_dim", "conv_kernel_size",
            "max_position_embeddings", "loss_chunk_size", "linear_gate_rank",
            "rope_original_max_position_embeddings",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_hidden_layers % self.full_attention_interval:
            raise ValueError("num_hidden_layers must be divisible by full_attention_interval")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Q heads must be divisible by KV heads")
        if not 0 < self.rotary_fraction <= 1:
            raise ValueError("rotary_fraction must be in (0, 1]")
        dim = self.head_dim * self.rotary_fraction
        if int(dim) != dim or int(dim) < 2 or int(dim) % 2:
            raise ValueError("rotary dimension must be an even positive integer >= 2")
        if not self.rope_theta > 1 or not self.rms_norm_eps > 0:
            raise ValueError("rope_theta must be > 1 and rms_norm_eps must be > 0")
        for name in ("rope_theta", "rms_norm_eps", "initializer_range", "rope_factor",
                     "rope_beta_fast", "rope_beta_slow", "mtp_loss_weight"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        if self.linear_attention not in {"kda", "gated_delta"}:
            raise ValueError("linear_attention must be kda or gated_delta")
        if self.rope_type not in {"default", "yarn"}:
            raise ValueError("rope_type must be default or yarn")
        if self.rope_factor < 1 or not 0 < self.rope_beta_slow < self.rope_beta_fast:
            raise ValueError("YaRN requires factor >= 1 and 0 < beta_slow < beta_fast")
        if self.rope_type == "yarn" and self.max_position_embeddings > self.rope_original_max_position_embeddings * self.rope_factor:
            raise ValueError("YaRN factor does not cover max_position_embeddings")
        if type(self.mtp_num_layers) is not int or self.mtp_num_layers < 0:
            raise ValueError("mtp_num_layers must be a nonnegative integer")
        if self.mtp_loss_weight < 0:
            raise ValueError("mtp_loss_weight must be nonnegative")
        if self.loss_backend not in {"cce", "chunked", "auto"}:
            raise ValueError("loss_backend must be cce, chunked, or auto")
        if self.linear_backend not in {"fla", "reference"}:
            raise ValueError("linear_backend must be 'fla' or 'reference'")
        if self.sdpa_backend not in {"auto", "flash", "math"}:
            raise ValueError("sdpa_backend must be auto, flash, or math")

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(
            "full" if (i + 1) % self.full_attention_interval == 0 else "linear"
            for i in range(self.num_hidden_layers)
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HybridConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        # Existing checkpoints must retain their original architecture and RoPE.
        # A file without the newly introduced fields describes the old GDN model.
        values.setdefault("linear_attention", "gated_delta")
        values.setdefault("rope_type", "default")
        values.setdefault("mtp_num_layers", 0)
        values.setdefault("loss_backend", "chunked")
        return cls(**values)
