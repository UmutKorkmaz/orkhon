"""Architecture contract for the Orkhon decoder-only Transformer.

This dataclass is the single source of truth for model shape. Every module in
``orkhon.model`` reads from it; the trainer, exporter, and KV-cache all depend on
these exact field names. Keep it dependency-light (stdlib only) so the model
package never needs pydantic/yaml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    # Vocabulary / context
    vocab_size: int = 4096
    block_size: int = 256  # max sequence length the model is trained/served at

    # Transformer shape
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2  # GQA: must divide n_heads; == n_heads is plain MHA
    head_dim: int | None = None  # defaults to d_model // n_heads

    # FFN (SwiGLU). If None, derived from d_model (see intermediate()).
    intermediate_size: int | None = None
    ffn_multiple: int = 64  # round derived FFN size up to this multiple

    # Normalization / RoPE
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Regularization / init / bias
    dropout: float = 0.0
    init_std: float = 0.02
    use_bias: bool = False
    tie_word_embeddings: bool = True

    # Attention backend: "auto" uses scaled_dot_product_attention when available,
    # "manual" forces the reference implementation (used in parity tests).
    attn_impl: str = "auto"

    # Long-context RoPE scaling (L0, inference-time, no GPU). None = standard RoPE
    # (table sized to block_size). Set to "linear" or "yarn" + scaling_factor>1 to
    # extend context beyond block_size at inference.
    rope_scaling_type: str | None = None
    rope_scaling_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads "
                f"({self.n_kv_heads}) for grouped-query attention"
            )
        if self.attn_impl not in ("auto", "manual", "sdpa"):
            raise ValueError(f"attn_impl must be auto|manual|sdpa, got {self.attn_impl!r}")
        if self.rope_scaling_type is not None:
            if self.rope_scaling_type not in ("linear", "ntk"):
                raise ValueError(f"rope_scaling_type must be None|linear|ntk, got {self.rope_scaling_type!r}")
            if self.rope_scaling_factor < 1.0:
                raise ValueError(f"rope_scaling_factor must be >= 1.0, got {self.rope_scaling_factor}")

    # --- Derived shapes (functions, not stored, so frozen dataclass stays clean) ---

    def hd(self) -> int:
        """Per-head dimension."""
        return self.head_dim if self.head_dim is not None else self.d_model // self.n_heads

    def n_rep(self) -> int:
        """How many query heads share each KV head (GQA repeat factor)."""
        return self.n_heads // self.n_kv_heads

    def intermediate(self) -> int:
        """SwiGLU hidden size. Default ~ (8/3)*d_model rounded to ffn_multiple."""
        if self.intermediate_size is not None:
            return self.intermediate_size
        raw = int(8 * self.d_model / 3)
        m = self.ffn_multiple
        return ((raw + m - 1) // m) * m

    def estimate_params(self) -> int:
        """Approximate parameter count (embeddings + blocks + final norm)."""
        v, d, L = self.vocab_size, self.d_model, self.n_layers
        hd, inter = self.hd(), self.intermediate()
        q = self.n_heads * hd
        kv = self.n_kv_heads * hd
        attn = d * q + 2 * (d * kv) + q * d  # q,k,v,o projections
        ffn = 3 * d * inter  # gate, up, down
        norms = 2 * d  # two RMSNorms per block
        per_block = attn + ffn + norms
        embed = v * d
        total = embed + L * per_block + d  # + final norm
        if not self.tie_word_embeddings:
            total += v * d
        return total

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})
