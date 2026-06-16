"""Orkhon — a complete, auditable, from-scratch LLM stack.

Pipeline: tokenizer -> data -> pretrain -> SFT -> DPO -> eval -> export -> serve.
The model (decoder-only Transformer with GQA, RoPE, RMSNorm, SwiGLU, KV-cache) is
implemented by hand for educational clarity. See docs/implementation-plan.md.
"""

__version__ = "0.1.0"
