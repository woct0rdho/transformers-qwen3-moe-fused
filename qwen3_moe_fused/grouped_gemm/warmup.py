"""Pre-run grouped-GEMM autotuning at model-load time.

Triton benches every autotune config the first time it sees a (N, K, NUM_EXPERTS)
key. If that first time happens inside a training step, VRAM is already full of
activations and each bench launch can run 10-100x slower than normal (measured on
an RTX 5090 at 2.6 GiB free: 0.97 s average, 53.6 s worst single config), so tuning
all keys silently takes tens of minutes at near-idle power and is indistinguishable
from a deadlock. Killing the run discards the results (the disk cache is written
only when a key's benching completes), so every retry pays the full cost again.

Calling warmup_autotune() once after the model's weights are on the GPU - before
any forward pass - runs the same benching while memory is free (~0.2 s per config)
and persists the results via Triton's autotune disk cache (cache_results=True).
Subsequent runs, including future processes, skip benching entirely.

Set AUTOTUNE_DISABLE=1 instead to skip benching altogether at some throughput cost.
"""

import logging

import torch

from .backward_dw import grouped_gemm_backward_dw
from .forward import grouped_gemm_forward

logger = logging.getLogger(__name__)


def warmup_autotune(
    hidden_size: int,
    moe_intermediate_size: int,
    num_experts: int,
    dtype: torch.dtype = torch.bfloat16,
    m_tokens: int = 65536,
    backward: bool = True,
    fused_gate_up: bool = False,
    device: str = "cuda",
) -> None:
    """Autotune all grouped-GEMM keys a model with these dimensions will hit.

    m_tokens should approximate tokens-per-microbatch x top_k during real use; the
    autotune key ignores M but the benched timings (and thus the chosen config)
    depend on it. Set fused_gate_up=True for module variants that fuse gate and up
    into one projection of width 2 * moe_intermediate_size (e.g. Qwen3.5 MoE).
    """
    counts = torch.full((num_experts,), m_tokens // num_experts, dtype=torch.int64)
    counts[: m_tokens % num_experts] += 1
    m_offsets = torch.cumsum(counts, 0).to(device, torch.int32)

    up_width = 2 * moe_intermediate_size if fused_gate_up else moe_intermediate_size
    for K, N in ((hidden_size, up_width), (moe_intermediate_size, hidden_size)):
        logger.info("autotuning grouped-GEMM key N=%d K=%d E=%d (first run may take minutes)", N, K, num_experts)
        x = torch.randn(m_tokens, K, device=device, dtype=dtype)
        w = torch.randn(num_experts, N, K, device=device, dtype=dtype)
        grouped_gemm_forward(x, w, m_offsets)
        if backward:
            dy = torch.randn(m_tokens, N, device=device, dtype=dtype)
            grouped_gemm_forward(dy, w, m_offsets, dtype, transpose_w=True)
            grouped_gemm_backward_dw(x, dy, m_offsets, dtype)
            del dy
        del x, w
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def warmup_autotune_from_config(config, **kwargs) -> None:
    """Convenience wrapper reading dimensions from a HF model config."""
    warmup_autotune(
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_experts=config.num_experts,
        **kwargs,
    )
