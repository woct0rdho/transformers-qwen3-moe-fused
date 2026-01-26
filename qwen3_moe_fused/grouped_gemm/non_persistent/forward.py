# y[m, n] = sum_k w[s[m], n, k] * x[m, k]

from functools import partial
from typing import Optional

import torch
import triton
import triton.language as tl

from ..autotuning import get_autotune_configs, get_autotune_keys, prune_configs
from ..forward import exceeds_smem_capacity, is_int_tensor


@triton.autotune(
    configs=get_autotune_configs(),
    key=get_autotune_keys(),
    prune_configs_by={"early_config_prune": partial(prune_configs, exceeds_smem_capacity)},
    cache_results=True,
)
@triton.jit
def _grouped_gemm_forward_kernel(
    # Pointers
    x_ptr,
    w_ptr,
    m_offsets_ptr,
    y_ptr,
    # Dimensions
    M: int,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    # Strides
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    # Metadata
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
) -> None:
    tile_idx = tl.program_id(0)
    expert_idx = tl.program_id(1)

    m_start = tl.load(m_offsets_ptr + expert_idx - 1, mask=expert_idx > 0, other=0).to(tl.int32)
    m_end = tl.load(m_offsets_ptr + expert_idx).to(tl.int32)
    m_size = m_end - m_start

    num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    if tile_idx >= num_m_tiles * num_n_tiles:
        return

    # Output tile for this thread block for this expert group
    tile_m_idx = tile_idx % num_m_tiles
    tile_n_idx = tile_idx // num_m_tiles

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_m = m_start + tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xk * offs_k[None, :]
    mask_m = offs_m < m_end

    offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[:, None] + stride_wk * offs_k[None, :]
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # GEMM main loop
    for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = offs_k < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :])

        accumulator += tl.dot(x, w.T)

        offs_k += BLOCK_SIZE_K
        x_ptrs += stride_xk * BLOCK_SIZE_K
        w_ptrs += stride_wk * BLOCK_SIZE_K

    y = accumulator.to(y_ptr.dtype.element_ty)
    y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
    tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


def grouped_gemm_forward(
    x: torch.Tensor, w: torch.Tensor, m_offsets: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    assert x.is_cuda
    assert w.device == x.device
    assert m_offsets.device == x.device
    assert is_int_tensor(m_offsets)
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_offsets.is_contiguous()
    assert x.ndim == 2
    assert w.ndim == 3
    assert m_offsets.ndim == 1
    M, _ = x.shape
    E, N, K = w.shape
    assert x.shape[1] == K
    assert m_offsets.numel() == E

    if dtype is None:
        dtype = x.dtype
    y = torch.empty((M, N), device=x.device, dtype=dtype)

    # Compute grid dimensions
    # We need the maximum m_size to size the grid efficiently.
    # Note: .item() causes a host-device sync
    # We can compute m_sizes from m_offsets for this
    m_sizes = torch.empty_like(m_offsets)
    m_sizes[0] = m_offsets[0]
    m_sizes[1:] = m_offsets[1:] - m_offsets[:-1]
    max_m = m_sizes.max().item()

    def grid(META):
        bm = META["BLOCK_SIZE_M"]
        bn = META["BLOCK_SIZE_N"]
        max_m_blocks = triton.cdiv(max_m, bm)
        num_n_tiles = triton.cdiv(N, bn)
        # Grid: (Max Tiles per Expert, Number of Experts)
        return (max_m_blocks * num_n_tiles, E)

    with torch.cuda.device(x.device):
        _grouped_gemm_forward_kernel[grid](
            # Pointers
            x,
            w,
            m_offsets,
            y,
            # Dimensions
            M,
            N,
            K,
            E,
            # Strides
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            y.stride(0),
            y.stride(1),
        )
    return y
