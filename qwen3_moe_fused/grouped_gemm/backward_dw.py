# w[e, n, k] = sum_m if(s[m] == e) y[m, n] * x[m, k]

from functools import partial

import torch
import triton
import triton.language as tl

from .autotuning import (
    GRID_FACTOR,
    get_autotune_configs,
    get_autotune_keys,
    get_num_sms,
    prune_configs,
)
from .forward import is_int_tensor


def exceeds_smem_capacity(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
    smem_size: int,
) -> bool:
    x_size = BLOCK_SIZE_M * BLOCK_SIZE_K * dtype.itemsize
    y_size = BLOCK_SIZE_M * BLOCK_SIZE_N * dtype.itemsize
    if num_stages <= 1:
        size = max(x_size, y_size)
    else:
        # (num_stages - 1) stages of the larger tile will be cached in smem
        size = min(x_size, y_size) + (num_stages - 1) * max(x_size, y_size)
    return size > smem_size


@triton.autotune(
    configs=get_autotune_configs(),
    prune_configs_by={"early_config_prune": partial(prune_configs, exceeds_smem_capacity)},
    key=get_autotune_keys(),
)
@triton.jit
def _grouped_gemm_backward_dw_kernel(
    # Pointers
    x_ptr,
    y_ptr,
    m_offsets_ptr,
    w_ptr,
    # Dimensions
    M: int,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # Strides
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    # Metadata
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
) -> None:
    tidx = tl.program_id(0)

    # Output tiles per expert, since each expert weight matrix is [N, K]
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles_per_expert = num_n_tiles * num_k_tiles
    total_tiles = num_tiles_per_expert * NUM_EXPERTS

    for work_idx in range(tidx, total_tiles, NUM_SMS):
        expert_idx = work_idx // num_tiles_per_expert
        tile_idx = work_idx % num_tiles_per_expert

        # Output tile index
        tile_n_idx = tile_idx % num_n_tiles
        tile_k_idx = tile_idx // num_n_tiles

        m_start = tl.load(m_offsets_ptr + expert_idx).to(tl.int32)
        m_end = tl.load(m_offsets_ptr + expert_idx + 1).to(tl.int32)
        m_size = m_end - m_start

        if m_size > 0:
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            mask_n = offs_n < N
            mask_k = offs_k < K

            offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)

            x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xk * offs_k[None, :]
            y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]

            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
            for _ in range(tl.cdiv(m_size, BLOCK_SIZE_M)):
                mask_m = offs_m < m_end
                x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
                y = tl.load(y_ptrs, mask=mask_m[:, None] & mask_n[None, :])

                accumulator += tl.dot(y.T, x)

                offs_m += BLOCK_SIZE_M
                x_ptrs += stride_xm * BLOCK_SIZE_M
                y_ptrs += stride_ym * BLOCK_SIZE_M
            w = accumulator.to(w_ptr.dtype.element_ty)

            w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[:, None] + stride_wk * offs_k[None, :]
            tl.store(w_ptrs, w, mask=mask_n[:, None] & mask_k[None, :])


def grouped_gemm_backward_dw(
    x: torch.Tensor, y: torch.Tensor, m_sizes: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    assert x.is_cuda
    assert y.device == x.device
    assert m_sizes.device == x.device
    assert is_int_tensor(m_sizes)
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert m_sizes.is_contiguous()
    assert x.ndim == 2
    assert y.ndim == 2
    assert m_sizes.ndim == 1
    M, K = x.shape
    _, N = y.shape
    assert y.shape[0] == M
    E = m_sizes.numel()

    w = torch.zeros((E, N, K), device=x.device, dtype=dtype)

    m_offsets = torch.zeros(E + 1, device=m_sizes.device, dtype=m_sizes.dtype)
    m_offsets[1:] = torch.cumsum(m_sizes, dim=0)

    NUM_SMS = get_num_sms()
    TOTAL_BLOCKS = NUM_SMS * GRID_FACTOR

    grid = lambda META: (TOTAL_BLOCKS,)
    with torch.cuda.device(x.device):
        _grouped_gemm_backward_dw_kernel[grid](
            # Pointers
            x,
            y,
            m_offsets,
            w,
            # Dimensions
            M,
            N,
            K,
            E,
            TOTAL_BLOCKS,
            # Strides
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            w.stride(0),
            w.stride(1),
            w.stride(2),
        )
    return w
