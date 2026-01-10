# y[m, n] = sum_k w[s[m], n, k] * x[m, k]
# Currently this is slower than the unfused dequant + linear when M is large

from typing import Optional

import torch
import triton
import triton.language as tl
from bitsandbytes.functional import QuantState, dequantize_blockwise

from ..autotuning import (
    get_autotune_configs,
    get_autotune_keys,
    get_num_sms,
    prune_configs,
)


@triton.autotune(
    configs=get_autotune_configs(),
    prune_configs_by={"early_config_prune": prune_configs},
    key=get_autotune_keys(),
)
@triton.jit
def _grouped_gemm_forward_4bit_kernel(
    # Pointers
    x_ptr,
    w_quant_ptr,
    w_code_ptr,
    w_absmax_ptr,
    w_blocksize: tl.constexpr,
    m_sizes_ptr,
    y_ptr,
    # Dimensions
    M: int,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
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
    tidx = tl.program_id(0)
    m_end = 0
    processed_tiles = 0
    for expert_idx in range(NUM_EXPERTS):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size
        if m_size > 0:
            # tiles for this group's GEMM
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles_per_expert = num_m_tiles * num_n_tiles

            # Lower bound and upper bound are defined relative to the total tiles processed so far
            # This ensures that we are only processing tiles for the current expert group AND
            # we never exceed the total number of tiles for all expert groups
            while tidx >= processed_tiles and tidx < processed_tiles + num_tiles_per_expert:
                tile_idx = tidx - processed_tiles

                # Output tile for this thread block for this expert group
                # TODO: Check if L2 cache re-use for this order is optimal
                tile_m_idx = tile_idx % num_m_tiles
                tile_n_idx = tile_idx // num_m_tiles

                offs_k = tl.arange(0, BLOCK_SIZE_K)
                offs_k_d2 = tl.arange(0, BLOCK_SIZE_K // 2)

                offs_m = m_start + tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xk * offs_k[None, :]
                mask_m = offs_m < m_end

                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                w_offs = stride_we * expert_idx + stride_wn * offs_n[:, None] + stride_wk * offs_k[None, :]
                w_offs_d2 = (
                    stride_we // 2 * expert_idx + stride_wn // 2 * offs_n[:, None] + stride_wk * offs_k_d2[None, :]
                )
                mask_n = offs_n < N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                # GEMM main loop
                for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
                    mask_k = offs_k < K
                    mask_k_d2 = offs_k_d2 < K // 2
                    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])

                    # Dequantize
                    w_code_offs = tl.load(w_quant_ptr + w_offs_d2, mask=mask_n[:, None] & mask_k_d2[None, :])
                    w_code_offs = tl.interleave(w_code_offs // 16, w_code_offs % 16)
                    w_code = tl.load(w_code_ptr + w_code_offs, mask=mask_n[:, None] & mask_k[None, :])
                    w_absmax = tl.load(w_absmax_ptr + w_offs // w_blocksize)
                    # w_quant_state.dtype is ignored, and w is always cast to x.dtype
                    w = (w_code * w_absmax).to(x.dtype)

                    accumulator += tl.dot(x, w.T)

                    offs_k += BLOCK_SIZE_K
                    offs_k_d2 += BLOCK_SIZE_K // 2
                    x_ptrs += stride_xk * BLOCK_SIZE_K
                    w_offs += stride_wk * BLOCK_SIZE_K
                    w_offs_d2 += stride_wk * BLOCK_SIZE_K // 2
                y = accumulator.to(y_ptr.dtype.element_ty)

                y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
                tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])

                # Move to the next tile within this expert group
                tidx += NUM_SMS

            # Update the total tiles count for the next expert group
            processed_tiles += num_tiles_per_expert


def is_int_tensor(x: torch.Tensor) -> bool:
    return x.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


def grouped_gemm_forward_4bit(
    x: torch.Tensor,
    w_quant: torch.Tensor,
    w_quant_state: QuantState,
    m_sizes: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    assert w_quant_state.quant_type == "nf4"
    assert w_quant_state.blocksize == triton.next_power_of_2(w_quant_state.blocksize)

    # code and absmax should be float32. After computing code * absmax, w may be cast to bfloat16
    w_code = w_quant_state.code
    assert w_code.dtype == torch.float32

    w_absmax = w_quant_state.absmax
    if w_quant_state.nested:
        w_absmax = dequantize_blockwise(w_absmax, w_quant_state.state2)
        w_absmax += w_quant_state.offset
    assert w_absmax.dtype == torch.float32

    assert x.is_cuda
    assert w_quant.device == x.device
    assert w_code.device == x.device
    assert w_absmax.device == x.device
    assert m_sizes.device == x.device
    assert is_int_tensor(w_quant)
    assert is_int_tensor(m_sizes)
    assert x.is_contiguous()
    assert w_quant.is_contiguous()
    assert w_code.is_contiguous()
    assert w_absmax.is_contiguous()
    assert m_sizes.is_contiguous()
    assert x.ndim == 2
    assert len(w_quant_state.shape) == 3
    assert m_sizes.ndim == 1
    M, K = x.shape
    E, N, _ = w_quant_state.shape
    assert w_quant_state.shape[2] == K
    assert K % 2 == 0
    assert E * N * K % w_quant_state.blocksize == 0
    assert w_quant.numel() == E * N * K // 2
    assert w_absmax.numel() == E * N * K // w_quant_state.blocksize
    assert m_sizes.numel() == E

    if dtype is None:
        dtype = x.dtype
    y = torch.empty((M, N), device=x.device, dtype=dtype)
    NUM_SMS = get_num_sms()
    grid = lambda META: (NUM_SMS,)
    with torch.cuda.device(x.device):
        _grouped_gemm_forward_4bit_kernel[grid](
            # Pointers
            x,
            w_quant,
            w_code,
            w_absmax,
            w_quant_state.blocksize,
            m_sizes,
            y,
            # Dimensions
            M,
            N,
            K,
            E,
            NUM_SMS,
            # Strides
            x.stride(0),
            x.stride(1),
            N * K,
            K,
            1,
            y.stride(0),
            y.stride(1),
        )
    return y
