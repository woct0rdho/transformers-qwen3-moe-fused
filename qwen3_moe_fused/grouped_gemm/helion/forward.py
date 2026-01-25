# y[m, n] = sum_k w[s[m], n, k] * x[m, k]
#
# On Windows, we need to set the environment variable HELION_AUTOTUNE_PRECOMPILE=spawn

from typing import Optional

import helion
import helion.language as hl
import torch


if torch.version.hip:
    configs = [
        # Strix Halo, large M
        helion.Config(
            block_sizes=[256, 64, 128],
            indexing=["tensor_descriptor", "pointer", "tensor_descriptor", "tensor_descriptor", "pointer"],
            load_eviction_policies=["", "last", "last", "first"],
            loop_orders=[[0, 1]],
            num_stages=2,
            num_warps=32,
            pid_type="persistent_blocked",
            range_flattens=[True, None, None, None],
            range_multi_buffers=[False, False, False, False],
            range_num_stages=[3, 1, 1, 1],
            range_unroll_factors=[1, 3, 4, 4],
            range_warp_specializes=[True, None, None, None],
        ),
    ]
else:
    configs = [
        # RTX 3090, small M
        helion.Config(
            block_sizes=[32, 64, 16],
            indexing=["pointer", "pointer", "pointer", "pointer", "pointer"],
            load_eviction_policies=["first", "", "", "first"],
            loop_orders=[[1, 0]],
            num_stages=6,
            num_warps=4,
            pid_type="flat",
            range_flattens=[None, None, False, False],
            range_multi_buffers=[None, None, True, None],
            range_num_stages=[0, 3, 0, 3],
            range_unroll_factors=[0, 1, 2, 1],
            range_warp_specializes=[],
        ),
        # RTX 3090, large M
        helion.Config(
            block_sizes=[128, 256, 64],
            indexing=["pointer", "pointer", "block_ptr", "block_ptr", "pointer"],
            load_eviction_policies=["last", "last", "last", "first"],
            loop_orders=[[1, 0]],
            num_stages=8,
            num_warps=8,
            pid_type="flat",
            range_flattens=[None, False, False, None],
            range_multi_buffers=[None, False, False, True],
            range_num_stages=[0, 4, 1, 3],
            range_unroll_factors=[0, 1, 4, 0],
            range_warp_specializes=[],
        ),
    ]


@helion.kernel(configs=configs, static_shapes=True)
def _grouped_gemm_forward_kernel(
    x: torch.Tensor, w: torch.Tensor, m_offsets: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    M, _ = x.shape
    E, N, K = w.shape

    y = torch.empty((M, N), device=x.device, dtype=dtype)
    for expert_idx in hl.grid(E):
        m_end = m_offsets[expert_idx]
        if expert_idx == 0:
            m_start = 0
        else:
            m_start = m_offsets[expert_idx - 1]
        m_size = m_end - m_start
        if m_size > 0:
            for tile_m, tile_n in hl.tile([m_size, N]):
                acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)
                for tile_k in hl.tile(K):
                    x_blk = x[m_start + tile_m.index, tile_k]
                    w_blk = w[expert_idx, tile_n, tile_k]
                    acc = torch.addmm(acc, x_blk, w_blk.T)
                y[m_start + tile_m.index, tile_n] = acc.to(y.dtype)

    return y


def grouped_gemm_forward(
    x: torch.Tensor, w: torch.Tensor, m_offsets: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    assert x.ndim == 2
    assert w.ndim == 3
    assert m_offsets.ndim == 1
    M, _ = x.shape
    E, N, K = w.shape
    assert x.shape[1] == K
    assert m_offsets.numel() == E

    if dtype is None:
        dtype = x.dtype

    m_offsets = m_offsets.to(torch.int32)

    return _grouped_gemm_forward_kernel(x, w, m_offsets, dtype)
