from __future__ import annotations

import torch
import triton
import triton.language as tl
from helion.runtime import default_launcher as _default_launcher

@triton.jit
def _helion__grouped_gemm_forward_kernel(m_offsets, x, w, y, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    # src[forward.py:18]: for expert_idx in hl.grid(E):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    # src[forward.py:19]: m_start = m_offsets[expert_idx]
    m_start = tl.load(m_offsets + offset_0 * 1, None, eviction_policy='evict_first')
    # src[forward.py:20]: m_end = m_offsets[expert_idx + 1]
    add = 1 + offset_0
    m_end = tl.load(m_offsets + add * 1, None)
    # src[forward.py:21]: m_size = m_end - m_start
    v_0 = m_end - m_start
    # src[forward.py:22]: if m_size > 0:
    v_1 = tl.full([], 0, tl.int32)
    v_2 = v_0 > v_1
    # src[forward.py:22]: if m_size > 0:
    # src[forward.py:23]:     for tile_m, tile_n in hl.tile([m_size, N]):
    # src[forward.py:24]:         acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)
    # src[forward.py:22-29]: ...
    if v_2:
        v_0_copy = v_0
        m_start_copy = m_start
        v_0_copy_0 = v_0_copy
        m_start_copy_0 = m_start_copy
        # src[forward.py:23]: for tile_m, tile_n in hl.tile([m_size, N]):
        # src[forward.py:24]:     acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)
        # src[forward.py:25]:     for tile_k in hl.tile(K):
        # src[forward.py:23-29]: ...
        for offset_2 in tl.range(0, 768, _BLOCK_SIZE_2, loop_unroll_factor=2, disallow_acc_multi_buffer=False, flatten=False):
            indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
            for offset_1 in tl.range(0, v_0_copy_0.to(tl.int32), _BLOCK_SIZE_1, loop_unroll_factor=1, num_stages=3):
                indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
                mask_1 = indices_1 < v_0_copy_0
                m_start_copy_0_copy = m_start_copy_0
                m_start_copy_0_copy_0 = m_start_copy_0_copy
                # src[forward.py:24]: acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)
                acc = tl.full([_BLOCK_SIZE_1, _BLOCK_SIZE_2], 0.0, tl.float32)
                # src[forward.py:25]: for tile_k in hl.tile(K):
                # src[forward.py:26]:     x_blk = x[m_start + tile_m.index, tile_k]
                # src[forward.py:27]:     w_blk = w[expert_idx, tile_n, tile_k]
                # src[forward.py:25-28]: ...
                for offset_3 in tl.range(0, 2048, _BLOCK_SIZE_3, loop_unroll_factor=1, num_stages=3, flatten=False):
                    indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
                    m_start_copy_0_copy_0_copy = m_start_copy_0_copy_0
                    acc_copy = acc
                    m_start_copy_0_copy_0_copy_0 = m_start_copy_0_copy_0_copy
                    acc_copy_0 = acc_copy
                    # src[forward.py:26]: x_blk = x[m_start + tile_m.index, tile_k]
                    v_3 = m_start_copy_0_copy_0_copy_0 + indices_1
                    x_blk = tl.load(x + (v_3[:, None] * 2048 + indices_3[None, :] * 1), mask_1[:, None], other=0)
                    # src[forward.py:27]: w_blk = w[expert_idx, tile_n, tile_k]
                    w_blk = tl.load(w + (offset_0 * 1572864 + indices_2[:, None] * 2048 + indices_3[None, :] * 1), None, eviction_policy='evict_first')
                    # src[forward.py:28]: acc = torch.addmm(acc, x_blk, w_blk.T)
                    permute = tl.permute(w_blk, [1, 0])
                    acc = tl.dot(tl.cast(x_blk, tl.bfloat16), tl.cast(permute, tl.bfloat16), acc=acc_copy_0, input_precision='tf32', out_dtype=tl.float32)
                # src[forward.py:29]: y[m_start + tile_m.index, tile_n] = acc.to(y.dtype)
                v_5 = tl.cast(acc, tl.bfloat16)
                v_6 = m_start_copy_0_copy_0 + indices_1
                tl.store(y + (v_6[:, None] * 768 + indices_2[None, :] * 1), v_5, mask_1[:, None])

def _grouped_gemm_forward_kernel(x: torch.Tensor, w: torch.Tensor, m_offsets: torch.Tensor, dtype: torch.dtype, *, _launcher=_default_launcher):
    # src[forward.py:14]: M, _ = x.shape
    M, _ = x.shape
    # src[forward.py:15]: E, N, K = w.shape
    E, N, K = w.shape
    # src[forward.py:17]: y = torch.empty((M, N), device=x.device, dtype=dtype)
    y = torch.empty((M, N), device=x.device, dtype=dtype)
    # src[forward.py:23]: for tile_m, tile_n in hl.tile([m_size, N]):
    # src[forward.py:24]:     acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)
    # src[forward.py:25]:     for tile_k in hl.tile(K):
    # src[forward.py:23-29]: ...
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_2 = 64
    # src[forward.py:25]: for tile_k in hl.tile(K):
    # src[forward.py:26]:     x_blk = x[m_start + tile_m.index, tile_k]
    # src[forward.py:27]:     w_blk = w[expert_idx, tile_n, tile_k]
    # src[forward.py:25-28]: ...
    _BLOCK_SIZE_3 = 16
    # src[forward.py:18]: for expert_idx in hl.grid(E):
    # src[forward.py:19]:     m_start = m_offsets[expert_idx]
    # src[forward.py:20]:     m_end = m_offsets[expert_idx + 1]
    # src[forward.py:18-29]: ...
    _launcher(_helion__grouped_gemm_forward_kernel, (128,), m_offsets, x, w, y, _BLOCK_SIZE_1, _BLOCK_SIZE_2, _BLOCK_SIZE_3, num_warps=4, num_stages=6)
    # src[forward.py:31]: return y
    return y