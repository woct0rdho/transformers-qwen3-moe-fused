# Uses https://github.com/fanshiqing/grouped_gemm

from typing import Optional

import grouped_gemm.ops
import torch


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

    m_sizes = torch.empty_like(m_offsets)
    m_sizes[0] = m_offsets[0]
    m_sizes[1:] = m_offsets[1:] - m_offsets[:-1]

    y = grouped_gemm.ops.gmm(x, w, m_sizes, trans_b=True)

    if dtype is None:
        dtype = x.dtype

    if y.dtype != dtype:
        y = y.to(dtype)

    return y
