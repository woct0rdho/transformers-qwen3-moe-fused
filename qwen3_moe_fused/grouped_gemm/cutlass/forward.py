# Uses https://github.com/fanshiqing/grouped_gemm

from typing import Optional

import grouped_gemm.ops
import torch


def grouped_gemm_forward(
    x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    assert x.ndim == 2
    assert w.ndim == 3
    assert m_sizes.ndim == 1
    M, _ = x.shape
    E, N, K = w.shape
    assert x.shape[1] == K
    assert m_sizes.numel() == E

    y = grouped_gemm.ops.gmm(x, w, m_sizes, trans_b=True)

    if dtype is None:
        dtype = x.dtype

    if y.dtype != dtype:
        y = y.to(dtype)

    return y
