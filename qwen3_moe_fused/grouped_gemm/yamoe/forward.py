# You may also load yamoe from HuggingFace Kernels, but currently it's not built for all environments

from typing import Optional

import torch
import yamoe


def grouped_gemm_forward(
    x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    assert x.ndim == 2
    assert w.ndim == 3
    assert m_sizes.ndim == 1
    M, K = x.shape
    E, N, _ = w.shape
    assert w.shape[2] == K
    assert m_sizes.numel() == E

    if dtype is None:
        dtype = x.dtype

    bins = torch.cumsum(m_sizes, dim=0, dtype=torch.int32)
    indices = torch.arange(M, device=x.device, dtype=torch.int32)
    C = m_sizes.max().item()
    x_padded = torch.empty((E, C, K), device=x.device, dtype=x.dtype)
    yamoe.gather(x, indices, bins, x_padded, E, C, 1)

    w_t = w.transpose(1, 2).contiguous()
    out_padded = torch.empty((E, C, N), device=x.device, dtype=dtype)
    yamoe.batch_mm(x_padded, w_t, m_sizes, out_padded, True)

    y = torch.zeros((M, N), device=x.device, dtype=dtype)
    ones = torch.ones(M, device=x.device, dtype=dtype)
    yamoe.scatter(out_padded, indices, bins, ones, y, M, E, C, 1)

    return y
