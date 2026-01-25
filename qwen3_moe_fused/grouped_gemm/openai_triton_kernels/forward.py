from typing import Optional

import torch
from kernels import get_kernel


triton_kernels = get_kernel("kernels-community/triton_kernels")
matmul_ogs = triton_kernels.matmul_ogs.matmul_ogs
RoutingData = triton_kernels.routing.RoutingData
compute_expt_data_torch = triton_kernels.routing.compute_expt_data_torch


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

    # x: (M, K)
    # w: (E, N, K) -> We need (E, K, N) for matmul_ogs where K matches x.shape[-1]
    # Transpose w to (E, K, N)
    w_transposed = w.transpose(1, 2)

    # Ensure m_offsets is int32
    m_offsets = m_offsets.to(torch.int32)
    m_sizes = torch.empty_like(m_offsets)
    m_sizes[0] = m_offsets[0]
    m_sizes[1:] = m_offsets[1:] - m_offsets[:-1]

    n_gates = M

    # Prepare RoutingData
    # Since x is already sorted/grouped by expert, we just need to provide the histogram (m_sizes)
    # and computed offsets.
    expt_data = compute_expt_data_torch(m_sizes, E, n_gates)

    routing_data = RoutingData(gate_scal=None, expt_hist=m_sizes, n_expts_tot=E, n_expts_act=1, expt_data=expt_data)

    # matmul_ogs returns (1, M, N) for non-batched x, so we squeeze it.
    y = matmul_ogs(x=x, w=w_transposed, bias=None, routing_data=routing_data)

    if dtype is None:
        dtype = x.dtype

    if y.dtype != dtype:
        y = y.to(dtype)

    return y
