from typing import Optional

import torch
from kernels import get_kernel

from ..forward import is_int_tensor


triton_kernels = get_kernel("kernels-community/triton_kernels")
matmul_ogs = triton_kernels.matmul_ogs.matmul_ogs
RoutingData = triton_kernels.routing.RoutingData
compute_expt_data_torch = triton_kernels.routing.compute_expt_data_torch


def grouped_gemm_forward(
    x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Grouped GEMM forward pass using triton_kernels.matmul_ogs.

    y[m, n] = sum_k w[s[m], n, k] * x[m, k]

    Args:
        x: Input tensor of shape (M, K)
        w: Weight tensor of shape (E, N, K)
        m_sizes: Tensor of shape (E,) containing number of rows per expert
        dtype: Optional output dtype

    Returns:
        y: Output tensor of shape (M, N)
    """
    assert x.is_cuda
    assert w.device == x.device
    assert m_sizes.device == x.device
    assert is_int_tensor(m_sizes)
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()
    assert x.ndim == 2
    assert w.ndim == 3
    assert m_sizes.ndim == 1

    M, K = x.shape
    E, N, _ = w.shape
    assert w.shape[2] == K
    assert m_sizes.numel() == E

    # x: (M, K)
    # w: (E, N, K) -> We need (E, K, N) for matmul_ogs where K matches x.shape[-1]
    # Transpose w to (E, K, N)
    w_transposed = w.transpose(1, 2)

    # Ensure m_sizes is int32 for compute_expt_data_torch
    m_sizes = m_sizes.to(torch.int32)

    n_gates = M

    # Prepare RoutingData
    # Since x is already sorted/grouped by expert, we just need to provide the histogram (m_sizes)
    # and computed offsets.
    expt_data = compute_expt_data_torch(m_sizes, E, n_gates)

    routing_data = RoutingData(
        gate_scal=None,
        expt_hist=m_sizes,
        n_expts_tot=E,
        n_expts_act=1,
        expt_data=expt_data
    )

    # matmul_ogs returns (1, M, N) for non-batched x, so we squeeze it.
    y = matmul_ogs(
        x=x,
        w=w_transposed,
        bias=None,
        routing_data=routing_data
    )

    if dtype is None:
        dtype = x.dtype

    if y.dtype != dtype:
        y = y.to(dtype)

    return y
