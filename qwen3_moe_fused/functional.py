import torch
from torch.nn import functional as F

from .grouped_gemm.interface import grouped_gemm


def moe_fused_linear_naive(input: torch.Tensor, weight: torch.Tensor, m_offsets: torch.Tensor) -> torch.Tensor:
    weight = weight.permute(0, 2, 1)
    return F.grouped_mm(input, weight, offs=m_offsets)


def moe_fused_linear(input: torch.Tensor, weight: torch.Tensor, m_offsets: torch.Tensor) -> torch.Tensor:
    """
    Computes a MoE linear operation using grouped GEMM.

    The operation is defined as:
    `output[b, o] = sum_i weight[selected_experts[b], o, i] * input[b, i]`

    Args:
        input (`torch.FloatTensor`): input tensor of shape `(batch_size, in_features)`.
        weight (`torch.FloatTensor`): weight tensor of shape `(num_experts, out_features, in_features)`.
        m_offsets (`torch.IntTensor`): offsets of experts in shape `(num_experts,)`. `m_offsets[i]` marks the end of
            expert `i`.

    Returns:
        output (`torch.FloatTensor`): output tensor of shape `(batch_size, out_features)`.
    """
    return grouped_gemm(input, weight, m_offsets)
