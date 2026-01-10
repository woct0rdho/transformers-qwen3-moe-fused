#!/usr/bin/env python3

import os
from math import sqrt

import torch
from bitsandbytes.functional import dequantize_nf4, quantize_nf4

from qwen3_moe_fused.functional import _moe_fused_linear_naive_fwd, moe_fused_linear
from qwen3_moe_fused.grouped_gemm.quantized.forward import grouped_gemm_forward_4bit
from qwen3_moe_fused.grouped_gemm.triton_kernels.forward import grouped_gemm_forward
from qwen3_moe_fused.kernels.indexing import get_expert_counts
from test_model import get_rtol_atol


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    batch_size = 1024
    in_features = 2048
    out_features = 768
    num_experts = 128
    device = "cuda"

    # For higher precision, set os.environ["TRITON_F32_DEFAULT"] = "ieee"
    # dtype = torch.float32
    # rtol = 1e-6
    # atol = 1e-6

    # dtype = torch.float32
    # rtol = 1e-4
    # atol = 1e-4

    dtype = torch.bfloat16
    rtol = 1e-2
    atol = 1e-2

    input = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    weight_quant, weight_quant_state = quantize_nf4(weight, blocksize=256, compress_statistics=True)
    weight = dequantize_nf4(weight_quant, weight_quant_state)
    selected_experts = torch.randint(0, num_experts, (batch_size,), device=device, dtype=torch.int32)
    # Assume selected_experts is sorted
    selected_experts, _ = torch.sort(selected_experts)
    m_sizes = get_expert_counts(selected_experts, num_experts)

    output_naive = _moe_fused_linear_naive_fwd(input, weight, selected_experts)
    print("output_naive", output_naive.shape, output_naive.dtype)

    output_grouped_gemm = moe_fused_linear(input, weight, m_sizes)
    print("output_grouped_gemm", output_grouped_gemm.shape, output_grouped_gemm.dtype)
    print(torch.allclose(output_grouped_gemm, output_naive, rtol=rtol, atol=atol))
    print(get_rtol_atol(output_grouped_gemm, output_naive))

    output_triton_kernels = grouped_gemm_forward(input, weight, m_sizes)
    print("output_triton_kernels", output_triton_kernels.shape, output_triton_kernels.dtype)
    print(torch.allclose(output_triton_kernels, output_naive, rtol=rtol, atol=atol))
    print(get_rtol_atol(output_triton_kernels, output_naive))

    # output_grouped_gemm_4bit = grouped_gemm_forward_4bit(input, weight_quant, weight_quant_state, m_sizes)
    # print("output_grouped_gemm_4bit", output_grouped_gemm_4bit.shape, output_grouped_gemm_4bit.dtype)
    # print(torch.allclose(output_grouped_gemm_4bit, output_naive, rtol=rtol, atol=atol))
    # print(get_rtol_atol(output_grouped_gemm_4bit, output_naive))


if __name__ == "__main__":
    main()
