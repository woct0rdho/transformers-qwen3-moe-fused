#!/usr/bin/env python3

import os
from math import sqrt

import torch

from qwen3_moe_fused.functional import moe_fused_linear, moe_fused_linear_naive
from qwen3_moe_fused.kernels.indexing import get_expert_counts
from test_utils import get_rtol_atol


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
    selected_experts = torch.randint(0, num_experts, (batch_size,), device=device, dtype=torch.int32)
    # Assume selected_experts is sorted
    selected_experts, _ = torch.sort(selected_experts)
    m_sizes = get_expert_counts(selected_experts, num_experts)
    grad_output = torch.randn(batch_size, out_features, device=device, dtype=dtype)

    input_auto = input.clone().requires_grad_()
    weight_auto = weight.clone().requires_grad_()
    output_auto = moe_fused_linear_naive(input_auto, weight_auto, m_sizes)
    output_auto.backward(gradient=grad_output)
    grad_input_auto = input_auto.grad
    grad_weight_auto = weight_auto.grad
    print("grad_input_auto", grad_input_auto.shape, grad_input_auto.dtype)
    print("grad_weight_auto", grad_weight_auto.shape, grad_weight_auto.dtype)

    input_grouped_gemm = input.clone().requires_grad_()
    weight_grouped_gemm = weight.clone().requires_grad_()
    output_grouped_gemm = moe_fused_linear(input_grouped_gemm, weight_grouped_gemm, m_sizes)
    output_grouped_gemm.backward(gradient=grad_output)
    grad_input_grouped_gemm = input_grouped_gemm.grad
    grad_weight_grouped_gemm = weight_grouped_gemm.grad
    print("grad_input_grouped_gemm", grad_input_grouped_gemm.shape, grad_input_grouped_gemm.dtype)
    print("grad_weight_grouped_gemm", grad_weight_grouped_gemm.shape, grad_weight_grouped_gemm.dtype)
    print(torch.allclose(grad_input_grouped_gemm, grad_input_auto, rtol=rtol, atol=atol))
    print(torch.allclose(grad_weight_grouped_gemm, grad_weight_auto, rtol=rtol, atol=atol * batch_size))
    print(get_rtol_atol(grad_input_grouped_gemm, grad_input_auto))
    print(get_rtol_atol(grad_weight_grouped_gemm, grad_weight_auto))


if __name__ == "__main__":
    main()
