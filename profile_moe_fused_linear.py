#!/usr/bin/env python3

from math import sqrt

import torch

from qwen3_moe_fused.grouped_gemm.forward import grouped_gemm_forward
from qwen3_moe_fused.kernels.indexing import get_expert_counts


def main():
    # Disable compile workers becaues they prevent ncu from exiting normally
    torch._inductor.config.compile_threads = 1

    torch.manual_seed(0)
    N = 16384
    in_features = 2048
    out_features = 768
    num_experts = 128
    device = "cuda"
    dtype = torch.bfloat16

    print("Setting up input...")
    input = torch.randn(N, in_features, device=device, dtype=dtype)
    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (N,), device=device, dtype=torch.int32)
    selected_experts, _ = torch.sort(selected_experts)
    m_sizes = get_expert_counts(selected_experts, num_experts)

    print("Warming up...")
    for _ in range(3):
        grouped_gemm_forward(input, weight, m_sizes)

    print("Profiling...")
    torch.cuda.synchronize()
    grouped_gemm_forward(input, weight, m_sizes)
    torch.cuda.synchronize()
    print("Done")


if __name__ == "__main__":
    main()
