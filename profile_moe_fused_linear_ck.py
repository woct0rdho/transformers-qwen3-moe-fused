#!/usr/bin/env python3

from math import sqrt

import torch

from qwen3_moe_fused.grouped_gemm.ck.binding import get_ck_module, grouped_gemm_forward
from qwen3_moe_fused.kernels.indexing import get_expert_offsets


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
    m_offsets = get_expert_offsets(selected_experts, num_experts)

    module = get_ck_module("tile128x64")

    print("Warming up...")
    for _ in range(3):
        grouped_gemm_forward(input, weight, m_offsets, ext_module=module)

    print("Profiling...")
    torch.cuda.synchronize()
    grouped_gemm_forward(input, weight, m_offsets, ext_module=module)
    torch.cuda.synchronize()
    print("Done")


if __name__ == "__main__":
    main()
