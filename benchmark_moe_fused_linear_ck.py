#!/usr/bin/env python3

import gc
import os
from math import sqrt

import torch
import triton

from qwen3_moe_fused.grouped_gemm.ck.binding import CK_CONFIGS, get_ck_module, grouped_gemm_forward
from qwen3_moe_fused.kernels.indexing import get_expert_offsets


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

configs = list(CK_CONFIGS.keys())


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["M"],
            x_vals=range(1024, 16384 + 1, 1024),
            line_arg="provider",
            line_vals=configs,
            line_names=configs,
            ylabel="GFLOPS",
            plot_name="moe_fused_linear_ck",
            args={},
        )
    ]
)
def benchmark(M, provider):
    print("M", M, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    in_features = 2048
    out_features = 768
    num_experts = 128
    device = "cuda"
    dtype = torch.bfloat16

    input = torch.randn(M, in_features, device=device, dtype=dtype)
    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (M,), device=device, dtype=torch.int32)
    # Assume selected_experts is sorted
    selected_experts, _ = torch.sort(selected_experts)
    m_offsets = get_expert_offsets(selected_experts, num_experts)

    module = get_ck_module(provider)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: grouped_gemm_forward(input, weight, m_offsets, ext_module=module),
        warmup=100,
        rep=1000,
        quantiles=quantiles,
    )

    perf = lambda ms: 2 * M * out_features * in_features / ms * 1e-6
    print("M", M, "provider", provider, "end", perf(ms))
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
