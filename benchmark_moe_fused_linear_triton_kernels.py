#!/usr/bin/env python3

import gc
import os
from math import sqrt


# Set this before importing qwen3_moe_fused.grouped_gemm
os.environ["AUTOTUNE_BATCH_SIZE"] = "1"

import torch
import triton

from qwen3_moe_fused.grouped_gemm.triton_kernels.forward import (
    RoutingData,
    compute_expt_data_torch,
    matmul_ogs,
)
from qwen3_moe_fused.kernels.indexing import get_expert_counts


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

providers = {
    "matmul_ogs": matmul_ogs,
}
provider_names = list(providers)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=range(1024, 16384 + 1, 1024),
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="GFLOPS",
            plot_name="moe_fused_linear_triton_kernels",
            args={},
        )
    ]
)
def benchmark(N, provider):
    print("N", N, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    in_features = 2048
    out_features = 768
    num_experts = 128
    device = "cuda"
    dtype = torch.bfloat16

    input = torch.randn(N, in_features, device=device, dtype=dtype)
    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (N,), device=device, dtype=torch.int32)
    # Assume selected_experts is sorted
    selected_experts, _ = torch.sort(selected_experts)
    m_sizes = get_expert_counts(selected_experts, num_experts)

    # Precompute everything else
    w_transposed = weight.transpose(1, 2)

    if m_sizes.dtype != torch.int32:
        m_sizes_int32 = m_sizes.to(torch.int32)
    else:
        m_sizes_int32 = m_sizes

    expt_data = compute_expt_data_torch(m_sizes_int32, num_experts, N)

    routing_data = RoutingData(
        gate_scal=None, expt_hist=m_sizes_int32, n_expts_tot=num_experts, n_expts_act=1, expt_data=expt_data
    )

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: providers[provider](input, w_transposed, None, routing_data), quantiles=quantiles
    )

    perf = lambda ms: 2 * N * out_features * in_features / ms * 1e-6
    print("N", N, "provider", provider, "end", perf(ms))
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
