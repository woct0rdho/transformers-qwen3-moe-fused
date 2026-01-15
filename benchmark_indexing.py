#!/usr/bin/env python3

import gc
import os

import torch
import triton

from qwen3_moe_fused.kernels.indexing import get_expert_counts_and_idx, get_expert_counts_and_idx_naive


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


providers = {
    "naive": get_expert_counts_and_idx_naive,
    "triton": get_expert_counts_and_idx,
}
provider_names = list(providers)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(10, 21)],
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="GB/s",
            plot_name="indexing",
            args={},
        )
    ]
)
def benchmark(N, provider):
    print("N", N, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    E = 128
    device = "cuda"
    dtype = torch.int32

    s = torch.randint(0, E, (N,), device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: providers[provider](s, E), quantiles=quantiles)

    gbps = lambda ms: 3 * N * 4 / ms * 1e-6
    print("N", N, "E", E, "provider", provider, "end", gbps(ms))
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
