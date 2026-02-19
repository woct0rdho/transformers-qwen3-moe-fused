#!/usr/bin/env python3

import gc
import os

import torch
import triton

from qwen3_moe_fused.kernels.softmax_topk import softmax_topk, softmax_topk_naive


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


softmax_topk_compiled = torch.compile(softmax_topk_naive, fullgraph=True, mode="max-autotune")


providers = {
    "naive": softmax_topk_naive,
    "compile": softmax_topk_compiled,
    "triton": softmax_topk,
}
provider_names = list(providers)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["M"],
            x_vals=range(1024, 16384 + 1, 1024),
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="GB/s",
            plot_name="softmax_topk",
            args={},
        )
    ]
)
def benchmark(M, provider):
    print("M", M, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    N = 128
    k = 8
    norm = True
    device = "cuda"
    dtype = torch.bfloat16

    logits = torch.randn(M, N, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: providers[provider](logits, k, norm), warmup=100, rep=1000, quantiles=quantiles
    )

    input_bytes = M * N * logits.element_size()
    output_bytes = (M * k * logits.element_size()) + (M * k * 4)  # indices is int32 (4 bytes)
    gbps = lambda ms: (input_bytes + output_bytes) / ms * 1e-6
    print("M", M, "provider", provider, "end", gbps(ms))
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path=".")
