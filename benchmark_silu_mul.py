#!/usr/bin/env python3

import gc
import os

import torch
import triton
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

from qwen3_moe_fused.kernels.silu_mul import silu_mul as silu_mul_triton


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def silu_mul(x, y):
    return torch.nn.functional.silu(x) * y


silu_mul_compiled = torch.compile(silu_mul, fullgraph=True, mode="max-autotune")


def silu_mul_liger(x, y):
    return LigerSiLUMulFunction.apply(x, y)


providers = {
    "torch": silu_mul,
    "compile": silu_mul_compiled,
    "liger": silu_mul_liger,
    "triton": silu_mul_triton,
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
            ylabel="GB/s",
            plot_name="silu_mul",
            args={},
        )
    ]
)
def benchmark(N, provider):
    print("N", N, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn(N, N, device=device, dtype=dtype)
    y = torch.randn(N, N, device=device, dtype=dtype)
    numel = x.numel()
    elsize = x.element_size()

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: providers[provider](x, y), quantiles=quantiles)

    gbps = lambda ms: 2 * numel * elsize / ms * 1e-6
    print("N", N, "provider", provider, "end", gbps(ms))
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
