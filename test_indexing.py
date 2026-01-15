#!/usr/bin/env python3

import torch

from qwen3_moe_fused.kernels.indexing import (
    get_expert_counts_and_idx_blocks,
    get_expert_counts_and_idx_naive,
    get_expert_counts_and_idx_parallel,
)


def main():
    N = 1024
    E = 128
    device = "cuda"

    s = torch.randint(0, E, (N,), device=device, dtype=torch.int32)

    counts_naive, inv_idx_naive, idx_naive = get_expert_counts_and_idx_naive(s, E)

    counts_parallel, inv_idx_parallel, idx_parallel = get_expert_counts_and_idx_parallel(s, E)
    torch.testing.assert_close(counts_parallel, counts_naive)
    torch.testing.assert_close(inv_idx_parallel, idx_naive)
    torch.testing.assert_close(idx_parallel, inv_idx_naive)

    counts_blocks, inv_idx_blocks, idx_blocks = get_expert_counts_and_idx_blocks(s, E)
    torch.testing.assert_close(counts_blocks, counts_naive)
    torch.testing.assert_close(inv_idx_blocks, idx_naive)
    torch.testing.assert_close(idx_blocks, inv_idx_naive)


if __name__ == "__main__":
    main()
