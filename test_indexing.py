#!/usr/bin/env python3

import torch

from qwen3_moe_fused.kernels.indexing import (
    get_expert_offsets_and_idx_blocks,
    get_expert_offsets_and_idx_naive,
    get_expert_offsets_and_idx_parallel,
)


def main():
    N = 1024
    E = 128
    device = "cuda"

    s = torch.randint(0, E, (N,), device=device, dtype=torch.int32)

    offsets_naive, idx_naive, inv_idx_naive = get_expert_offsets_and_idx_naive(s, E)

    offsets_parallel, idx_parallel, inv_idx_parallel = get_expert_offsets_and_idx_parallel(s, E)
    torch.testing.assert_close(offsets_parallel, offsets_naive)
    torch.testing.assert_close(idx_parallel, idx_naive)
    torch.testing.assert_close(inv_idx_parallel, inv_idx_naive)

    offsets_blocks, idx_blocks, inv_idx_blocks = get_expert_offsets_and_idx_blocks(s, E)
    torch.testing.assert_close(offsets_blocks, offsets_naive)
    torch.testing.assert_close(idx_blocks, idx_naive)
    torch.testing.assert_close(inv_idx_blocks, inv_idx_naive)


if __name__ == "__main__":
    main()
