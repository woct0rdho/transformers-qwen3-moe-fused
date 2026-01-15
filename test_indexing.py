#!/usr/bin/env python3

import torch

from qwen3_moe_fused.kernels.indexing import get_expert_counts_and_idx, get_expert_counts_and_idx_naive


def main():
    N = 1024
    E = 128
    device = "cuda"

    s = torch.randint(0, E, (N,), device=device, dtype=torch.int32)

    counts_ref, inv_idx_ref, idx_ref = get_expert_counts_and_idx_naive(s, E)

    counts, inv_idx, idx = get_expert_counts_and_idx(s, E)

    torch.testing.assert_close(counts, counts_ref)
    torch.testing.assert_close(idx, idx_ref)
    torch.testing.assert_close(inv_idx, inv_idx_ref)


if __name__ == "__main__":
    main()
