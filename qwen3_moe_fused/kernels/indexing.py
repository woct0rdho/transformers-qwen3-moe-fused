from functools import partial

import torch
import triton
import triton.language as tl


# Faster than torch.histc when each element of s is an int in [0, E)
def get_expert_offsets(s: torch.Tensor, E: int) -> torch.Tensor:
    s = s.to(torch.int32)
    arange = torch.arange(E, device=s.device, dtype=torch.int32)
    counts = (arange[:, None] == s[None, :]).sum(dim=1, dtype=torch.int32)
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    return offsets


def get_expert_offsets_and_idx_naive(s: torch.Tensor, E: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = get_expert_offsets(s, E)

    idx = torch.argsort(s, stable=True).to(torch.int32)

    s_arange = torch.arange(s.numel(), device=s.device, dtype=torch.int32)
    inv_idx = torch.empty_like(idx)
    inv_idx[idx] = s_arange

    return offsets, idx, inv_idx


@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def get_expert_offsets_and_idx_parallel(s: torch.Tensor, E: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = s.to(torch.int32)
    E_arange = torch.arange(E, device=s.device, dtype=torch.int32)
    compare = E_arange[:, None] == s[None, :]
    ranks_in_bin = compare.cumsum(dim=1, dtype=torch.int32)
    counts = ranks_in_bin[:, -1]
    offsets = counts.cumsum(dim=0, dtype=torch.int32)

    s_arange = torch.arange(s.numel(), device=s.device, dtype=torch.int32)
    ranks_in_bin = ranks_in_bin[s, s_arange]

    start_offsets = offsets - counts
    idx = ranks_in_bin + start_offsets[s] - 1  # int32

    inv_idx = torch.empty_like(idx)  # int32
    inv_idx[idx] = s_arange

    # The above definition of idx is the opposite of torch.sort
    return offsets, inv_idx, idx


@triton.jit
def _count_kernel(
    s_ptr,
    counts_ptr,
    N,
    E,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # Load s, using E as sentinel for out-of-bounds
    s = tl.load(s_ptr + offs, mask=mask, other=E)
    row_start = pid * E
    tl.atomic_add(counts_ptr + row_start + s, 1, mask=s < E)


@triton.jit
def _index_kernel(
    s_ptr,
    idx_ptr,
    inv_idx_ptr,
    diff_offsets_ptr,
    N,
    E,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # Load s, using E as sentinel for out-of-bounds
    s = tl.load(s_ptr + offs, mask=mask, other=E)

    # Sort key: (expert_idx << 32) | local_idx
    local_idx = tl.arange(0, BLOCK_SIZE)
    packed = (s.to(tl.uint64) << 32) | local_idx.to(tl.uint64)

    sorted_packed = tl.sort(packed)

    sorted_s = (sorted_packed >> 32).to(tl.int32)
    sorted_local_idx = (sorted_packed & 0xFFFFFFFF).to(tl.int32)

    # Fetch precomputed offsets
    row_start = pid * E
    valid_mask = sorted_s < E
    diff = tl.load(diff_offsets_ptr + row_start + sorted_s, mask=valid_mask)
    dest = tl.arange(0, BLOCK_SIZE) + diff

    # Map back to global indices
    orig_idx = pid * BLOCK_SIZE + sorted_local_idx
    tl.store(inv_idx_ptr + dest, orig_idx, mask=valid_mask)
    tl.store(idx_ptr + orig_idx, dest, mask=valid_mask)


@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def get_expert_offsets_and_idx_blocks(s: torch.Tensor, E: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = s.numel()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)

    block_counts = torch.zeros((num_blocks, E), device=s.device, dtype=torch.int32)
    _count_kernel[(num_blocks,)](s, block_counts, N, E, BLOCK_SIZE)
    counts = block_counts.sum(dim=0, dtype=torch.int32)
    offsets = counts.cumsum(dim=0, dtype=torch.int32)

    diff_offsets = block_counts.cumsum(dim=0, dtype=torch.int32)
    diff_offsets -= block_counts

    diff_offsets += (offsets - counts).unsqueeze(0)

    local_offsets = block_counts.cumsum(dim=1, dtype=torch.int32)
    local_offsets -= block_counts
    diff_offsets -= local_offsets

    idx = torch.empty(N, device=s.device, dtype=torch.int32)
    inv_idx = torch.empty(N, device=s.device, dtype=torch.int32)

    _index_kernel[(num_blocks,)](s, idx, inv_idx, diff_offsets, N, E, BLOCK_SIZE)

    return offsets, inv_idx, idx


def get_expert_offsets_and_idx(s: torch.Tensor, E: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if s.numel() <= 16384:
        return get_expert_offsets_and_idx_parallel(s, E)
    else:
        return get_expert_offsets_and_idx_blocks(s, E)
