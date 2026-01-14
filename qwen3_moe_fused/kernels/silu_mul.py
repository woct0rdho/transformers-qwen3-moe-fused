# h = silu(e) * g

from itertools import product

import torch
import triton
import triton.language as tl


_autotune_configs = []
for m, w, s in product([64, 128, 256, 512, 1024], [4, 8], [2, 3]):
    _autotune_configs.append(triton.Config({"BLOCK_SIZE": m}, num_warps=w, num_stages=s))


@triton.autotune(
    configs=_autotune_configs,
    key=[],
    cache_results=True,
)
@triton.jit
def _silu_mul_forward_kernel(
    e_ptr,
    g_ptr,
    h_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr = 128,
) -> None:
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e = tl.load(e_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask)

    f = e * tl.sigmoid(e)  # f32
    f = f.to(e_ptr.dtype.element_ty)
    h = f * g

    tl.store(h_ptr + offsets, h, mask=mask)


@triton.autotune(
    configs=_autotune_configs,
    key=[],
    cache_results=True,
)
@triton.jit
def _silu_mul_backward_kernel(
    dh_ptr,
    e_ptr,
    g_ptr,
    de_ptr,
    dg_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr = 128,
) -> None:
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dh = tl.load(dh_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask)

    se = tl.sigmoid(e)  # f32
    ese = e * se  # f32
    f = ese.to(e_ptr.dtype.element_ty)
    df = dh * g
    dg = f * dh

    de = df.to(tl.float32) * se * (1.0 + e - ese)  # f32
    de = de.to(de_ptr.dtype.element_ty)

    tl.store(de_ptr + offsets, de, mask=mask)
    tl.store(dg_ptr + offsets, dg, mask=mask)


@triton.autotune(
    configs=_autotune_configs,
    key=[],
    restore_value=["dh_ptr", "e_ptr", "g_ptr"],
    cache_results=True,
)
@triton.jit
def _silu_mul_backward_inplace_kernel(
    dh_ptr,
    e_ptr,
    g_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr = 128,
) -> None:
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dh = tl.load(dh_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask)

    se = tl.sigmoid(e)  # f32
    ese = e * se  # f32
    f = ese.to(e_ptr.dtype.element_ty)
    h = f * g
    df = dh * g
    dg = f * dh

    de = df.to(tl.float32) * se * (1.0 + e - ese)  # f32
    de = de.to(e_ptr.dtype.element_ty)

    tl.store(dh_ptr + offsets, h, mask=mask)
    tl.store(e_ptr + offsets, de, mask=mask)
    tl.store(g_ptr + offsets, dg, mask=mask)


def silu_mul_forward(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    assert e.is_cuda
    assert g.device == e.device
    assert e.is_contiguous()
    assert g.is_contiguous()
    assert g.numel() == e.numel()

    n_elements = e.numel()
    h = torch.empty_like(e)
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    with torch.cuda.device(e.device):
        _silu_mul_forward_kernel[grid](e, g, h, n_elements)
    return h


def silu_mul_backward(dh: torch.Tensor, e: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert e.is_cuda
    assert g.device == e.device
    assert dh.device == e.device
    assert e.is_contiguous()
    assert g.is_contiguous()
    assert dh.is_contiguous()
    assert g.numel() == e.numel()
    assert dh.numel() == e.numel()

    n_elements = e.numel()
    de = torch.empty_like(e)
    dg = torch.empty_like(g)
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    with torch.cuda.device(e.device):
        _silu_mul_backward_kernel[grid](dh, e, g, de, dg, n_elements)
    return de, dg


# dh, e, g are modified in place to h, de, dg
def silu_mul_backward_inplace(dh: torch.Tensor, e: torch.Tensor, g: torch.Tensor) -> None:
    assert e.is_cuda
    assert g.device == e.device
    assert dh.device == e.device
    assert e.is_contiguous()
    assert g.is_contiguous()
    assert dh.is_contiguous()
    assert g.numel() == e.numel()
    assert dh.numel() == e.numel()

    n_elements = e.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    with torch.cuda.device(e.device):
        _silu_mul_backward_inplace_kernel[grid](dh, e, g, n_elements)


class SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, g):
        ctx.save_for_backward(e, g)
        return silu_mul_forward(e, g)

    @staticmethod
    def backward(ctx, dh):
        e, g = ctx.saved_tensors
        return silu_mul_backward(dh, e, g)


silu_mul = SiluMul.apply
