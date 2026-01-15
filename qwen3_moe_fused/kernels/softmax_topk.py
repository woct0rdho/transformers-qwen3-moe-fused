from itertools import product

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


def softmax_topk_naive(logits: torch.Tensor, k: int, norm_topk_prob: bool) -> tuple[torch.Tensor, torch.Tensor]:
    weights = F.softmax(logits, dim=1)
    weights, indices = torch.topk(weights, k, dim=-1)
    if norm_topk_prob:
        weights /= weights.sum(dim=-1, keepdim=True)
    return weights, indices


if torch.version.hip:
    DEFAULT_NUM_STAGES = [1, 2, 3]
else:
    DEFAULT_NUM_STAGES = [3, 4, 5, 6]

_autotune_configs = []
for m, w, s in product([16, 32, 64], [4, 8], DEFAULT_NUM_STAGES):
    _autotune_configs.append(triton.Config({"BLOCK_SIZE_M": m}, num_warps=w, num_stages=s))


@triton.autotune(
    configs=_autotune_configs,
    key=[],
    cache_results=True,
)
@triton.jit
def _softmax_topk_kernel(
    # Pointers
    logits_ptr,
    weights_ptr,
    indices_ptr,
    norm_topk_prob: tl.constexpr,
    # Dimensions
    M: int,
    N: tl.constexpr,
    k: tl.constexpr,
    # Strides
    stride_lm: tl.constexpr,
    stride_ln: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    # Metadata
    BLOCK_SIZE_M: tl.constexpr = 16,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_m = offs_m[:, None]
    mask_m = offs_m < M
    offs_n = tl.arange(0, N)
    offs_n = offs_n[None, :]

    logits_ptrs = logits_ptr + stride_lm * offs_m + stride_ln * offs_n
    logits = tl.load(logits_ptrs, mask=mask_m, other=-float("inf"))

    # Softmax
    # Always cast to fp32
    logits = logits.to(tl.float32)
    logits -= tl.max(logits, axis=1, keep_dims=True)
    numerator = tl.exp(logits)
    denominator = tl.sum(numerator, axis=1, keep_dims=True)
    probs = numerator / denominator

    # Top k
    # We use a packing trick to sort values and indices together
    # packed = (prob << 16) | index
    # We can bitcast prob to uint32 because they're positive
    probs_u32 = probs.to(tl.uint32, bitcast=True)
    packed = (probs_u32.to(tl.uint64) << 16) | offs_n.to(tl.uint64)

    top_packed = tl.topk(packed, k, dim=1)

    top_probs = (top_packed >> 16).to(tl.uint32).to(tl.float32, bitcast=True)
    top_indices = (top_packed & 0xFFFF).to(tl.int32)

    if norm_topk_prob:
        top_sum = tl.sum(top_probs, axis=1, keep_dims=True)
        top_probs = top_probs / top_sum

    top_probs = top_probs.to(weights_ptr.dtype.element_ty)

    offs_k = tl.arange(0, k)
    offs_k = offs_k[None, :]
    weights_ptrs = weights_ptr + stride_wm * offs_m + stride_wk * offs_k
    indices_ptrs = indices_ptr + stride_im * offs_m + stride_ik * offs_k
    tl.store(weights_ptrs, top_probs, mask=mask_m)
    tl.store(indices_ptrs, top_indices, mask=mask_m)


@triton.autotune(
    configs=_autotune_configs,
    key=[],
    cache_results=True,
)
@triton.jit
def _softmax_topk_norm_backward_kernel(
    # Pointers
    grad_weights_ptr,
    weights_ptr,
    indices_ptr,
    grad_logits_ptr,
    # Dimensions
    M: int,
    N: tl.constexpr,
    k: tl.constexpr,
    # Strides
    stride_gwm: tl.constexpr,
    stride_gwk: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_glm: tl.constexpr,
    stride_gln: tl.constexpr,
    # Metadata
    BLOCK_SIZE_M: tl.constexpr = 16,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_m = offs_m[:, None]
    mask_m = offs_m < M
    offs_k = tl.arange(0, k)
    offs_k = offs_k[None, :]

    grad_weights_ptrs = grad_weights_ptr + stride_gwm * offs_m + stride_gwk * offs_k
    weights_ptrs = weights_ptr + stride_wm * offs_m + stride_wk * offs_k
    indices_ptrs = indices_ptr + stride_im * offs_m + stride_ik * offs_k

    grad_weights = tl.load(grad_weights_ptrs, mask=mask_m)
    weights = tl.load(weights_ptrs, mask=mask_m)
    indices = tl.load(indices_ptrs, mask=mask_m)

    gl_sum = tl.sum(grad_weights * weights, axis=1, keep_dims=True)
    gl = weights * (grad_weights - gl_sum)
    gl = gl.to(grad_logits_ptr.dtype.element_ty)

    gl_ptrs = grad_logits_ptr + stride_glm * offs_m + stride_gln * indices
    tl.store(gl_ptrs, gl, mask=mask_m)


def _softmax_topk_forward(logits: torch.Tensor, k: int, norm_topk_prob: bool) -> tuple[torch.Tensor, torch.Tensor]:
    assert logits.is_cuda
    assert logits.is_contiguous()
    assert logits.ndim == 2
    M, N = logits.shape

    weights = torch.empty((M, k), device=logits.device, dtype=logits.dtype)
    indices = torch.empty((M, k), device=logits.device, dtype=torch.int32)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)
    _softmax_topk_kernel[grid](
        # Pointers
        logits,
        weights,
        indices,
        norm_topk_prob,
        # Dimensions
        M,
        N,
        k,
        # Strides
        logits.stride(0),
        logits.stride(1),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
    )

    return weights, indices


def _softmax_topk_backward(
    grad_weights: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, N: int, norm_topk_prob: bool
) -> torch.Tensor:
    # TODO: Implement norm_topk_prob == False
    assert norm_topk_prob

    assert grad_weights.is_cuda
    assert weights.device == grad_weights.device
    assert indices.device == grad_weights.device
    assert grad_weights.is_contiguous()
    assert weights.is_contiguous()
    assert indices.is_contiguous()
    assert grad_weights.ndim == 2
    assert weights.shape == grad_weights.shape
    assert indices.shape == grad_weights.shape
    M, k = grad_weights.shape

    grad_logits = torch.zeros((M, N), device=grad_weights.device, dtype=grad_weights.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)
    _softmax_topk_norm_backward_kernel[grid](
        # Pointers
        grad_weights,
        weights,
        indices,
        grad_logits,
        # Dimensions
        M,
        N,
        k,
        # Strides
        grad_weights.stride(0),
        grad_weights.stride(1),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        grad_logits.stride(0),
        grad_logits.stride(1),
    )

    return grad_logits


class SoftmaxTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, k, norm_topk_prob):
        weights, indices = _softmax_topk_forward(logits, k, norm_topk_prob)
        ctx.save_for_backward(weights, indices)
        ctx.N = logits.shape[1]
        ctx.norm_topk_prob = norm_topk_prob
        return weights, indices

    @staticmethod
    def backward(ctx, grad_weights, grad_indices):
        weights, indices = ctx.saved_tensors
        grad_logits = _softmax_topk_backward(grad_weights, weights, indices, ctx.N, ctx.norm_topk_prob)
        return grad_logits, None, None


softmax_topk = SoftmaxTopK.apply
