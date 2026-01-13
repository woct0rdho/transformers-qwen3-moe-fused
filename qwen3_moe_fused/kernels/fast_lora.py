# Analogous to Unsloth fast LoRA: https://github.com/unslothai/unsloth/blob/91598a6ee8ecda6dbaa2c9fd1ea9c75719da54a6/unsloth/kernels/fast_lora.py
#
# In the forward pass, we have the inputs and their shapes:
# x                          (B, H)
# G, U                       (E, M, H)
# Ag, Au                     (E, R, H)
# Bg, Bu                     (E, M, R)
# W                          (E, H, M)
# Aw                         (E, R, M)
# Bw                         (E, H, R)
# Sg, Su, Sw                 scalar
#
# The intermediate results, without the matrix multiplication notations:
# G' = G + Sg * Ag * Bg      (E, M, H)
# U' = U + Su * Au * Bu      (E, M, H)
# W' = W + Sw * Aw * Bw      (E, H, M)
# e = G' * x                 (B, M)
# g = U' * x                 (B, M)
# h = e * σ(e) * g           (B, M)
# y = W' * h                 (B, H)
#
# In the backward pass, we have:
# dy                         (B, H)
# dh = W' * dy               (B, M)
# h = e * σ(e) * g           (B, M)
# de = dh * d(e * σ(e)) * g  (B, M)     d(e * σ(e)) = σ(e) * (1 + e - e * σ(e))
# dg = e * σ(e) * dh         (B, M)
#
# dW' = dy * h               (E, H, M)
# dAw = Sw * dW' * Bw        (E, R, M)
# dBw = Sw * Aw * dW'        (E, H, R)
#
# dU' = dg * x               (E, M, H)
# dAu = Su * dU' * Bu        (E, R, H)
# dBu = Su * Au * dU'        (E, M, R)
#
# dG' = de * x               (E, M, H)
# dAg = Sg * dG' * Bg        (E, R, H)
# dBg = Sg * Ag * dG'        (E, M, R)
#
# dx = G' * de + U' * dg     (B, H)
#
# Then we can write out the matrix multiplications according to the shapes
#
# We only need to store the intermediate results e and g
# If VRAM is very limited, we can recompute them rather than store them

import torch
from bitsandbytes.functional import dequantize_4bit

from ..grouped_gemm.backward_dw import grouped_gemm_backward_dw
from ..grouped_gemm.forward import grouped_gemm_forward
from .silu_mul import silu_mul_backward_inplace, silu_mul_forward


# TODO: Fuse dequant and linear operations
def _maybe_dequant(weight, quant_state):
    if quant_state is None:
        return weight
    else:
        out = dequantize_4bit(weight, quant_state)
        return out


class FastLora(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Gq, Gqs, Ag, Bg, Sg, Uq, Uqs, Au, Bu, Su, Wq, Wqs, Aw, Bw, Sw, m_sizes):
        # Cast all weights to x.dtype
        # The grouped GEMM kernels use float32 accumulator
        if Gqs is None:
            Gq = Gq.to(x.dtype)
        else:
            Gqs.dtype = x.dtype
        Ag = Ag.to(x.dtype)
        Bg = Bg.to(x.dtype)

        if Uqs is None:
            Uq = Uq.to(x.dtype)
        else:
            Uqs.dtype = x.dtype
        Au = Au.to(x.dtype)
        Bu = Bu.to(x.dtype)

        if Wqs is None:
            Wq = Wq.to(x.dtype)
        else:
            Wqs.dtype = x.dtype
        Aw = Aw.to(x.dtype)
        Bw = Bw.to(x.dtype)

        def mv(_w, _x):
            return grouped_gemm_forward(_x, _w, m_sizes, x.dtype)

        def mv_lora(_x, _wq, _wqs, _a, _b, _s):
            _w = _maybe_dequant(_wq, _wqs)
            _y = mv(_w, _x)
            _y += mv(_b, mv(_a, _x)) * _s
            return _y

        e = mv_lora(x, Gq, Gqs, Ag, Bg, Sg)
        g = mv_lora(x, Uq, Uqs, Au, Bu, Su)
        h = silu_mul_forward(e, g)
        y = mv_lora(h, Wq, Wqs, Aw, Bw, Sw)

        ctx.custom_saved_tensors = (Gq, Gqs, Sg, Uq, Uqs, Su, Wq, Wqs, Sw, m_sizes)
        ctx.save_for_backward(x, Ag, Bg, Au, Bu, Aw, Bw, e, g)
        return y

    @staticmethod
    def backward(ctx, dy):
        Gq, Gqs, Sg, Uq, Uqs, Su, Wq, Wqs, Sw, m_sizes = ctx.custom_saved_tensors
        x, Ag, Bg, Au, Bu, Aw, Bw, e, g = ctx.saved_tensors

        def vm(_x, _w):
            return grouped_gemm_forward(_x, _w, m_sizes, x.dtype, transpose_w=True)

        def vv(_y, _x):
            return grouped_gemm_backward_dw(_x, _y, m_sizes, x.dtype)

        def emv(_w, _x):
            return torch.einsum("eri,eji->erj", _w, _x).to(x.dtype)

        def evm(_x, _w):
            return torch.einsum("eij,eir->ejr", _x, _w).to(x.dtype)

        def vm_lora(_x, _wq, _wqs, _a, _b, _s):
            _w = _maybe_dequant(_wq, _wqs)
            _y = vm(_x, _w)
            _y += vm(vm(_x, _b), _a) * _s
            return _y

        dh = vm_lora(dy, Wq, Wqs, Aw, Bw, Sw)
        # dh, e, g are modified in place to h, de, dg
        silu_mul_backward_inplace(dh, e, g)
        h, de, dg = dh, e, g
        del dh, e, g

        dW = vv(dy, h)
        dAw = evm(Bw, dW) * Sw
        dBw = emv(dW, Aw) * Sw

        dU = vv(dg, x)
        dAu = evm(Bu, dU) * Su
        dBu = emv(dU, Au) * Su

        dG = vv(de, x)
        dAg = evm(Bg, dG) * Sg
        dBg = emv(dG, Ag) * Sg

        dx = vm_lora(de, Gq, Gqs, Ag, Bg, Sg)
        dx += vm_lora(dg, Uq, Uqs, Au, Bu, Su)

        return (
            dx,
            None,  # Gq
            None,  # Gqs
            dAg,
            dBg,
            None,  # Sg
            None,  # Uq
            None,  # Uqs
            dAu,
            dBu,
            None,  # Su
            None,  # Wq
            None,  # Wqs
            dAw,
            dBw,
            None,  # Sw
            None,  # m_sizes
        )


fast_lora = FastLora.apply
