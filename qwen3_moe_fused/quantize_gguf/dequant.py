# Modified from https://github.com/city96/ComfyUI-GGUF/blob/795e45156ece99afbc3efef911e63fcb46e6a20d/dequant.py

import gguf
import numpy as np
import torch
from gguf.quants import IQ1_S, IQ2_S, IQ2_XXS, IQ3_S, IQ3_XXS


# IQ quants
KVALUES = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.int8)

IQ3_S.init_grid()
GRID_IQ3_S = torch.from_numpy(IQ3_S.grid).squeeze()

IQ3_XXS.init_grid()
GRID_IQ3_XXS = torch.from_numpy(IQ3_XXS.grid).squeeze()
KSIGNS_IQ2_XXS = torch.from_numpy(np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).copy())

IQ2_S.init_grid()
GRID_IQ2_S = torch.from_numpy(IQ2_S.grid).squeeze()

IQ1_S.init_grid()
GRID_IQ1_S = torch.from_numpy(IQ1_S.grid).squeeze()

TORCH_COMPATIBLE_QTYPES = {
    gguf.GGMLQuantizationType.F32: torch.float32,
    gguf.GGMLQuantizationType.F16: torch.float16,
}


def dequantize(data, qtype, oshape, dtype=None):
    if qtype in TORCH_COMPATIBLE_QTYPES:
        if dtype is None:
            dtype = TORCH_COMPATIBLE_QTYPES[qtype]
        return data.to(dtype)

    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]

    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


def to_uint32(x):
    # no uint32 :(
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


# Full weights #
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


# Legacy Quants #
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = ql | (qh << 4)
    return (d * qs) + m


def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m


def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


# K Quants #
QK_K = 256
K_SCALE_SIZE = 12


def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    (
        ql,
        qh,
        scales,
        d,
    ) = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)

    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 2, 1)
    )
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8) - 32

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)

    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))


def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int64)

    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))

    return d * qs


def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = to_uint16(scales_h)

    shift_a = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2))
    shift_b = torch.tensor([2 * i for i in range(QK_K // 32)], device=d.device, dtype=torch.uint8).reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03

    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & 0x0F

    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64)).reshape((n_blocks, -1, 32))

    return (dl * qs).reshape((n_blocks, -1))


def dequantize_blocks_IQ3_S(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs, qh, signs, scales = split_block_dims(blocks, 2, 64, 8, 32)

    d = d.view(torch.float16).to(dtype)

    # Scales
    scales = scales.view(torch.uint8)
    scales = torch.stack([scales & 0xF, scales >> 4], dim=-1).reshape(n_blocks, 8)
    db = d * (1 + 2 * scales.to(dtype))
    db = db.reshape(n_blocks, 8, 1, 1)

    # Signs
    shifts = torch.arange(8, device=d.device, dtype=torch.uint8).reshape(1, 1, 8)
    signs = (signs.unsqueeze(-1) >> shifts) & 1
    signs = torch.where(
        signs == 0, torch.tensor(1.0, dtype=dtype, device=d.device), torch.tensor(-1.0, dtype=dtype, device=d.device)
    )
    signs = signs.reshape(n_blocks, 8, 8, 4)

    # Grid
    qh_bits = (qh.unsqueeze(-1) >> shifts) & 1
    qh_bits = qh_bits.reshape(n_blocks, 64)
    qs = qs.to(torch.int16) | (qh_bits.to(torch.int16) << 8)

    grid_val = GRID_IQ3_S.to(dtype=dtype, device=d.device)[qs.to(torch.long)]
    grid_val = grid_val.reshape(n_blocks, 8, 8, 4)

    return (db * grid_val * signs).reshape(n_blocks, 256)


def dequantize_blocks_IQ3_XXS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs, scales, _ = split_block_dims(blocks, 2, 64, 32)

    d = d.view(torch.float16).to(dtype)
    scales = scales.clone().view(torch.int32)

    db = d * (0.5 + ((scales >> 28) & 0xF).to(dtype)) * 0.5
    db = db.reshape(n_blocks, 8, 1, 1)

    # signs
    shifts = torch.tensor([0, 7, 14, 21], device=d.device, dtype=torch.int32).reshape(1, 1, 4)
    signs = (scales.reshape(n_blocks, 8, 1) >> shifts) & 0x7F

    signs = KSIGNS_IQ2_XXS.to(d.device)[signs]  # (n_blocks, 8, 4)

    shifts_bits = torch.arange(8, device=d.device, dtype=torch.uint8).reshape(1, 1, 1, 8)
    signs = (signs.unsqueeze(-1) >> shifts_bits) & 1
    signs = torch.where(
        signs == 0, torch.tensor(1.0, dtype=dtype, device=d.device), torch.tensor(-1.0, dtype=dtype, device=d.device)
    )

    # grid
    grid_val = GRID_IQ3_XXS.to(dtype=dtype, device=d.device)[qs.to(torch.long)]
    grid_val = grid_val.reshape(n_blocks, 8, 4, 8)

    return (db * grid_val * signs).reshape(n_blocks, 256)


def dequantize_blocks_IQ1_S(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs, qh = split_block_dims(blocks, 2, 32)
    d = d.view(torch.float16).to(dtype)

    # qh is 16 bytes -> 8 uint16s
    # Offset 34 is aligned to 2, so direct view is safe
    qh = qh.view(torch.int16).to(torch.int32) & 0xFFFF  # (N, 8)

    # dl and delta from original qh (N, 8)
    dl = d * (2 * ((qh >> 12) & 7).to(dtype) + 1)
    # dl shape (N, 8)

    delta = torch.where(
        (qh & 0x8000) == 0,
        torch.tensor(0.125, dtype=dtype, device=d.device),
        torch.tensor(-0.125, dtype=dtype, device=d.device),
    )
    # delta shape (N, 8)

    # Unpack qh for grid indices
    # qh (N, 8) -> (N, 8, 1) >> [0, 3, 6, 9] -> (N, 8, 4)
    shifts = torch.tensor([0, 3, 6, 9], device=d.device, dtype=torch.int32).reshape(1, 1, 4)
    qh_bits = (qh.reshape(n_blocks, 8, 1) >> shifts) & 7
    # qh_bits (N, 8, 4)

    # qs (N, 32) uint8.
    qs = qs.view(torch.uint8).to(torch.int32)
    qs = qs.reshape(n_blocks, 8, 4)

    # Combine
    qs = qs | (qh_bits << 8)  # (N, 8, 4)

    # Grid lookup
    grid_val = GRID_IQ1_S.to(dtype=dtype, device=d.device)[qs.to(torch.long)]
    # grid_val shape (N, 8, 4, 8)

    dl = dl.reshape(n_blocks, 8, 1, 1)
    delta = delta.reshape(n_blocks, 8, 1, 1)

    res = dl * (grid_val + delta)

    return res.reshape(n_blocks, 256)


def dequantize_blocks_IQ2_S(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs, signs, qh, scales = split_block_dims(blocks, 2, 32, 32, 8)
    d = d.view(torch.float16).to(dtype)

    # scales: 8 bytes -> 16 scales
    scales = scales.view(torch.uint8)
    scales = torch.stack([scales & 0xF, scales >> 4], dim=-1).reshape(n_blocks, 16)
    db = d * (0.5 + scales.to(dtype)) * 0.25
    db = db.reshape(n_blocks, 16, 1, 1)

    # signs: 32 bytes -> 256 signs
    shifts = torch.arange(8, device=d.device, dtype=torch.uint8).reshape(1, 1, 8)
    signs = (signs.unsqueeze(-1) >> shifts) & 1
    signs = torch.where(
        signs == 0, torch.tensor(1.0, dtype=dtype, device=d.device), torch.tensor(-1.0, dtype=dtype, device=d.device)
    )
    signs = signs.reshape(n_blocks, 16, 2, 8)

    # qh: 8 bytes -> 32 high bit groups (2 bits each)
    qh_shifts = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(1, 1, 4)
    qh_bits = (qh.view(torch.uint8).reshape(n_blocks, 8, 1) >> qh_shifts) & 3
    qh_bits = qh_bits.reshape(n_blocks, 32)

    # qs: 32 bytes -> 32 low bits
    qs = qs.view(torch.uint8).to(torch.int32)

    # Combined indices (10-bit)
    indices = qs | (qh_bits.to(torch.int32) << 8)

    # Grid lookup
    grid_val = GRID_IQ2_S.to(dtype=dtype, device=d.device)[indices.to(torch.long)]
    grid_val = grid_val.reshape(n_blocks, 16, 2, 8)

    res = db * grid_val * signs
    return res.reshape(n_blocks, 256)


dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
    gguf.GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS: dequantize_blocks_IQ4_XS,
    gguf.GGMLQuantizationType.IQ3_S: dequantize_blocks_IQ3_S,
    gguf.GGMLQuantizationType.IQ3_XXS: dequantize_blocks_IQ3_XXS,
    gguf.GGMLQuantizationType.IQ2_S: dequantize_blocks_IQ2_S,
    gguf.GGMLQuantizationType.IQ1_S: dequantize_blocks_IQ1_S,
}
