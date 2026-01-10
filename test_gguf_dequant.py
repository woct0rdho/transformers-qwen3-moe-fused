#!/usr/bin/env python3

import ctypes
from math import prod

import gguf
import numpy as np
import torch

from qwen3_moe_fused.quantize_gguf.dequant import dequantize, dequantize_functions
from test_utils import get_rtol_atol


# Modified from https://github.com/ggml-org/llama.cpp/blob/e54d41befcc1575f4c898c5ff4ef43970cead75f/gguf-py/tests/test_quants.py
class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


class GGMLQuants:
    def __init__(self, libggml_path: str):
        self.libggml = ctypes.CDLL(libggml_path)
        self.libggml.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.libggml.ggml_quantize_chunk.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        )

        self.libggml.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self.libggml.ggml_quantize_requires_imatrix.argtypes = (ctypes.c_int,)

        if hasattr(self.libggml, "ggml_init"):
            self.libggml.ggml_init.argtypes = (ggml_init_params,)
            self.libggml.ggml_init(ggml_init_params(1 * 1024 * 1024, 0, False))

    def quantize(self, data: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
        data = data.astype(np.float32, copy=False)
        result = np.zeros(gguf.quant_shape_to_byte_shape(data.shape, qtype), dtype=np.uint8, order="C")

        c_float_p = ctypes.POINTER(ctypes.c_float)

        if self.libggml.ggml_quantize_requires_imatrix(qtype.value):
            qw = np.sum((data * data).reshape((-1, data.shape[-1])), axis=0).ctypes.data_as(c_float_p)
        else:
            qw = ctypes.cast(0, c_float_p)

        self.libggml.ggml_quantize_chunk(
            qtype.value,
            data.ctypes.data_as(c_float_p),
            result.ctypes.data_as(ctypes.c_void_p),
            0,
            prod(data.shape[:-1]),
            data.shape[-1],
            qw,
        )
        return result


def main():
    n_blocks = 32
    device = "cuda"

    dll_path = r"C:\llama.cpp\build\bin\ggml-base.dll"
    ggml_quants = GGMLQuants(dll_path)

    for qtype in dequantize_functions.keys():
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        numel = n_blocks * block_size
        weights = np.random.uniform(-1, 1, numel).astype(np.float32)

        # Quantize may be slow. We only need to dequantize fast
        quantized = ggml_quants.quantize(weights, qtype)
        print("quantized", qtype.name, quantized.shape, quantized.dtype)

        out_ref = gguf.quants.dequantize(quantized, qtype)
        out_ref = torch.from_numpy(out_ref)

        quantized_gpu = torch.from_numpy(quantized).to(device)
        out = dequantize(quantized_gpu, qtype, (numel,), torch.float32).cpu()

        if torch.isnan(out).any():
            print("out contains NaN")
        if torch.isnan(out_ref).any():
            print("ref contains NaN")

        rtol = 1e-8
        atol = 1e-8
        print(torch.allclose(out, out_ref, rtol=rtol, atol=atol))
        print(get_rtol_atol(out, out_ref))


if __name__ == "__main__":
    main()
