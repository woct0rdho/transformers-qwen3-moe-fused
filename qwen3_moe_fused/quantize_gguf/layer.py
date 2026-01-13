from typing import Optional

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from ..modular_qwen3_moe_fused import MoeFusedLinear, moe_fused_linear
from .dequant import dequantize


def reverse_permute_weights(weights: torch.Tensor, n_head: int, num_kv_heads: Optional[int] = None) -> torch.Tensor:
    if num_kv_heads is not None and n_head != num_kv_heads:
        n_head = num_kv_heads
    dim = weights.shape[0] // n_head // 2
    w = weights.view(n_head, dim, 2, *weights.shape[1:])
    return w.transpose(2, 1).reshape(weights.shape)


class GGUFEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.register_buffer("weight", None)
        self.tensor_type = None
        self.original_shape = None
        self.compute_dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tensor_type is None:
            if self.weight is not None and self.weight.is_floating_point():
                return F.embedding(x, self.weight, self.padding_idx)
            raise RuntimeError("GGUFEmbedding not initialized with GGUF data")

        dtype = self.compute_dtype if self.compute_dtype is not None else torch.get_default_dtype()
        w = dequantize(self.weight, self.tensor_type, self.original_shape, dtype)
        w = w.to(x.device)

        expected = (self.num_embeddings, self.embedding_dim)
        if w.shape == expected:
            pass
        elif w.shape == (self.embedding_dim, self.num_embeddings):
            w = w.T
        elif w.numel() == np.prod(expected):
            w = w.view(expected)
        else:
            raise RuntimeError(f"Shape mismatch in GGUFEmbedding: expected {expected}, got {w.shape}")

        return F.embedding(x, w, self.padding_idx)


class GGUFLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        n_head: Optional[int] = None,
        n_kv_head: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.tensor_type = None
        self.original_shape = None

        self.n_head = n_head
        self.n_kv_head = n_kv_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tensor_type is None:
            if self.weight is not None and self.weight.is_floating_point():
                return F.linear(x, self.weight, self.bias)
            raise RuntimeError("GGUFLinear not initialized with GGUF data")

        w = dequantize(self.weight, self.tensor_type, self.original_shape, x.dtype)
        w = w.to(x.device)

        expected = (self.out_features, self.in_features)
        if w.shape == expected:
            pass
        elif w.shape == (self.in_features, self.out_features):
            w = w.T
        elif w.numel() == self.out_features * self.in_features:
            w = w.view(expected)
        else:
            raise RuntimeError(f"Shape mismatch in GGUFLinear: expected {expected}, got {w.shape}")

        if self.n_head is not None:
            w = reverse_permute_weights(w, self.n_head, self.n_kv_head)

        return F.linear(x, w, self.bias)


class GGUFMoeFusedLinear(MoeFusedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(in_features, out_features, num_experts, device=device, dtype=dtype)
        if hasattr(self, "weight"):
            del self.weight
        self.register_buffer("weight", None)
        self.tensor_type = None
        self.original_shape = None
        self.compute_dtype = dtype

    def forward(self, x: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
        if self.tensor_type is None:
            if self.weight is not None and self.weight.is_floating_point():
                return moe_fused_linear(x, self.weight, m_sizes)
            raise RuntimeError("GGUFMoeFusedLinear not initialized with GGUF data")

        w = dequantize(self.weight, self.tensor_type, self.original_shape, x.dtype)
        w = w.to(x.device)

        expected = (self.num_experts, self.out_features, self.in_features)
        if w.shape == expected:
            pass
        elif w.shape == (self.num_experts, self.in_features, self.out_features):
            w = w.transpose(1, 2)
        elif w.numel() == np.prod(expected):
            w = w.view(expected)
        else:
            raise RuntimeError(f"Shape mismatch in GGUFMoeFusedLinear: expected {expected}, got {w.shape}")

        return moe_fused_linear(x, w, m_sizes)
