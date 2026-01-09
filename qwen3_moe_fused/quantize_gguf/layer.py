from typing import Optional

import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize
from torch import nn as nn
from torch.nn import functional as F

from ..modular_qwen3_moe_fused import MoeFusedLinear, moe_fused_linear


def gguf_dequantize(weight_data: torch.Tensor, tensor_type: GGMLQuantizationType, shape: tuple) -> torch.Tensor:
    if weight_data.device.type != "cpu":
        data_np = weight_data.detach().cpu().numpy()
    else:
        data_np = weight_data.detach().numpy()
    w_np = dequantize(data_np, tensor_type)
    w_np = w_np.reshape(shape)
    return torch.from_numpy(w_np)


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

        w = gguf_dequantize(self.weight, self.tensor_type, self.original_shape)
        dtype = self.compute_dtype if self.compute_dtype is not None else torch.get_default_dtype()
        w = w.to(device=x.device, dtype=dtype)

        if w.shape != (self.num_embeddings, self.embedding_dim):
            if w.shape == (self.embedding_dim, self.num_embeddings):
                w = w.t()
            else:
                if w.numel() == self.num_embeddings * self.embedding_dim:
                    w = w.view(self.num_embeddings, self.embedding_dim)

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

        w = gguf_dequantize(self.weight, self.tensor_type, self.original_shape)
        w = w.to(device=x.device, dtype=x.dtype)

        if w.shape == (self.out_features, self.in_features):
            pass
        elif w.shape == (self.in_features, self.out_features):
            w = w.t()
        else:
            if w.numel() == self.out_features * self.in_features:
                w = w.view(self.out_features, self.in_features)
            else:
                raise RuntimeError(
                    f"Shape mismatch in GGUFLinear: expected {(self.out_features, self.in_features)}, "
                    f"got {w.shape} (original {self.original_shape})"
                )

        if self.n_head is not None:
            w = reverse_permute_weights(w, self.n_head, self.n_kv_head)

        return F.linear(x, w, self.bias)


class MoeFusedLinearGGUF(MoeFusedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(in_features, out_features, num_experts, device=device, dtype=dtype)
        self.register_buffer("weight", None)
        self.tensor_type = None
        self.original_shape = None
        self.compute_dtype = dtype

    def forward(self, x: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
        if self.tensor_type is None:
            if self.weight is not None and self.weight.is_floating_point():
                if moe_fused_linear:
                    return moe_fused_linear(x, self.weight, m_sizes)
            raise RuntimeError("MoeFusedLinearGGUF not initialized with GGUF data")

        w = gguf_dequantize(self.weight, self.tensor_type, self.original_shape)
        w = w.to(device=x.device, dtype=x.dtype)
        expected = (self.num_experts, self.out_features, self.in_features)

        if w.shape == expected:
            pass
        elif w.shape == (self.num_experts, self.in_features, self.out_features):
            w = w.transpose(1, 2)
        else:
            if w.numel() == np.prod(expected):
                w = w.view(expected)
            else:
                raise RuntimeError(f"Shape mismatch in MoeFusedLinearGGUF: expected {expected}, got {w.shape}")

        if moe_fused_linear:
            return moe_fused_linear(x, w, m_sizes)
        else:
            raise ImportError("moe_fused_linear not found")
