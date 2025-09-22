# Modified from https://github.com/huggingface/transformers/blob/bdf5fb70aa11782cce22027d76879f71f4e41c1e/src/transformers/models/qwen3_moe/modular_qwen3_moe.py

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
)
from transformers.utils.generic import OutputRecorder
from .functional import moe_fused_linear
from .kernels.indexing import get_expert_counts_and_idx


def moe_fused_kaiming_uniform_(weight: torch.Tensor) -> None:
    # Kaiming uniform on in_features
    # Although Qwen's default activation is silu, we set the gain `a = sqrt(5)` following the original Linear
    in_features = weight.shape[-1]
    bound = math.sqrt(3 * 5 / in_features)
    nn.init.uniform_(weight, -bound, bound)


class MoeFusedLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_experts"]
    in_features: int
    out_features: int
    num_experts: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty((num_experts, out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        moe_fused_kaiming_uniform_(self.weight)

    def forward(self, input: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
        return moe_fused_linear(input, self.weight, m_sizes)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, num_experts={self.num_experts}"


# This class follows the implementation in HF Transformers
# patch_Qwen3MoeFusedSparseMoeBlock_forward can make it faster
class Qwen3MoeFusedSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.num_selected = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.gate_proj = MoeFusedLinear(self.hidden_size, self.moe_intermediate_size, config.num_experts)
        self.up_proj = MoeFusedLinear(self.hidden_size, self.moe_intermediate_size, config.num_experts)
        self.down_proj = MoeFusedLinear(self.moe_intermediate_size, self.hidden_size, config.num_experts)
        assert config.hidden_act == "silu"

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        M = batch_size * sequence_length

        hidden_states = hidden_states.view(M, hidden_dim)
        # router_logits: (M, num_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        # routing_weights, selected_experts: (M, num_selected)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_selected, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        hidden_states = hidden_states.unsqueeze(1).expand(M, self.num_selected, hidden_dim)
        # hidden_states must be contiguous
        hidden_states = hidden_states.reshape(M * self.num_selected, hidden_dim)
        selected_experts = selected_experts.view(M * self.num_selected)

        # Sort selected_experts and hidden_states for better memory coalescence of weight
        # It's possible to fuse a sort and a MoeFusedLinear layer, but for now we separate them for clarity
        m_sizes, sort_idx, inv_sort_idx = get_expert_counts_and_idx(selected_experts, self.num_experts)
        hidden_states = hidden_states[sort_idx]

        # It's possible to fuse gate_h and up_h, but this affects the shape of LoRA
        gate_h = self.gate_proj(hidden_states, m_sizes)
        up_h = self.up_proj(hidden_states, m_sizes)
        hidden_states = F.silu(gate_h) * up_h
        del gate_h, up_h
        hidden_states = self.down_proj(hidden_states, m_sizes)

        hidden_states = hidden_states[inv_sort_idx]

        hidden_states = hidden_states.view(M, self.num_selected, hidden_dim)
        hidden_states = torch.einsum("beo,be->bo", hidden_states, routing_weights)

        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits


class Qwen3MoeFusedDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeFusedSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)


class Qwen3MoeFusedModel(Qwen3MoeModel):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3MoeFusedDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class Qwen3MoeFusedForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__(config)
        self.model = Qwen3MoeFusedModel(config)
        self._can_record_outputs["router_logits"] = OutputRecorder(Qwen3MoeFusedSparseMoeBlock, index=1)
        self._can_record_outputs["hidden_states"] = Qwen3MoeFusedDecoderLayer
