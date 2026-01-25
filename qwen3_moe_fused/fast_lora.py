from typing import Optional

import torch
from bitsandbytes.functional import QuantState
from torch import nn

from .kernels.fast_lora import fast_lora
from .kernels.indexing import get_expert_offsets_and_idx
from .kernels.softmax_topk import softmax_topk
from .modular_qwen3_moe_fused import Qwen3MoeFusedSparseMoeBlock


def get_lora_parameters(
    proj: nn.Module,
) -> tuple[nn.Parameter, Optional[QuantState], nn.Parameter, nn.Parameter, float]:
    base_layer = getattr(proj, "base_layer", proj)
    W = base_layer.weight

    if getattr(proj, "disable_adapters", False) or getattr(proj, "merged", False):
        raise NotImplementedError

    adapters = getattr(proj, "active_adapters", None)
    if adapters is None:
        adapter = getattr(proj, "active_adapter", "default")
    else:
        assert len(adapters) == 1
        adapter = adapters[0]

    assert isinstance(proj.lora_dropout[adapter], nn.Identity)
    return (
        W,
        getattr(W, "quant_state", None),
        proj.lora_A[adapter].weight,
        proj.lora_B[adapter].weight,
        proj.scaling[adapter],
    )


def fast_Qwen3MoeFusedSparseMoeBlock_forward(
    self: Qwen3MoeFusedSparseMoeBlock, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    M = batch_size * sequence_length

    hidden_states = hidden_states.view(M, hidden_dim)
    # router_logits: (M, num_experts)
    router_logits = self.gate(hidden_states)

    routing_weights, selected_experts = softmax_topk(router_logits, self.num_selected, self.norm_topk_prob)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    hidden_states = hidden_states.unsqueeze(1).expand(M, self.num_selected, hidden_dim)
    # hidden_states must be contiguous
    hidden_states = hidden_states.reshape(M * self.num_selected, hidden_dim)
    selected_experts = selected_experts.view(M * self.num_selected)

    # Sort selected_experts and hidden_states for better memory coalescence of weight
    # It's possible to fuse a sort and a MoeFusedLinear layer, but for now we separate them for clarity
    m_offsets, sort_idx, inv_sort_idx = get_expert_offsets_and_idx(selected_experts, self.num_experts)
    hidden_states = hidden_states[sort_idx]

    Gq, Gqs, Ag, Bg, Sg = get_lora_parameters(self.gate_proj)
    Uq, Uqs, Au, Bu, Su = get_lora_parameters(self.up_proj)
    Wq, Wqs, Aw, Bw, Sw = get_lora_parameters(self.down_proj)
    hidden_states = fast_lora(hidden_states, Gq, Gqs, Ag, Bg, Sg, Uq, Uqs, Au, Bu, Su, Wq, Wqs, Aw, Bw, Sw, m_offsets)

    hidden_states = hidden_states[inv_sort_idx]

    hidden_states = hidden_states.view(M, self.num_selected, hidden_dim)
    hidden_states = torch.einsum("beo,be->bo", hidden_states, routing_weights)

    hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
    return hidden_states, router_logits


def patch_Qwen3MoeFusedSparseMoeBlock_forward() -> None:
    Qwen3MoeFusedSparseMoeBlock.forward = fast_Qwen3MoeFusedSparseMoeBlock_forward
