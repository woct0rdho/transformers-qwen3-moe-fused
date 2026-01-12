# Modified from https://github.com/huggingface/peft/blob/e34852f7b67d51ba7ef871051b1236e9558c650e/src/peft/tuners/lora/layer.py#L585

import functools
import math
from typing import Optional, Union

import torch
from peft import LoraConfig
from peft.tuners.lora.layer import Embedding as LoraEmbedding
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.layer import LoraLayer
from torch import nn

from .modular_qwen3_moe_fused import MoeFusedLinear, moe_fused_kaiming_uniform_
from .quantize_gguf.layer import GGUFEmbedding, GGUFLinear, GGUFMoeFusedLinear


class LoraMoeFusedLinear(nn.Module, LoraLayer):
    def __init__(
        self,
        base_layer: MoeFusedLinear,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        if init_lora_weights not in {True, False}:
            raise NotImplementedError
        if use_dora:
            raise NotImplementedError
        if lora_bias:
            raise NotImplementedError

        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.num_experts = base_layer.num_experts
        self._active_adapter = adapter_name

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
        )

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: Union[bool, str],
        use_rslora: bool,
    ) -> None:
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Actual trainable parameters
        self.lora_A[adapter_name] = MoeFusedLinear(self.in_features, r, self.num_experts)
        self.lora_B[adapter_name] = MoeFusedLinear(r, self.out_features, self.num_experts)

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights is True:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: Union[bool, str]) -> None:
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A:
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                moe_fused_kaiming_uniform_(self.lora_A[adapter_name].weight)
                nn.init.zeros_(self.lora_B[adapter_name].weight)
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError

    def unmerge(self) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, m_sizes: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, m_sizes, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError
            # result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            # In _mixed_batch_forward, we need to change `lora_B(lora_A(dropout(x)))`
            # to `lora_B(lora_A(dropout(x), m_sizes), m_sizes)`
        elif self.merged:
            result = self.base_layer(x, m_sizes, *args, **kwargs)
        else:
            result = self.base_layer(x, m_sizes, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                result = result + lora_B(lora_A(dropout(x), m_sizes), m_sizes) * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def patch_lora_config(*, rank_pattern: Optional[dict[str, int]] = None) -> None:
    old_init = LoraConfig.__init__

    @functools.wraps(old_init)
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self._register_custom_module(
            {
                MoeFusedLinear: LoraMoeFusedLinear,
                GGUFEmbedding: LoraEmbedding,
                GGUFLinear: LoraLinear,
                GGUFMoeFusedLinear: LoraMoeFusedLinear,
            }
        )

    LoraConfig.__init__ = new_init
