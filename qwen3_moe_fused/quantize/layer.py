# Modified from https://github.com/bitsandbytes-foundation/bitsandbytes/blob/888788d75db8ff8e8888838307119f98d1235c24/bitsandbytes/nn/modules.py#L377
# TODO: support IPEX

import warnings
from typing import Any, Optional

import torch
from bitsandbytes.functional import dequantize_4bit
from bitsandbytes.nn.modules import Params4bit, fix_4bit_weight_quant_state_from_module
from torch import nn

from ..functional import moe_fused_linear
from ..modular_qwen3_moe_fused import MoeFusedLinear


# TODO: Fuse this
def moe_fused_linear_4bit(input: torch.Tensor, weight: Params4bit, m_offsets: torch.Tensor) -> torch.Tensor:
    assert not weight.requires_grad
    # Cast weight to input.dtype
    # The grouped GEMM kernels use float32 accumulator
    weight = dequantize_4bit(weight, weight.quant_state).to(input.dtype)
    return moe_fused_linear(input, weight, m_offsets)


class MoeFusedLinear4bit(MoeFusedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        *,
        weight: Optional[nn.Parameter] = None,  # Used for initializing from a non-quantized module
        compute_dtype: Optional[torch.dtype] = None,
        compress_statistics: bool = True,
        quant_type: str = "fp4",
        quant_storage: torch.dtype = torch.uint8,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(in_features, out_features, num_experts, device=device)
        self.weight = Params4bit(
            self.weight,
            requires_grad=False,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
            quant_storage=quant_storage,
            module=self,
        )
        # self.persistent_buffers = []  # TODO consider as way to save quant state
        self.compute_dtype = compute_dtype
        self.compute_type_is_set = compute_dtype is not None
        self.quant_state = None
        self.quant_storage = quant_storage

    def set_compute_type(self, x: torch.Tensor) -> None:
        if x.dtype in [torch.float32, torch.bfloat16]:
            # the input is in a dtype that is safe to compute in, we switch
            # to this type for speed and stability
            self.compute_dtype = x.dtype
        elif x.dtype == torch.float16:
            # we take the compoute dtype passed into the layer
            if self.compute_dtype in [None, torch.float32] and (x.numel() == x.shape[-1]):
                # single batch inference with input torch.float16 and compute_dtype float32 -> slow inference when it could be fast
                # warn the user about this
                warnings.warn(
                    "Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). "
                    "This will lead to slow inference.",
                )
                warnings.filterwarnings("ignore", message=".*inference.")
            if self.compute_dtype in [None, torch.float32] and (x.numel() != x.shape[-1]):
                warnings.warn(
                    "Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). "
                    "This will lead to slow inference or training speed.",
                )
                warnings.filterwarnings("ignore", message=".*inference or training")

    def _save_to_state_dict(self, destination: dict[str, Any], prefix: str, keep_vars: bool) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)

        if getattr(self.weight, "quant_state", None) is not None:
            for k, v in self.weight.quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()

    def forward(self, x: torch.Tensor, m_offsets: torch.Tensor) -> torch.Tensor:
        fix_4bit_weight_quant_state_from_module(self)

        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        x = moe_fused_linear_4bit(x, self.weight, m_offsets)
        x = x.to(inp_dtype)
        return x
