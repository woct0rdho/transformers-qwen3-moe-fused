from inspect import signature
from typing import Any, Optional, Union

from accelerate import init_empty_weights
from bitsandbytes.nn import Linear4bit
from torch import nn
from transformers import BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.utils import logging

from ..modular_qwen3_moe_fused import MoeFusedLinear
from .layer import MoeFusedLinear4bit


logger = logging.get_logger(__name__)


def _param_needs_quantization(self: Bnb4BitHfQuantizer, model: PreTrainedModel, param_name: str, **kwargs) -> bool:
    import bitsandbytes as bnb

    # They are on the params themselves, so we cannot easily extract the module from the name
    if any(param_name.endswith(x) for x in self.bnb_keys):
        return True
    module, name = get_module_from_name(model, param_name)
    return isinstance(module, (bnb.nn.Linear4bit, MoeFusedLinear4bit)) and name != "bias"


# Modified from https://github.com/huggingface/transformers/blob/508a7040556dc6b45f09174c662a9632284b2445/src/transformers/integrations/bitsandbytes.py#L150
def _replace_with_bnb_moe_fused_linear(
    model: nn.Module,
    modules_to_not_convert: list[str],
    current_key_name: list[str],
    quantization_config: BitsAndBytesConfig,
    has_been_replaced: bool,
) -> bool:
    for name, module in model.named_children():
        current_key_name.append(name)

        if isinstance(module, (nn.Linear, Conv1D, MoeFusedLinear)) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                num_experts = None
                if isinstance(module, MoeFusedLinear):
                    in_features = module.in_features
                    out_features = module.out_features
                    num_experts = module.num_experts
                elif isinstance(module, Conv1D):
                    in_features, out_features = module.weight.shape
                else:
                    in_features = module.in_features
                    out_features = module.out_features

                if isinstance(module, MoeFusedLinear):
                    model._modules[name] = MoeFusedLinear4bit(
                        in_features,
                        out_features,
                        num_experts,
                        compute_dtype=quantization_config.bnb_4bit_compute_dtype,
                        compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                        quant_type=quantization_config.bnb_4bit_quant_type,
                        quant_storage=quantization_config.bnb_4bit_quant_storage,
                    )
                else:
                    extra_kwargs = (
                        {"quant_storage": quantization_config.bnb_4bit_quant_storage}
                        if "quant_storage" in list(signature(Linear4bit).parameters)
                        else {}
                    )
                    model._modules[name] = Linear4bit(
                        in_features,
                        out_features,
                        module.bias is not None,
                        quantization_config.bnb_4bit_compute_dtype,
                        compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                        quant_type=quantization_config.bnb_4bit_quant_type,
                        **extra_kwargs,
                    )

                has_been_replaced = True
                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            has_been_replaced = _replace_with_bnb_moe_fused_linear(
                module, modules_to_not_convert, current_key_name, quantization_config, has_been_replaced
            )

        # Remove the last key for recursion
        current_key_name.pop(-1)

    return has_been_replaced


# model is modified in place
def replace_with_bnb_moe_fused_linear(
    model: nn.Module, modules_to_not_convert: Optional[list[str]], quantization_config: BitsAndBytesConfig
) -> None:
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    with init_empty_weights():
        has_been_replaced = _replace_with_bnb_moe_fused_linear(
            model, modules_to_not_convert, [], quantization_config, False
        )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )


def _process_model_before_weight_loading(
    self: Bnb4BitHfQuantizer,
    model: PreTrainedModel,
    device_map: Union[str, dict[str, Any]],
    keep_in_fp32_modules: Optional[list[str]] = None,
    **kwargs,
) -> None:
    self.modules_to_not_convert = self.get_modules_to_not_convert(
        model, self.quantization_config.llm_int8_skip_modules, keep_in_fp32_modules
    )

    # Extend `self.modules_to_not_convert` to keys that are supposed to be offloaded to `cpu` or `disk`
    if isinstance(device_map, dict) and len(device_map) > 1:
        keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]
        self.modules_to_not_convert.extend(keys_on_cpu)

    replace_with_bnb_moe_fused_linear(
        model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
    )

    model.config.quantization_config = self.quantization_config


def patch_bnb_quantizer() -> None:
    Bnb4BitHfQuantizer.param_needs_quantization = _param_needs_quantization
    Bnb4BitHfQuantizer._process_model_before_weight_loading = _process_model_before_weight_loading
