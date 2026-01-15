from typing import Any, Optional, Union

import torch
from torch import nn as nn
from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.utils.quantization_config import QuantizationConfigMixin

from ..modular_qwen3_moe_fused import MoeFusedLinear
from .dequant import dequantize
from .layer import GGUFEmbedding, GGUFLinear, GGUFMoeFusedLinear
from .utils import GGUFQuantizedTensor, load_gguf_checkpoint_quantized


class GGUFConfig(QuantizationConfigMixin):
    def __init__(self, **kwargs) -> None:
        kwargs["quant_method"] = "gguf"
        super().__init__(**kwargs)


class GGUFHfQuantizer(HfQuantizer):
    requires_parameters_quantization = True
    requires_calibration = False

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs) -> None:
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs) -> None:
        pass

    def update_torch_dtype(self, dtype: torch.dtype) -> torch.dtype:
        return dtype

    def _process_model_before_weight_loading(self, model: nn.Module, **kwargs) -> None:
        replace_with_gguf_linear(model, modules_to_not_convert=self.modules_to_not_convert)
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: nn.Module, **kwargs) -> nn.Module:
        model.is_quantized = True
        return model

    def param_needs_quantization(self, model: nn.Module, param_name: str, **kwargs) -> bool:
        module, name = get_module_from_name(model, param_name)
        return isinstance(module, (GGUFEmbedding, GGUFLinear, GGUFMoeFusedLinear)) and name == "weight"

    def create_quantized_param(
        self,
        model: nn.Module,
        param_value: Any,
        param_name: str,
        target_device: torch.device,
        **kwargs,
    ) -> None:
        module, name = get_module_from_name(model, param_name)

        if isinstance(param_value, GGUFQuantizedTensor):
            module.tensor_type = param_value.tensor_type
            module.original_shape = param_value.shape
            data_tensor = param_value.data.to(target_device)
            module.register_buffer("weight", data_tensor)
        else:
            new_param = nn.Parameter(
                param_value.to(device=target_device, dtype=model.dtype if hasattr(model, "dtype") else torch.float32),
                requires_grad=False,
            )
            if name in module._parameters:
                module._parameters[name] = new_param
            else:
                module.register_parameter(name, new_param)
            module.tensor_type = None

    def is_serializable(self, safe_serialization: Any = None) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return False


# Does not require patch_load_gguf
def load_gguf_to_model(
    model: nn.Module,
    gguf_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    parsed = load_gguf_checkpoint_quantized(gguf_path, return_tensors=True, model_to_load=model)
    state_dict = parsed["tensors"]

    quantized_names = [k for k, v in state_dict.items() if isinstance(v, GGUFQuantizedTensor)]
    quantized_module_names = [name.rsplit(".", 1)[0] for name in quantized_names]

    if model.config.tie_word_embeddings:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()

        def find_name(target_mod):
            for name, mod in model.named_modules():
                if mod is target_mod:
                    return name
            return None

        input_name = find_name(input_embeddings)
        output_name = find_name(output_embeddings)

        if input_name in quantized_module_names and output_name and output_name not in quantized_module_names:
            quantized_module_names.append(output_name)

    replace_with_gguf_linear(model, quantized_module_names=quantized_module_names, target_dtype=dtype)

    quantizer = GGUFHfQuantizer(GGUFConfig())

    for param_name, param_value in state_dict.items():
        if quantizer.param_needs_quantization(model, param_name):
            quantizer.create_quantized_param(model, param_value, param_name, device)
        else:
            module, tensor_name = get_module_from_name(model, param_name)
            if isinstance(param_value, GGUFQuantizedTensor):
                param_value = param_value.dequantize()

            if tensor_name in module._parameters:
                new_param = nn.Parameter(param_value.to(device=device, dtype=dtype), requires_grad=False)
                module._parameters[tensor_name] = new_param
            else:
                module.register_buffer(tensor_name, param_value.to(device=device, dtype=dtype), persistent=True)

    for name, param in model.named_parameters(recurse=True):
        if param.device.type == "meta":
            module, tensor_name = get_module_from_name(model, name)
            new_param = nn.Parameter(torch.zeros(param.shape, device=device, dtype=dtype), requires_grad=False)
            setattr(module, tensor_name, new_param)
        elif param.device.type != torch.device(device).type:
            module, tensor_name = get_module_from_name(model, name)
            new_param = nn.Parameter(param.to(device=device, dtype=dtype), requires_grad=False)
            setattr(module, tensor_name, new_param)

    for name, buffer in model.named_buffers(recurse=True):
        if "inv_freq" in name:
            module, tensor_name = get_module_from_name(model, name)
            rope_base = getattr(model.config, "rope_theta", 10000.0)
            dim = buffer.shape[0] * 2
            inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2, device=device).float() / dim))
            module.register_buffer(tensor_name, inv_freq, persistent=False)
            continue

        if buffer.device.type == "meta":
            module, tensor_name = get_module_from_name(model, name)
            module.register_buffer(tensor_name, torch.zeros(buffer.shape, device=device, dtype=dtype), persistent=True)
        elif buffer.device.type != torch.device(device).type:
            module, tensor_name = get_module_from_name(model, name)
            module.register_buffer(tensor_name, buffer.to(device=device), persistent=True)

    if model.config.tie_word_embeddings:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()

        is_input_gguf = isinstance(input_embeddings, GGUFEmbedding)
        is_output_gguf = isinstance(output_embeddings, GGUFLinear)

        if is_input_gguf and not is_output_gguf:
            w = dequantize(
                input_embeddings.weight,
                input_embeddings.tensor_type,
                input_embeddings.original_shape,
                device=output_embeddings.weight.device,
                dtype=output_embeddings.weight.dtype,
            )

            if w.shape == output_embeddings.weight.shape:
                pass
            elif w.shape == (output_embeddings.weight.shape[1], output_embeddings.weight.shape[0]):
                w = w.T

            output_embeddings.weight.data = w

        elif is_input_gguf and is_output_gguf:
            # If output_embeddings.weight is None, it means it wasn't loaded from GGUF (because it's tied),
            # so we must link it. If it's not None, check if it points to different memory.
            if (
                output_embeddings.weight is None
                or input_embeddings.weight.data_ptr() != output_embeddings.weight.data_ptr()
            ):
                output_embeddings.weight = input_embeddings.weight
                output_embeddings.tensor_type = input_embeddings.tensor_type
                output_embeddings.original_shape = input_embeddings.original_shape

        elif not is_input_gguf and not is_output_gguf:
            if input_embeddings.weight.data_ptr() != output_embeddings.weight.data_ptr():
                output_embeddings.weight = input_embeddings.weight

    return model


def replace_with_gguf_linear(
    model: nn.Module,
    modules_to_not_convert: Optional[list[str]] = None,
    quantized_module_names: Optional[list[str]] = None,
    prefix: str = "",
    target_dtype: Optional[torch.dtype] = None,
) -> None:
    modules_to_not_convert = modules_to_not_convert or []

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        should_convert = False
        if quantized_module_names is not None:
            if full_name in quantized_module_names:
                should_convert = True
        else:
            if isinstance(module, (nn.Embedding, nn.Linear, MoeFusedLinear)) and name not in modules_to_not_convert:
                should_convert = True

        if should_convert:
            if target_dtype is not None:
                curr_dtype = target_dtype
            else:
                curr_dtype = module.weight.dtype if hasattr(module, "weight") and module.weight is not None else None

            new_module = None
            if isinstance(module, nn.Embedding):
                new_module = GGUFEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    padding_idx=module.padding_idx,
                    device=module.weight.device if module.weight is not None else None,
                    dtype=curr_dtype,
                )
            elif isinstance(module, nn.Linear):
                new_module = GGUFLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device if module.weight is not None else None,
                    dtype=curr_dtype,
                )
            elif isinstance(module, MoeFusedLinear):
                new_module = GGUFMoeFusedLinear(
                    module.in_features,
                    module.out_features,
                    module.num_experts,
                    device=module.weight.device if module.weight is not None else None,
                    dtype=curr_dtype,
                )

            if new_module is not None:
                model._modules[name] = new_module

        replace_with_gguf_linear(
            module,
            modules_to_not_convert,
            quantized_module_names,
            full_name,
            target_dtype=target_dtype,
        )


def patch_load_gguf() -> None:
    from transformers import configuration_utils, modeling_gguf_pytorch_utils
    from transformers.quantizers import auto

    if "gguf" not in auto.AUTO_QUANTIZER_MAPPING:
        auto.AUTO_QUANTIZER_MAPPING["gguf"] = GGUFHfQuantizer

    if "gguf" not in auto.AUTO_QUANTIZATION_CONFIG_MAPPING:
        auto.AUTO_QUANTIZATION_CONFIG_MAPPING["gguf"] = GGUFConfig

    configuration_utils.load_gguf_checkpoint = load_gguf_checkpoint_quantized
    modeling_gguf_pytorch_utils.load_gguf_checkpoint = load_gguf_checkpoint_quantized
