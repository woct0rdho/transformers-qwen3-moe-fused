import re
from typing import Any, NamedTuple, Optional

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, dequantize
from torch import nn
from tqdm import tqdm
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_TO_TRANSFORMERS_MAPPING,
    TENSOR_PROCESSORS,
    TensorProcessor,
    _gguf_parse_value,
    get_gguf_hf_weights_map,
    read_field,
)


class GGUFQuantizedTensor:
    def __init__(self, data: np.ndarray, tensor_type: GGMLQuantizationType, shape: tuple, name: str) -> None:
        self.data = data
        self.tensor_type = tensor_type
        self.shape = shape
        self.name = name

    # Returns float32 numpy array
    def dequantize(self) -> np.ndarray:
        return dequantize(self.data, self.tensor_type).reshape(self.shape)

    def __repr__(self) -> str:
        return f"GGUFQuantizedTensor(name={self.name}, type={self.tensor_type}, shape={self.shape})"


class GGUFProcessResult(NamedTuple):
    weights: GGUFQuantizedTensor
    name: str
    metadata: dict


class GGUFTensorProcessor(TensorProcessor):
    def process(self, weights: GGUFQuantizedTensor, name: str, **kwargs) -> GGUFProcessResult:
        return GGUFProcessResult(weights, name, {})


def load_gguf_checkpoint_quantized(
    gguf_checkpoint_path: str,
    return_tensors: bool = False,
    model_to_load: Optional[nn.Module] = None,
    keep_quantized_tensors: bool = False,
) -> dict[str, Any]:
    """
    Loads a GGUF file and returns a dictionary of parsed parameters containing tensors (some quantized),
    the parsed tokenizer, and config attributes.
    """
    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}

    # Config parsing (copied from transformers)

    architecture = read_field(reader, "general.architecture")[0]
    # Some GGUF files may miss `general.name` field in metadata
    model_name = read_field(reader, "general.name")

    updated_architecture = None
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    elif "t5" in architecture or "t5encoder" in architecture:
        # T5 logic simplified for brevity, assuming standard T5/UMT5 for now
        updated_architecture = "t5"
        if "t5encoder" in architecture:
            parsed_parameters["config"]["architectures"] = ["T5EncoderModel"]
    else:
        updated_architecture = architecture

    if "qwen2moe" in architecture:
        updated_architecture = "qwen2_moe"
    elif "qwen3moe" in architecture:
        updated_architecture = "qwen3_moe"

    if "stablelm" in architecture:
        attn_bias_name = {"attn_q.bias", "attn_k.bias", "attn_v.bias"}
        ffn_norm_name = "ffn_norm"
        qkv_bias = any(bias_name in tensor.name for tensor in reader.tensors for bias_name in attn_bias_name)
        use_parallel_residual = any(ffn_norm_name in tensor.name for tensor in reader.tensors)
        parsed_parameters["config"]["use_qkv_bias"] = qkv_bias
        parsed_parameters["config"]["use_parallel_residual"] = not use_parallel_residual

    # Tie word embeddings
    exceptions = ["falcon", "bloom"]
    parsed_parameters["config"]["tie_word_embeddings"] = (
        all("output.weight" != tensor.name for tensor in reader.tensors) or architecture in exceptions
    )

    # General fields
    for gguf_key, field in reader.fields.items():
        gguf_key = gguf_key.replace(architecture, updated_architecture)
        split = gguf_key.split(".")
        prefix = split[0]
        config_key = ".".join(split[1:])

        value = [_gguf_parse_value(field.parts[_data_index], field.types) for _data_index in field.data]
        if len(value) == 1:
            value = value[0]
        if isinstance(value, str) and architecture in value:
            value = value.replace(architecture, updated_architecture)

        for parameter, parameter_renames in GGUF_TO_TRANSFORMERS_MAPPING.items():
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames[prefix][config_key]
                if renamed_config_key == -1:
                    continue
                if renamed_config_key is not None:
                    parsed_parameters[parameter][renamed_config_key] = value
                if gguf_key in reader_keys:
                    reader_keys.remove(gguf_key)

    # Vocab size fallback
    if "vocab_size" not in parsed_parameters["config"] and "tokens" in parsed_parameters["tokenizer"]:
        parsed_parameters["config"]["vocab_size"] = len(parsed_parameters["tokenizer"]["tokens"])

    if return_tensors:
        parsed_parameters["tensors"] = {}
        tensor_key_mapping = get_gguf_hf_weights_map(model_to_load)

        # Patch mapping for fused MoE
        if updated_architecture in ("qwen2_moe", "qwen3_moe") and model_to_load is not None:
            for name, module in model_to_load.named_modules():
                if (
                    hasattr(module, "gate_proj")
                    and hasattr(module, "up_proj")
                    and hasattr(module, "down_proj")
                    and hasattr(module, "num_experts")
                ):
                    # Check if it is the fused block by checking type of gate_proj
                    if "MoeFusedLinear" in str(type(module.gate_proj)):
                        match = re.search(r"layers\.(\d+)\.mlp", name)
                        if match:
                            layer_idx = match.group(1)
                            tensor_key_mapping[f"blk.{layer_idx}.ffn_gate_exps.weight"] = f"{name}.gate_proj.weight"
                            tensor_key_mapping[f"blk.{layer_idx}.ffn_up_exps.weight"] = f"{name}.up_proj.weight"
                            tensor_key_mapping[f"blk.{layer_idx}.ffn_down_exps.weight"] = f"{name}.down_proj.weight"

        config = parsed_parameters.get("config", {})

        # Use original processor logic only when dequantizing
        OriginalProcessorClass = TENSOR_PROCESSORS.get(architecture, TensorProcessor)
        if model_to_load is not None and "Qwen3MoeFused" in model_to_load.__class__.__name__:
            OriginalProcessorClass = TensorProcessor

        original_processor = OriginalProcessorClass(config=config)
        # Custom processor that passes through, effectively disabling splitting/permuting for quantized
        pass_through_processor = GGUFTensorProcessor(config=config)

        for tensor in tqdm(reader.tensors, desc="Loading GGUF tensors..."):
            name = tensor.name

            keep_quantized = False
            if keep_quantized_tensors:
                if "weight" in name and "norm" not in name and "bias" not in name:
                    keep_quantized = True
                if "norm" in name or "bias" in name:
                    keep_quantized = False

            if keep_quantized:
                raw_data = np.copy(tensor.data)

                # GGUF stores shape (dim0, ..., dimN), where dim0 is fastest varying,
                # while PyTorch and NumPy expect (dimN, ..., dim0)
                torch_shape = tuple(tensor.shape)[::-1]

                weights = GGUFQuantizedTensor(raw_data, tensor.tensor_type, torch_shape, name)
                result = pass_through_processor.process(
                    weights=weights,
                    name=name,
                    tensor_key_mapping=tensor_key_mapping,
                    parsed_parameters=parsed_parameters,
                )
            else:
                # Dequantize immediately
                weights = dequantize(tensor.data, tensor.tensor_type)
                result = original_processor.process(
                    weights=weights,
                    name=name,
                    tensor_key_mapping=tensor_key_mapping,
                    parsed_parameters=parsed_parameters,
                )

            weights = result.weights
            name = result.name

            if name is None:
                # Processor might have stored it elsewhere (e.g. split experts)
                continue

            if name not in tensor_key_mapping:
                continue

            hf_name = tensor_key_mapping[name]

            if isinstance(weights, GGUFQuantizedTensor):
                # Store GGUFQuantizedTensor directly
                parsed_parameters["tensors"][hf_name] = weights
            else:
                # Convert to torch tensor
                parsed_parameters["tensors"][hf_name] = torch.from_numpy(np.copy(weights))

    return parsed_parameters
