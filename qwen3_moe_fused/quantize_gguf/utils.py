import re
from typing import Any, NamedTuple, Optional

import torch
from gguf import GGMLQuantizationType, GGUFReader
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
from transformers.utils.logging import get_logger

from .dequant import dequantize


logger = get_logger(__name__)


class GGUFQuantizedTensor:
    def __init__(
        self, data: torch.Tensor, tensor_type: GGMLQuantizationType, shape: tuple[int, ...], name: str
    ) -> None:
        self.data = data
        self.tensor_type = tensor_type
        self.shape = shape
        self.name = name

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return dequantize(self.data, self.tensor_type, self.shape, dtype)

    def to(self, *args, **kwargs) -> "GGUFQuantizedTensor":
        device_to_move = None

        if len(args) > 0:
            arg0 = args[0]
            if isinstance(arg0, (torch.device, str, int)):
                device_to_move = arg0
            elif isinstance(arg0, torch.Tensor):
                device_to_move = arg0.device

        if "device" in kwargs:
            device_to_move = kwargs["device"]

        new_data = self.data
        if device_to_move is not None:
            new_data = self.data.to(device_to_move)

        # We ignore dtype casting for the quantized tensor wrapper

        if new_data is self.data:
            return self
        return GGUFQuantizedTensor(new_data, self.tensor_type, self.shape, self.name)

    def contiguous(self, memory_format=torch.contiguous_format) -> "GGUFQuantizedTensor":
        return GGUFQuantizedTensor(
            self.data.contiguous(memory_format=memory_format), self.tensor_type, self.shape, self.name
        )

    def is_contiguous(self, memory_format=torch.contiguous_format) -> bool:
        return self.data.is_contiguous(memory_format=memory_format)

    def __getitem__(self, idx) -> "GGUFQuantizedTensor":
        if idx is Ellipsis or idx == slice(None):
            return self
        raise NotImplementedError(f"Slicing GGUFQuantizedTensor is not supported: {idx}")

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device

    def __repr__(self) -> str:
        return (
            f"GGUFQuantizedTensor(name={self.name}, type={self.tensor_type}, shape={self.shape}, device={self.device})"
        )


class GGUFProcessResult(NamedTuple):
    weights: GGUFQuantizedTensor
    name: str
    metadata: dict


class GGUFTensorProcessor(TensorProcessor):
    def process(self, weights: GGUFQuantizedTensor, name: str, **kwargs) -> GGUFProcessResult:
        return GGUFProcessResult(weights, name, {})


# Modified from https://github.com/huggingface/transformers/blob/2ccc6cae21faaf11631efa5fb9054687ae5dc931/src/transformers/modeling_gguf_pytorch_utils.py#L363
def load_gguf_checkpoint_quantized(
    gguf_checkpoint_path: str, return_tensors: bool = False, model_to_load: Optional[nn.Module] = None
) -> dict[str, Any]:
    """
    Loads a GGUF file and returns a dictionary of parsed parameters containing tensors (some quantized),
    the parsed tokenizer, and config attributes.
    """
    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}

    architecture = read_field(reader, "general.architecture")[0]
    # NOTE: Some GGUF checkpoints may miss `general.name` field in metadata
    model_name = read_field(reader, "general.name")

    updated_architecture = None
    # in llama.cpp mistral models use the same architecture as llama. We need
    # to add this patch to ensure things work correctly on our side.
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    # FIXME: Currently this implementation is only for flan-t5 architecture.
    # It needs to be developed for supporting legacy t5.
    elif "t5" in architecture or "t5encoder" in architecture:
        parsed_parameters["config"]["is_gated_act"] = True
        if model_name and "umt5" in model_name[0].lower():
            updated_architecture = "umt5"
            if "t5encoder" in architecture:
                parsed_parameters["config"]["architectures"] = ["UMT5EncoderModel"]
        else:
            if "t5encoder" in architecture:
                parsed_parameters["config"]["architectures"] = ["T5EncoderModel"]
            updated_architecture = "t5"
    else:
        updated_architecture = architecture

    if "qwen2moe" in architecture:
        updated_architecture = "qwen2_moe"
    elif "qwen3moe" in architecture:
        updated_architecture = "qwen3_moe"

    # For stablelm architecture, we need to set qkv_bias and use_parallel_residual from tensors
    # If `qkv_bias=True`, qkv_proj with bias will be present in the tensors
    # If `use_parallel_residual=False`, ffn_norm will be present in the tensors
    if "stablelm" in architecture:
        attn_bias_name = {"attn_q.bias", "attn_k.bias", "attn_v.bias"}
        ffn_norm_name = "ffn_norm"
        qkv_bias = any(bias_name in tensor.name for tensor in reader.tensors for bias_name in attn_bias_name)
        use_parallel_residual = any(ffn_norm_name in tensor.name for tensor in reader.tensors)
        parsed_parameters["config"]["use_qkv_bias"] = qkv_bias
        parsed_parameters["config"]["use_parallel_residual"] = not use_parallel_residual

    # Handle tie_word_embeddings, if lm_head.weight is not present in tensors,
    # tie_word_embeddings is true otherwise false
    exceptions = ["falcon", "bloom"]
    parsed_parameters["config"]["tie_word_embeddings"] = (
        all("output.weight" != tensor.name for tensor in reader.tensors) or architecture in exceptions
    )

    # List all key-value pairs in a columnized format
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

        if gguf_key in reader_keys:
            logger.info(f"Some keys were not parsed and added into account {gguf_key} | {value}")

    # Gemma3 GGUF checkpoint only contains weights of text backbone
    if parsed_parameters["config"]["model_type"] == "gemma3":
        parsed_parameters["config"]["model_type"] = "gemma3_text"

    if parsed_parameters["config"]["model_type"] == "lfm2":
        gguf_num_key_value_heads = parsed_parameters["config"]["num_key_value_heads"]
        # LFM2 GGUF checkpoint defines num_key_value_heads as a list of integers .e.g [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0] but we need to set it to the max value for HF
        parsed_parameters["config"]["num_key_value_heads"] = max(gguf_num_key_value_heads)
        ## we already read the correct intermediate_size from the GGUF checkpoint so we need to set block_auto_adjust_ff_dim to False
        parsed_parameters["config"]["block_auto_adjust_ff_dim"] = False

        ## llama.cpp defines the layers that are full-attention by looking at num_key_value_heads
        ## we need to set the full_attn_idxs to the layers that are full-attention
        parsed_parameters["config"]["full_attn_idxs"] = [
            i for i, num_kv_heads in enumerate(gguf_num_key_value_heads) if num_kv_heads > 0
        ]

    # retrieve config vocab_size from tokenizer
    # Please refer to https://github.com/huggingface/transformers/issues/32526 for more details
    if "vocab_size" not in parsed_parameters["config"]:
        tokenizer_parameters = parsed_parameters["tokenizer"]
        if "tokens" in tokenizer_parameters:
            parsed_parameters["config"]["vocab_size"] = len(tokenizer_parameters["tokens"])
        else:
            logger.warning(
                "Can't find a way to retrieve missing config vocab_size from tokenizer parameters. "
                "This will use default value from model config class and cause unexpected behavior."
            )

    # Inject quantization config to trigger GGUFHfQuantizer
    parsed_parameters["config"]["quantization_config"] = {"quant_method": "gguf"}

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
            data = torch.from_numpy(tensor.data.copy())

            # GGUF stores shape (dim0, ..., dimN), where dim0 is fastest varying,
            # while PyTorch and NumPy expect (dimN, ..., dim0)
            shape = tuple(tensor.shape)[::-1]

            if "weight" in name and "norm" not in name and "bias" not in name:
                # Dequantize on demand
                weights = GGUFQuantizedTensor(data, tensor.tensor_type, shape, name)
                processor = pass_through_processor
            else:
                # Dequantize immediately
                weights = dequantize(data, tensor.tensor_type, shape)
                processor = original_processor

            result = processor.process(
                weights=weights,
                name=name,
                tensor_key_mapping=tensor_key_mapping,
                parsed_parameters=parsed_parameters,
            )

            # Now weights is either GGUFQuantizedTensor or torch.Tensor
            weights = result.weights
            name = result.name

            if name is None:
                # Processor might have stored it elsewhere (e.g. split experts)
                continue

            if name not in tensor_key_mapping:
                continue

            name = tensor_key_mapping[name]

            parsed_parameters["tensors"][name] = weights

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters
