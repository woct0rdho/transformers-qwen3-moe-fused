#!/usr/bin/env python3
#
# Run test_model.py first

import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3MoeModel, set_seed

from qwen3_moe_fused.convert import convert_lora_to_fused, convert_lora_to_unfused
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import (
    Qwen3MoeFusedModel,
    moe_fused_kaiming_uniform_,
)
from test_utils import get_rtol_atol


os.environ["AUTOTUNE_DISABLE"] = "1"


def main():
    patch_lora_config()

    model_dir = "./pretrained/qwen-moe-tiny"
    lora_dir = "./pretrained/qwen-moe-tiny-lora"
    model_fused_dir = "./pretrained/qwen-moe-tiny-fused"
    lora_fused_dir = "./pretrained/qwen-moe-tiny-lora-fused"
    model_roundtrip_dir = "./pretrained/qwen-moe-tiny-roundtrip"
    lora_roundtrip_dir = "./pretrained/qwen-moe-tiny-lora-roundtrip"
    device = "cuda"
    dtype = torch.float32
    set_seed(42)

    vocab_size = 151936
    batch_size = 7
    seq_len = 13

    lora_config = LoraConfig(
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # We can set a smaller rank for MoE layers
        # With rslora, we don't need to set a different alpha for them
        rank_pattern={
            "q_proj": 16,
            "k_proj": 16,
            "v_proj": 16,
            "o_proj": 16,
            "gate": 16,
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        },
        lora_alpha=1,
        use_rslora=True,
    )

    model = Qwen3MoeModel.from_pretrained(model_dir, device_map=device, torch_dtype=dtype)
    model = get_peft_model(model, lora_config)

    # lora_B.weight is inited to zeros. For testing, we make it non-zero
    for name, param in model.named_parameters():
        if name.endswith("lora_B.default.weight"):
            # print("Init", name)
            moe_fused_kaiming_uniform_(param)

    model.save_pretrained(lora_dir)

    convert_lora_to_fused(lora_dir, lora_fused_dir)
    model_fused = Qwen3MoeFusedModel.from_pretrained(model_fused_dir, device_map=device, torch_dtype=dtype)
    model_fused = PeftModel.from_pretrained(model_fused, lora_fused_dir, device_map=device, torch_dtype=dtype)

    convert_lora_to_unfused(lora_fused_dir, lora_roundtrip_dir)
    model_roundtrip = Qwen3MoeModel.from_pretrained(model_roundtrip_dir, device_map=device, torch_dtype=dtype)
    model_roundtrip = PeftModel.from_pretrained(
        model_roundtrip, lora_roundtrip_dir, device_map=device, torch_dtype=dtype
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.int32)
    hidden = model(input_ids=input_ids).last_hidden_state
    hidden_fused = model_fused(input_ids=input_ids).last_hidden_state
    hidden_roundtrip = model_roundtrip(input_ids=input_ids).last_hidden_state
    # print(hidden.shape, hidden.device, hidden.dtype)
    # print(hidden_fused.shape, hidden_fused.device, hidden_fused.dtype)
    # print(hidden_roundtrip.shape, hidden_roundtrip.device, hidden_roundtrip.dtype)
    print(get_rtol_atol(hidden_fused, hidden))
    print(get_rtol_atol(hidden_roundtrip, hidden))


if __name__ == "__main__":
    main()
