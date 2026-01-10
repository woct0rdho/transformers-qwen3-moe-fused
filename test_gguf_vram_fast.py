#!/usr/bin/env python3

import os

import torch
from transformers import AutoConfig

from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize_gguf.quantizer import (
    load_gguf_to_model,
    patch_load_gguf,
)
from test_gguf_vram import print_vram


def main():
    patch_load_gguf()

    gguf_path = r"C:\models\Qwen3-30B-A3B-UD-IQ1_S.gguf"
    device = "cuda"
    dtype = torch.bfloat16

    model_dir = os.path.dirname(gguf_path)
    gguf_file = os.path.basename(gguf_path)

    print_vram("Initial state")

    print("Initializing model skeleton...")
    config = AutoConfig.from_pretrained(model_dir, gguf_file=gguf_file)
    config.dtype = dtype
    with torch.device("meta"):
        model = Qwen3MoeFusedForCausalLM(config)

    print_vram("After initializing skeleton")

    print("Loading GGUF weights (with on-demand dequantization)...")
    model = load_gguf_to_model(model, gguf_path, device=device, dtype=dtype)

    print_vram("After loading weights")

    print("Running forward pass...")
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.inference_mode():
        output = model(input_ids).logits
    print(f"Output shape: {output.shape}")

    print_vram("After forward pass")


if __name__ == "__main__":
    main()
