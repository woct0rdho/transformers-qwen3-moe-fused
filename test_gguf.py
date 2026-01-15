#!/usr/bin/env python3

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3ForCausalLM

from qwen3_moe_fused.quantize_gguf.quantizer import load_gguf_to_model
from test_utils import get_rtol_atol


def main():
    gguf_path = r"C:\models\Qwen3-0.6B-UD-IQ1_S.gguf"
    device = "cuda"
    dtype = torch.bfloat16

    model_dir = os.path.dirname(gguf_path)
    gguf_file = os.path.basename(gguf_path)

    print("Loading reference model...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_dir, gguf_file=gguf_file, device_map=device, torch_dtype=dtype
    )

    print("Running forward pass...")
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.inference_mode():
        out_ref = model_ref(input_ids).logits

    del model_ref

    print("Initializing model skeleton...")
    config = AutoConfig.from_pretrained(model_dir, gguf_file=gguf_file)
    config.dtype = dtype
    with torch.device("meta"):
        model_test = Qwen3ForCausalLM(config)

    print("Loading GGUF weights (with on-demand dequantization)...")
    model_test = load_gguf_to_model(model_test, gguf_path, device=device, dtype=dtype)

    print("Running forward pass...")
    with torch.inference_mode():
        out_test = model_test(input_ids).logits

    print("Comparison results:")
    print(get_rtol_atol(out_test, out_ref))


if __name__ == "__main__":
    main()
