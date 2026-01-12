#!/usr/bin/env python3
#
# Example to inference the GGUF version of Qwen3-30B-A3B with fused MoE and on-demand dequant

import os

import torch
from transformers import AutoConfig, AutoTokenizer

from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize_gguf.quantizer import (
    load_gguf_to_model,
    patch_load_gguf,
)


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    patch_load_gguf()

    gguf_path = r"C:\models\Qwen3-30B-A3B-UD-IQ1_S.gguf"
    device = "cuda"
    dtype = torch.bfloat16

    model_dir = os.path.dirname(gguf_path)
    gguf_file = os.path.basename(gguf_path)

    config = AutoConfig.from_pretrained(model_dir, gguf_file=gguf_file)
    config.dtype = dtype
    with torch.device("meta"):
        model = Qwen3MoeFusedForCausalLM(config)
    model = load_gguf_to_model(model, gguf_path, device=device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    # Modified from https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/README.md
    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=10)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids)
    print(content)


if __name__ == "__main__":
    main()
