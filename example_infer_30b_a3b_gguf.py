#!/usr/bin/env python3
#
# Example to inference the GGUF version of Qwen3-30B-A3B with a LoRA, with fused MoE and on-demand dequant
# You can replace lora_id with the local path of your trained LoRA

import os

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer

from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import (
    Qwen3MoeFusedForCausalLM,
    patch_Qwen3MoeSparseMoeBlock_init,
)
from qwen3_moe_fused.quantize_gguf.quantizer import load_gguf_to_model


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    patch_Qwen3MoeSparseMoeBlock_init()
    patch_lora_config()

    gguf_path = r"C:\models\Qwen3-30B-A3B-Instruct-2507-UD-IQ3_XXS.gguf"
    lora_id = "woctordho/Qwen3-30B-A3B-abliterated-lora-fused"
    device = "cuda"
    dtype = torch.bfloat16

    model_dir = os.path.dirname(gguf_path)
    gguf_file = os.path.basename(gguf_path)

    # TODO: Support loading GGUF using AutoModel.from_pretrained
    config = AutoConfig.from_pretrained(model_dir, gguf_file=gguf_file)
    config.dtype = dtype
    with torch.device("meta"):
        model = Qwen3MoeFusedForCausalLM(config)
    model = load_gguf_to_model(model, gguf_path, device=device, dtype=dtype)
    model = PeftModel.from_pretrained(model, lora_id)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")

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
