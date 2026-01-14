#!/usr/bin/env python3
#
# Example to inference the fused and quantized version of Qwen3-30B-A3B with a LoRA
# You can replace lora_id with the local path of your trained LoRA

import os

from peft import PeftModel
from transformers import AutoTokenizer

from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import (
    Qwen3MoeFusedForCausalLM,
    patch_Qwen3MoeSparseMoeBlock_init,
)
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    patch_Qwen3MoeSparseMoeBlock_init()
    patch_bnb_quantizer()
    patch_lora_config()

    model_id = "bash99/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit"
    lora_id = "woctordho/Qwen3-30B-A3B-abliterated-lora-fused"

    model = Qwen3MoeFusedForCausalLM.from_pretrained(model_id)
    model = PeftModel.from_pretrained(model, lora_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
