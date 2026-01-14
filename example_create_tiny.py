#!/usr/bin/env python3
#
# Randomly initialize a tiny model and its quantized version
# Then it can be trained in example_train_tiny.py

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3MoeConfig

from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


def main():
    patch_bnb_quantizer()

    model_dir = "./pretrained/qwen-moe-tiny-lm"
    model_quantized_dir = "./pretrained/qwen-moe-tiny-lm-quantized"

    # Create the model
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=5,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_window_layers=2,
        moe_intermediate_size=3,
        num_experts=16,
        norm_topk_prob=True,
    )
    model = Qwen3MoeFusedForCausalLM(config)
    model.save_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.save_pretrained(model_dir)

    # Load and quantize the model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3MoeFusedForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config)
    model.save_pretrained(model_quantized_dir)

    tokenizer.save_pretrained(model_quantized_dir)


if __name__ == "__main__":
    main()
