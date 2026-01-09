#!/usr/bin/env python3
#
# Run test_lora.py first

import os

import torch
from peft import PeftModel
from transformers import set_seed

from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedModel
from test_model import get_rtol_atol


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def collect_grads(model):
    names = []
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
            params.append(param.grad.clone())
    return names, params


def main():
    patch_lora_config()

    model_dir = "./pretrained/qwen-moe-tiny-fused"
    lora_dir = "./pretrained/qwen-moe-tiny-lora-fused"
    device = "cuda"
    dtype = torch.float32
    set_seed(42)

    batch_size = 7
    seq_len = 13

    model = Qwen3MoeFusedModel.from_pretrained(model_dir, device_map=device, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, lora_dir, is_trainable=True, device_map=device, torch_dtype=dtype)
    hidden_size = model.config.hidden_size

    input = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    input = input.requires_grad_()
    input_fast = input.clone().requires_grad_()
    grad_output = torch.randn_like(input)

    output = model(inputs_embeds=input).last_hidden_state
    output.backward(gradient=grad_output)
    names, grads = collect_grads(model)
    model.zero_grad()

    patch_Qwen3MoeFusedSparseMoeBlock_forward()

    output_fast = model(inputs_embeds=input_fast).last_hidden_state
    output_fast.backward(gradient=grad_output)
    names_fast, grads_fast = collect_grads(model)
    model.zero_grad()

    assert names_fast == names
    print("output", get_rtol_atol(output_fast, output))
    for name, grad, grad_fast in zip(names, grads, grads_fast):
        print(name, get_rtol_atol(grad_fast, grad))


if __name__ == "__main__":
    main()
