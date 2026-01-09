import os

import torch
from transformers import AutoModelForCausalLM

from qwen3_moe_fused.quantize_gguf.quantizer import (
    load_gguf_to_model,
    patch_load_gguf,
)


def print_vram(label):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{label}] VRAM allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")


def main():
    patch_load_gguf()

    gguf_path = r"C:\models\Qwen3-0.6B-UD-IQ1_S.gguf"
    device = "cuda"
    dtype = torch.bfloat16

    model_dir = os.path.dirname(gguf_path)
    gguf_file = os.path.basename(gguf_path)

    print_vram("Initial state")

    print("Initializing model skeleton...")
    # Use device_map="meta" to load only the model skeleton
    model = AutoModelForCausalLM.from_pretrained(model_dir, gguf_file=gguf_file, device_map="meta", torch_dtype=dtype)

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
