#!/usr/bin/env python3
#
# Example to train a LoRA on the fused and quantized version of Qwen3-30B-A3B using Unsloth
# 24 GB VRAM is recommended
#
# Important: We cache autotuned Triton kernels by default. If you did some small-scale tests, then you should
# clear the Triton cache and the TorchInductor cache before the actual training

import os

from unsloth import FastModel

# Import unsloth before others
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from qwen3_moe_fused.compile_utils import compile_layers
from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM, patch_Qwen3MoeSparseMoeBlock_init
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    patch_Qwen3MoeSparseMoeBlock_init()
    patch_bnb_quantizer()
    patch_lora_config()
    patch_Qwen3MoeFusedSparseMoeBlock_forward()

    model_id = "bash99/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit"

    model, tokenizer = FastModel.from_pretrained(model_id, auto_model=Qwen3MoeFusedForCausalLM)

    model = FastModel.get_peft_model(
        model,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # We can set a smaller rank for MoE layers,
        # see https://github.com/woct0rdho/transformers-qwen3-moe-fused/issues/3#issuecomment-3144009673
        # With rslora, we don't need to set a different alpha for them
        # It's possible to create a LoRA on the routing gate, but this may make the training unstable
        rank_pattern={
            "q_proj": 16,
            "k_proj": 16,
            "v_proj": 16,
            "o_proj": 16,
            # "gate": 16,
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        },
        lora_alpha=1,
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    compile_layers(model)

    dataset = load_dataset("stanfordnlp/imdb", split="train")

    sft_config = SFTConfig(
        per_device_train_batch_size=1,  # Increase batch size if you have more memory
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        # For MoE models, weight decay can be smaller than for dense models,
        # because not every expert has gradient in every step, but weight decay is applied to every expert
        weight_decay=1e-3,
        num_train_epochs=1,
        lr_scheduler_type="linear",
        warmup_steps=1000,
        logging_steps=1,
        save_steps=100,
        save_total_limit=5,
        bf16=True,
        optim="adamw_8bit",
        dataset_text_field="text",
        dataset_num_proc=1,
        torch_compile=True,
        torch_compile_mode="max-autotune",
        report_to="none",  # You may report to Wandb
        seed=3407,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    trainer_stats = trainer.train()
    print("trainer_stats")
    print(trainer_stats)


if __name__ == "__main__":
    main()
