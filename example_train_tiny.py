#!/usr/bin/env python3
#
# Example to train a tiny model
# Run example_create_tiny.py first

import os

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    patch_bnb_quantizer()
    patch_lora_config()
    patch_Qwen3MoeFusedSparseMoeBlock_forward()

    model_dir = "./pretrained/qwen-moe-tiny-lm"

    model = Qwen3MoeFusedForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

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
        # We can set a smaller rank for MoE layers,
        # see https://github.com/woct0rdho/transformers-qwen3-moe-fused/issues/3#issuecomment-3144009673
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
    model = get_peft_model(model, lora_config)

    dataset = Dataset.from_dict({"text": [x * 100 for x in "abcdefghijkl"]})

    # These hyperparameters are for exaggerating the training of the tiny model
    # Don't use them in actual training
    sft_config = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-2,
        weight_decay=1e-2,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=3,
        bf16=True,
        optim="adamw_8bit",
        dataset_text_field="text",
        dataset_num_proc=1,
        report_to="none",
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
