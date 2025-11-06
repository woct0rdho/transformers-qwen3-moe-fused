# Qwen3 MoE Fused

**Update:** Transformers 5 will soon be released and it supports the fused MoE kernels. We will implement the LoRA in PEFT. This repo only supports Transformers < 4.57 .

The Qwen3 MoE model (and all other MoE models) in HF Transformers is notoriously slow, because it uses a [for loop](https://github.com/huggingface/transformers/blob/bdf5fb70aa11782cce22027d76879f71f4e41c1e/src/transformers/models/qwen3_moe/modular_qwen3_moe.py#L103) to access the experts. The purpose of this repo is to fine-tune Qwen3-30B-A3B on a single GPU with 24GB VRAM and achieve high throughput. The implementation is compatible with the HF Transformers ecosystem, such as LoRA, bitsandbytes 4-bit quantization, and Unsloth. See [`example_train_30b_a3b_unsloth.py`](https://github.com/woct0rdho/transformers-qwen3-moe-fused/blob/master/example_train_30b_a3b_unsloth.py) for the usage.

## Fused linear layer

The critical part is to implement the [`moe_fused_linear`](https://github.com/woct0rdho/transformers-qwen3-moe-fused/blob/master/qwen3_moe_fused/functional.py) function:
```
output[b, o] = sum_i weight[selected_experts[b], o, i] * input[b, i]
```
There are already several good implementations, such as [triton-kernels](https://github.com/triton-lang/triton/blob/dd1c3d429d1c24904722ac699ea5750bc694c4d6/python/triton_kernels/triton_kernels/matmul_ogs.py), [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/a0535ffa0d35fccfec3e1a0a3bfc9dbb6054d7c0/ggml/src/ggml-cuda/ggml-cuda.cu#L2065), [vLLM](https://github.com/vllm-project/vllm/blob/015fab8c2fa4db8776f7e91abd50371911673d88/vllm/model_executor/layers/fused_moe/fused_moe.py), [fanshiqing/grouped_gemm](https://github.com/fanshiqing/grouped_gemm). `torch._grouped_mm` is also being implemented. We need to sort `input` by the experts to improve the memory coalescence of `weight`, and more optimizations are explained in https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/

The implementation in this repo is largely based on the MoE kernel in [Unsloth](https://github.com/unslothai/unsloth/blob/2bfc39b6387577457834059c59f83fcdb954c9bd/unsloth/kernels/moe), which is based on the Triton [grouped GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html). I've added strides, masks, and autotune configs for small or 'thin' matrices, which are needed for LoRA.

I aim to keep the code readable and easy to follow. I only used the most mature features of Triton, such as load and store, rather than things like TMA and swizzle. I've benchmarked it on RTX 3080 and it's close to the theoretical fp16 and bf16 performance.

### LoRA

The LoRA for the fused linear layer is defined by first creating a LoRA for the linear layer in each expert, then stack them along the experts dimension. For the weight tensor with shape `(num_experts, out_features, in_features)`, the two LoRA weights have shape `lora_A: (num_experts, lora_rank, in_features), lora_B: (num_experts, out_features, lora_rank)`. Therefore, we can losslessly convert between the fused and the unfused formats, and a previously trained LoRA can continue to be trained.

The functions in [`qwen3_moe_fused/convert.py`](https://github.com/woct0rdho/transformers-qwen3-moe-fused/blob/master/qwen3_moe_fused/convert.py) can convert a model or a LoRA between the fused and the unfused formats. After you train a LoRA in the fused format, you can convert it to the unfused format, then merge it into the base model, or convert it to other formats such as GGUF. llama.cpp and vLLM already support this kind of LoRA.

### TODO

* This should work with Qwen3-Next with minimal modification. I haven't started trying this, but feel free to ask if you need it.
* Multi-GPU support. I don't have multiple GPUs at home so I'm not focusing on this. It's straightforward to do DDP using HF Accelerate, see https://github.com/woct0rdho/transformers-qwen3-moe-fused/issues/1#issuecomment-3243600437 . FSDP may also work, but expert parallel is out of the scope of this repo. If you use Unsloth, you can follow https://docs.unsloth.ai/basics/multi-gpu-training-with-unsloth . Feel free to ask if you see any error.
* Fuse 4-bit dequant and MoE linear, see [`qwen3_moe_fused/quantize/layer.py`](https://github.com/woct0rdho/transformers-qwen3-moe-fused/blob/master/qwen3_moe_fused/quantize/layer.py). Currently I've written a kernel in [`qwen3_moe_fused/grouped_gemm/forward_4bit.py`](https://github.com/woct0rdho/transformers-qwen3-moe-fused/blob/master/qwen3_moe_fused/grouped_gemm/forward_4bit.py) but it's slower than the unfused version when the batch size is large.

### License

The files in `qwen3_moe_fused/grouped_gemm/` are modified from the Unsloth MoE kernels so they are AGPLv3 licensed, see the [explanation](https://github.com/unslothai/unsloth/discussions/2890#discussioncomment-13675890). For more robust and performant integration, it's possible to use the MIT licensed [triton-kernels](https://github.com/triton-lang/triton/tree/main/python/triton_kernels/triton_kernels) as an alternative.

The rest of this repo, including files modified from Transformers, PEFT, and bitsandbytes, are Apache-2.0 licensed.
