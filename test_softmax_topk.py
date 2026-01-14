#!/usr/bin/env python3

import os

import torch

from qwen3_moe_fused.kernels.softmax_topk import softmax_topk, softmax_topk_naive


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    device = "cuda"

    configs = [
        (1024, 128, 8, True),
    ]
    for M, N, k, norm in configs:
        print(M, N, k, norm)
        logits = torch.randn(M, N, device=device, dtype=torch.bfloat16, requires_grad=True)

        ref_weights, ref_indices = softmax_topk_naive(logits, k, norm)
        ref_indices = ref_indices.to(torch.int32)
        print("ref_weights", ref_weights.shape, ref_weights.dtype)
        print("ref_indices", ref_indices.shape, ref_indices.dtype)

        out_weights, out_indices = softmax_topk(logits, k, norm)
        print("out_weights", out_weights.shape, out_weights.dtype)
        print("out_indices", out_indices.shape, out_indices.dtype)

        torch.testing.assert_close(out_weights, ref_weights, rtol=1e-2, atol=1e-2)
        # We cannot directly compare indices,
        # because different entries may have the same weight within floating point error
        for i in range(M):
            torch.testing.assert_close(logits[i][out_indices[i]], logits[i][ref_indices[i]], rtol=1e-2, atol=1e-2)

        grad_weights = torch.randn_like(out_weights)

        logits.grad = None
        ref_weights, _ = softmax_topk_naive(logits, k, norm)
        ref_weights.backward(grad_weights)
        ref_grad_logits = logits.grad.clone()
        print("ref_grad_logits", ref_grad_logits.shape, ref_grad_logits.dtype)

        logits.grad = None
        out_weights, _ = softmax_topk(logits, k, norm)
        out_weights.backward(grad_weights)
        out_grad_logits = logits.grad.clone()
        print("out_grad_logits", out_grad_logits.shape, out_grad_logits.dtype)

        torch.testing.assert_close(out_grad_logits, ref_grad_logits, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
