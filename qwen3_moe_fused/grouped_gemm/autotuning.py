import os
from itertools import product
from typing import Any, Callable

import torch
import triton


if torch.version.hip:
    DEFAULT_M_BLOCK_SIZES = [32, 64, 128]
    DEFAULT_N_BLOCK_SIZES = [32, 64, 128]
    DEFAULT_K_BLOCK_SIZES = [32, 64, 128]
    DEFAULT_NUM_WARPS = [4, 8]
    DEFAULT_NUM_STAGES = [1, 2, 3]
    # Strix Halo and RDNA benefit significantly from having multiple active blocks per CU
    GRID_FACTOR = 64
else:
    DEFAULT_M_BLOCK_SIZES = [16, 32, 64, 128, 256]
    DEFAULT_N_BLOCK_SIZES = [16, 32, 64, 128, 256]
    DEFAULT_K_BLOCK_SIZES = [16, 32, 64, 128, 256]
    DEFAULT_NUM_WARPS = [4, 8]
    DEFAULT_NUM_STAGES = [3, 4, 5, 6]
    GRID_FACTOR = 1


def get_num_sms() -> int:
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def get_autotune_configs() -> list[triton.Config]:
    configs = []
    for m, n, k, w, s in product(
        DEFAULT_M_BLOCK_SIZES,
        DEFAULT_N_BLOCK_SIZES,
        DEFAULT_K_BLOCK_SIZES,
        DEFAULT_NUM_WARPS,
        DEFAULT_NUM_STAGES,
    ):
        configs.append(
            triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k}, num_warps=w, num_stages=s)
        )
    return configs


def _get_device_properties() -> dict[str, Any]:
    return triton.runtime.driver.active.utils.get_device_properties(torch.cuda.current_device())


def _common_prune_criteria(smem_criteria: Callable, config: triton.Config, kwargs: dict[str, Any]) -> bool:
    num_stages = config.num_stages
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]
    dtype = kwargs["x_ptr"].dtype
    device_properties = _get_device_properties()
    smem_size = device_properties["max_shared_mem"]
    if smem_criteria(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, smem_size):
        return True

    M = kwargs["M"]
    N = kwargs["N"]
    K = kwargs["K"]
    num_experts = kwargs["NUM_EXPERTS"]
    tokens_per_expert = M // num_experts
    max_block_size_M = max(tokens_per_expert * 2, DEFAULT_M_BLOCK_SIZES[0])
    max_block_size_N = max(N, DEFAULT_N_BLOCK_SIZES[0])
    max_block_size_K = max(K, DEFAULT_K_BLOCK_SIZES[0])
    if BLOCK_SIZE_M > max_block_size_M:
        return True
    if BLOCK_SIZE_N > max_block_size_N:
        return True
    if BLOCK_SIZE_K > max_block_size_K:
        return True
    if BLOCK_SIZE_N * BLOCK_SIZE_K >= 256 * 256:
        # Almost always useless and makes autotune much slower
        return True

    min_block_size_M = min(triton.next_power_of_2(tokens_per_expert // 2 + 1), 16)
    min_block_size_N = min(triton.next_power_of_2(N // 2 + 1), 64)
    min_block_size_K = min(triton.next_power_of_2(K // 2 + 1), 64)
    if BLOCK_SIZE_M * BLOCK_SIZE_N < min_block_size_M * min_block_size_N:
        return True
    if BLOCK_SIZE_M * BLOCK_SIZE_K < min_block_size_M * min_block_size_K:
        return True
    if BLOCK_SIZE_N * BLOCK_SIZE_K < min_block_size_N * min_block_size_K:
        return True

    return False


def prune_configs(smem_criteria: Callable, configs: list[triton.Config], args, **kwargs) -> list[triton.Config]:
    pruned_configs = []
    for config in configs:
        if _common_prune_criteria(smem_criteria, config, args):
            continue
        pruned_configs.append(config)
    return pruned_configs


# We need to autotune on batch size only when benchmarking with a large range of batch sizes
def get_autotune_keys() -> list[str]:
    if os.getenv("AUTOTUNE_BATCH_SIZE", "0") == "1":
        return ["M", "N", "K", "NUM_EXPERTS"]
    else:
        return ["N", "K", "NUM_EXPERTS"]
