# Currently this is only optimized for Strix Halo

import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import _import_module_from_library, load


# Change the include paths on your machine
CK_ROOT = os.path.expanduser("~/rocm-libraries/projects/composablekernel")
CK_INCLUDES = [
    os.path.join(os.path.dirname(__file__), "include"),
    os.path.join(CK_ROOT, "include"),
    # os.path.join(CK_ROOT, "library", "include"),
    os.path.expanduser("~/venv_torch/lib/python3.13/site-packages/_rocm_sdk_core/include"),
]


def get_rocm_lib_dirs() -> list[str]:
    rocm_lib_dirs = []
    for env_var in ("ROCM_HOME", "ROCM_PATH"):
        rocm_home = os.environ.get(env_var)
        if rocm_home:
            rocm_lib_dirs.append(os.path.join(rocm_home, "lib"))
            rocm_lib_dirs.append(os.path.join(rocm_home, "lib64"))
    for mod_name in ("_rocm_sdk_devel", "_rocm_sdk_core"):
        try:
            mod = __import__(mod_name)
            mod_dir = os.path.dirname(mod.__file__)
            rocm_lib_dirs.append(os.path.join(mod_dir, "lib"))
        except Exception:
            continue
    return [d for d in rocm_lib_dirs if os.path.isdir(d)]


def load_ck_extension(name: str, extra_cuda_cflags: Optional[list[str]] = None):
    print(f"Loading CK kernel {name}...")
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(cur_dir, "build", name)
    os.makedirs(build_dir, exist_ok=True)

    source_file = os.path.join(cur_dir, "ck_grouped_gemm.cu")
    ninja_log = os.path.join(build_dir, ".ninja_log")
    should_rebuild = False

    if os.path.exists(source_file) and os.path.exists(ninja_log):
        if os.path.getmtime(source_file) > os.path.getmtime(ninja_log):
            print(f"Source {os.path.basename(source_file)} is newer than build cache, rebuilding...")
            should_rebuild = True

    if not should_rebuild:
        try:
            module = _import_module_from_library(name, build_dir, is_python_module=True)
            return module
        except ImportError:
            pass

    print(f"Compiling CK kernel {name}...")
    source_files = [os.path.join(cur_dir, "ck_grouped_gemm.cu")]

    cflags = ["-O3", "-std=c++20"]
    cuda_cflags = [
        "-O3",
        "-std=c++20",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_HALF2_OPERATORS__",
        # "-D__HIP_PLATFORM_AMD__=1",
        # "-DUSE_ROCM=1",
        # "-DHIPBLAS_V2=1",
        # "-DCUDA_HAS_FP16=1",
        # "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
        # "--offload-arch=gfx1151",
    ]
    if extra_cuda_cflags:
        cuda_cflags.extend(extra_cuda_cflags)

    ldflags = []
    for lib_dir in dict.fromkeys(get_rocm_lib_dirs()):
        ldflags.extend([f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"])

    module = load(
        name=name,
        sources=source_files,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=ldflags,
        extra_include_paths=CK_INCLUDES,
        build_directory=build_dir,
        verbose=False,
        with_cuda=True,
    )
    Path(ninja_log).touch(exist_ok=True)
    return module


CK_CONFIGS = {
    # Baseline (128x128x64)
    "baseline": {
        "BLOCK_SIZE": 256,
        "M_PER_BLOCK": 128,
        "N_PER_BLOCK": 128,
        "K_PER_BLOCK": 64,
        "AK1": 8,
        "BK1": 8,
        "M_PER_WMMA": 16,
        "N_PER_WMMA": 16,
        "M_REPEAT": 2,
        "N_REPEAT": 4,
        "A_CLUSTER_K0": 8,
        "A_CLUSTER_M": 32,
        "A_CLUSTER_K1": 1,
        "B_CLUSTER_K0": 8,
        "B_CLUSTER_N": 32,
        "B_CLUSTER_K1": 1,
        "C_CLUSTER_LENGTH_0": 1,
        "C_CLUSTER_LENGTH_1": 64,
        "C_CLUSTER_LENGTH_2": 1,
        "C_CLUSTER_LENGTH_3": 4,
        "PIPELINE_VER": "v1",
    },
    # Tile 128x64 (Solid all-rounder)
    "tile128x64": {
        "BLOCK_SIZE": 256,
        "M_PER_BLOCK": 128,
        "N_PER_BLOCK": 64,
        "K_PER_BLOCK": 64,
        "AK1": 8,
        "BK1": 8,
        "M_PER_WMMA": 16,
        "N_PER_WMMA": 16,
        "M_REPEAT": 2,
        "N_REPEAT": 2,
        "A_CLUSTER_K0": 8,
        "A_CLUSTER_M": 32,
        "A_CLUSTER_K1": 1,
        "B_CLUSTER_K0": 8,
        "B_CLUSTER_N": 32,
        "B_CLUSTER_K1": 1,
        "C_CLUSTER_LENGTH_0": 1,
        "C_CLUSTER_LENGTH_1": 64,
        "C_CLUSTER_LENGTH_2": 1,
        "C_CLUSTER_LENGTH_3": 4,
        "PIPELINE_VER": "v1",
    },
    # Tile 128x64 with K=128 (Best for small-medium M)
    "tile128x64_k128": {
        "BLOCK_SIZE": 256,
        "M_PER_BLOCK": 128,
        "N_PER_BLOCK": 64,
        "K_PER_BLOCK": 128,
        "AK1": 8,
        "BK1": 8,
        "M_PER_WMMA": 16,
        "N_PER_WMMA": 16,
        "M_REPEAT": 2,
        "N_REPEAT": 2,
        "A_CLUSTER_K0": 16,
        "A_CLUSTER_M": 16,
        "A_CLUSTER_K1": 1,
        "B_CLUSTER_K0": 16,
        "B_CLUSTER_N": 16,
        "B_CLUSTER_K1": 1,
        "C_CLUSTER_LENGTH_0": 1,
        "C_CLUSTER_LENGTH_1": 64,
        "C_CLUSTER_LENGTH_2": 1,
        "C_CLUSTER_LENGTH_3": 4,
        "PIPELINE_VER": "v1",
    },
    # Tile 128x64 with interwave scheduling (Best for large M)
    "tile128x64_interwave": {
        "BLOCK_SIZE": 256,
        "M_PER_BLOCK": 128,
        "N_PER_BLOCK": 64,
        "K_PER_BLOCK": 64,
        "AK1": 8,
        "BK1": 8,
        "M_PER_WMMA": 16,
        "N_PER_WMMA": 16,
        "M_REPEAT": 2,
        "N_REPEAT": 2,
        "A_CLUSTER_K0": 8,
        "A_CLUSTER_M": 32,
        "A_CLUSTER_K1": 1,
        "B_CLUSTER_K0": 8,
        "B_CLUSTER_N": 32,
        "B_CLUSTER_K1": 1,
        "C_CLUSTER_LENGTH_0": 1,
        "C_CLUSTER_LENGTH_1": 64,
        "C_CLUSTER_LENGTH_2": 1,
        "C_CLUSTER_LENGTH_3": 4,
        "PIPELINE_VER": "v1",
        "PIPELINE_SCHED": "ck::BlockGemmPipelineScheduler::Interwave",
    },
}

_MODULE_CACHE: dict[str, object] = {}
_AUTOTUNE_CACHE: dict[tuple[int, int, int, int], str] = {}


def get_ck_module(config_name: str):
    if config_name in _MODULE_CACHE:
        return _MODULE_CACHE[config_name]

    params = CK_CONFIGS[config_name]
    extra_cuda_cflags = [f"-DCK_PARAM_{k}={v}" for k, v in params.items()]
    extension_name = f"ck_grouped_gemm_{config_name}"
    module = load_ck_extension(extension_name, extra_cuda_cflags)
    _MODULE_CACHE[config_name] = module
    print(f"Loaded CK kernel {extension_name}")
    return module


def _autotune_select(x, w, m_offsets, y) -> str:
    print(f"Autotuning Grouped GEMM for shape x={tuple(x.shape)}, w={tuple(w.shape)}...")

    modules = []
    for name, config in CK_CONFIGS:
        try:
            modules.append((name, get_ck_module(config)))
        except Exception as e:
            print(f"Failed to load config {name}: {e}")

    best_config = next(CK_CONFIGS.keys())
    best_time = float("inf")
    for name, module in modules:
        try:
            # Warmup
            module.grouped_gemm_forward(x, w, m_offsets, y)

            # Benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()
            iters = 10
            for _ in range(iters):
                module.grouped_gemm_forward(x, w, m_offsets, y)
            torch.cuda.synchronize()
            avg_time = (time.perf_counter() - start) / iters
            print(f"  Config {name}: {avg_time * 1000:.3f} ms")

            if avg_time < best_time:
                best_time = avg_time
                best_config = name
        except Exception as e:
            print(f"  Config {name} failed runtime: {e}")

    print(f"Best config: {best_config} ({best_time * 1000:.3f} ms)")
    return best_config


def grouped_gemm_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    m_offsets: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    transpose_w: bool = False,
    ext_module=None,
) -> torch.Tensor:
    if transpose_w:
        # TODO
        pass

    M, _ = x.shape
    E, N, K = w.shape
    assert x.shape[1] == K

    if dtype is None:
        dtype = x.dtype

    y = torch.empty((M, N), device=x.device, dtype=dtype)

    module = ext_module
    if module is None:
        key = (M, N, K, E)
        if key in _AUTOTUNE_CACHE:
            config_name = _AUTOTUNE_CACHE[key]
            module = get_ck_module(config_name)
        else:
            config_name = _autotune_select(x, w, m_offsets, y)
            _AUTOTUNE_CACHE[key] = config_name
            module = get_ck_module(config_name)

    module.grouped_gemm_forward(x, w, m_offsets, y)
    return y
