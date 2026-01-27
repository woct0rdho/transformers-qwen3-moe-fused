#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <tuple>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_splitk_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/utility/ignore.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include <c10/cuda/CUDAStream.h>

// Type definitions matching the BF16 WMMA example
using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = BF16;
using BDataType        = BF16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DsDataType       = ck::Tuple<>;
using EDataType        = BF16;

using ALayout  = Row;
using BLayout  = Col; // B is KxN ColumnMajor (transposed W which is NxK RowMajor)
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

// Tuning Parameters with Defaults
#ifndef CK_PARAM_BLOCK_SIZE
#define CK_PARAM_BLOCK_SIZE 256
#endif
#ifndef CK_PARAM_M_PER_BLOCK
#define CK_PARAM_M_PER_BLOCK 128
#endif
#ifndef CK_PARAM_N_PER_BLOCK
#define CK_PARAM_N_PER_BLOCK 64
#endif
#ifndef CK_PARAM_K_PER_BLOCK
#define CK_PARAM_K_PER_BLOCK 64
#endif
#ifndef CK_PARAM_AK1
#define CK_PARAM_AK1 8
#endif
#ifndef CK_PARAM_BK1
#define CK_PARAM_BK1 8
#endif
#ifndef CK_PARAM_M_PER_WMMA
#define CK_PARAM_M_PER_WMMA 16
#endif
#ifndef CK_PARAM_N_PER_WMMA
#define CK_PARAM_N_PER_WMMA 16
#endif
#ifndef CK_PARAM_M_REPEAT
#define CK_PARAM_M_REPEAT 2
#endif
#ifndef CK_PARAM_N_REPEAT
#define CK_PARAM_N_REPEAT 2
#endif

#ifndef CK_PARAM_A_CLUSTER_K0
#define CK_PARAM_A_CLUSTER_K0 8
#endif
#ifndef CK_PARAM_A_CLUSTER_M
#define CK_PARAM_A_CLUSTER_M 32
#endif
#ifndef CK_PARAM_A_CLUSTER_K1
#define CK_PARAM_A_CLUSTER_K1 1
#endif

#ifndef CK_PARAM_B_CLUSTER_K0
#define CK_PARAM_B_CLUSTER_K0 8
#endif
#ifndef CK_PARAM_B_CLUSTER_N
#define CK_PARAM_B_CLUSTER_N 32
#endif
#ifndef CK_PARAM_B_CLUSTER_K1
#define CK_PARAM_B_CLUSTER_K1 1
#endif

#ifndef CK_PARAM_PIPELINE_VER
#define CK_PARAM_PIPELINE_VER v1
#endif

#ifndef CK_PARAM_C_CLUSTER_LENGTH_0
#define CK_PARAM_C_CLUSTER_LENGTH_0 1
#endif
#ifndef CK_PARAM_C_CLUSTER_LENGTH_1
#define CK_PARAM_C_CLUSTER_LENGTH_1 64
#endif
#ifndef CK_PARAM_C_CLUSTER_LENGTH_2
#define CK_PARAM_C_CLUSTER_LENGTH_2 1
#endif
#ifndef CK_PARAM_C_CLUSTER_LENGTH_3
#define CK_PARAM_C_CLUSTER_LENGTH_3 4
#endif

#ifndef CK_PARAM_PIPELINE_SCHED
#define CK_PARAM_PIPELINE_SCHED ck::BlockGemmPipelineScheduler::Intrawave
#endif

// MNKPadding to handle cases where dimensions aren't multiples of block sizes
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGroupedGemm_Wmma_CShuffleV3<
    ALayout,
    BLayout,
    DsLayout,
    ELayout,
    ADataType,
    BDataType,
    AccDataType,
    CShuffleDataType,
    DsDataType,
    EDataType,
    AElementOp,
    BElementOp,
    CDEElementOp,
    GemmSpec,
    CK_PARAM_BLOCK_SIZE,
    CK_PARAM_M_PER_BLOCK,
    CK_PARAM_N_PER_BLOCK,
    CK_PARAM_K_PER_BLOCK,
    CK_PARAM_AK1,
    CK_PARAM_BK1,
    CK_PARAM_M_PER_WMMA,
    CK_PARAM_N_PER_WMMA,
    CK_PARAM_M_REPEAT,
    CK_PARAM_N_REPEAT,
    S<CK_PARAM_A_CLUSTER_K0, CK_PARAM_A_CLUSTER_M, CK_PARAM_A_CLUSTER_K1>, // ABlockTransferThreadCluster
    S<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>, // ABlockTransferSrcAccessOrder
    2, // ABlockTransferSrcVectorDim
    8, // ABlockTransferSrcScalar
    8, // ABlockTransferDstScalar
    1, // ABlockLdsAddExtraM
    S<CK_PARAM_B_CLUSTER_K0, CK_PARAM_B_CLUSTER_N, CK_PARAM_B_CLUSTER_K1>, // BBlockTransferThreadCluster
    S<1, 0, 2>, // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>, // BBlockTransferSrcAccessOrder
    2, // BBlockTransferSrcVectorDim
    8, // BBlockTransferSrcScalar
    8, // BBlockTransferDstScalar
    1, // BBlockLdsAddExtraN
    1, // CShuffleMRepeat
    1, // CShuffleNRepeat
    S<CK_PARAM_C_CLUSTER_LENGTH_0, CK_PARAM_C_CLUSTER_LENGTH_1, CK_PARAM_C_CLUSTER_LENGTH_2, CK_PARAM_C_CLUSTER_LENGTH_3>, // CBlockTransferClusterLengths_MBlock_MRepeat
    8, // CBlockTransferScalarPerVector
    CK_PARAM_PIPELINE_SCHED,
    ck::BlockGemmPipelineVersion::CK_PARAM_PIPELINE_VER>;

void grouped_gemm_forward(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& m_offsets,
    torch::Tensor& y)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(m_offsets.is_cuda(), "m_offsets must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(w.dtype() == torch::kBFloat16, "w must be bfloat16");
    TORCH_CHECK(y.dtype() == torch::kBFloat16, "y must be bfloat16");
    TORCH_CHECK(m_offsets.dtype() == torch::kInt32, "m_offsets must be int32");

    const int64_t M_total = x.size(0);
    const int64_t K = x.size(1);
    const int64_t E = w.size(0);
    const int64_t N = w.size(1);

    TORCH_CHECK(w.size(2) == K, "w last dim must match x last dim (K)");
    TORCH_CHECK(y.size(0) == M_total, "y rows must match x rows");
    TORCH_CHECK(y.size(1) == N, "y cols must match w cols (N)");

    // Move m_offsets to CPU to construct GemmDescs
    const auto m_offsets_cpu = m_offsets.to(torch::kCPU);
    const auto m_offsets_acc = m_offsets_cpu.accessor<int32_t, 1>();

    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
    std::vector<const void*> p_a;
    std::vector<const void*> p_b;
    std::vector<void*> p_c;
    std::vector<std::array<const void*, 0>> p_Ds; // Empty Ds

    const int stride_A = x.stride(0);
    const int stride_we = w.stride(0);
    const int stride_B = w.stride(1);
    const int stride_C = y.stride(0);

    const char* const x_ptr_base = reinterpret_cast<const char*>(x.data_ptr());
    const char* const w_ptr_base = reinterpret_cast<const char*>(w.data_ptr());
    char* const y_ptr_base = reinterpret_cast<char*>(y.data_ptr());

    int32_t m_end = 0;
    for (int i = 0; i < E; ++i) {
        const int32_t m_start = m_end;
        m_end = m_offsets_acc[i];
        const int32_t m_size = m_end - m_start;
        if (m_size > 0) {
            gemm_descs.push_back({m_size, (int)N, (int)K, stride_A, stride_B, stride_C, {}});
            const size_t a_offset = (size_t)m_start * stride_A * 2; // bf16 is 2 bytes
            p_a.push_back((const void*)(x_ptr_base + a_offset));
            const size_t b_offset = (size_t)i * stride_we * 2;
            p_b.push_back((const void*)(w_ptr_base + b_offset));
            const size_t c_offset = (size_t)m_start * stride_C * 2;
            p_c.push_back((void*)(y_ptr_base + c_offset));
            p_Ds.push_back({});
        }
    }

    if (gemm_descs.empty()) {
        return;
    }

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CDEElementOp{};

    auto gemm = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    auto argument = gemm.MakeArgument(p_a, p_b, p_Ds, p_c, gemm_descs, a_element_op, b_element_op, c_element_op);

    // Set k_batch = 1 (disable split-K split, or rather single batch)
    gemm.SetKBatchSize(&argument, 1);

    const size_t workspace_size = gemm.GetWorkSpaceSize(&argument);
    const size_t kargs_size = gemm.GetDeviceKernelArgSize(&argument);

    torch::Tensor workspace;
    if (workspace_size > 0) {
        workspace = torch::empty({(long)workspace_size}, torch::TensorOptions().device(x.device()).dtype(torch::kUInt8));
        gemm.SetWorkSpacePointer(&argument, workspace.data_ptr());
    }

    torch::Tensor device_kargs;
    if (kargs_size > 0) {
        device_kargs = torch::empty({(long)kargs_size}, torch::TensorOptions().device(x.device()).dtype(torch::kUInt8));
        gemm.SetDeviceKernelArgs(&argument, device_kargs.data_ptr());
    }

    if (!gemm.IsSupportedArgument(argument)) {
        throw std::runtime_error("CK Grouped GEMM: Unsupported argument");
    }

    const hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    invoker.Run(argument, StreamConfig{stream, false});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_gemm_forward", &grouped_gemm_forward, "Grouped GEMM Forward (CK)");
}
