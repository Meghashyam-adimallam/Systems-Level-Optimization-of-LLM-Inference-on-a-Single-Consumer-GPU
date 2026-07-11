/*
 * CUDA kernel inventory — Zephyr-7B inference (A100)
 * Extracted from torch.profiler: v3/results/profiles/kernel_table_zephyr-7b_bs4_tok32.txt
 *
 * These are the dominant GPU kernels under our serving load. Python servers call
 * into PyTorch/vLLM; actual matmul/attention work runs as CUDA kernels below.
 */

#include <cstdint>

struct KernelStat {
    const char* name;
    float cuda_pct;
    int num_calls;
};

// Top kernels by Self CUDA % (batch=4, 32 new tokens)
static const KernelStat kZephyrTopKernels[] = {
    {"aten::mm / ampere_fp16 GEMM", 74.99f, 7200},
    {"ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_6", 35.50f, 1984},
    {"ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f", 29.29f, 3040},
    {"aten::mul elementwise", 6.70f, 9408},
    {"aten::cat / CatArrayBatchedCopy", 3.86f, 4224},
    {"aten::copy_ / D2D memcpy", 2.83f, 4519},
    {"aten::mean / reduce_kernel", 2.64f, 2080},
    {"cublasLt splitKreduce", 2.03f, 1984},
};

static const int kZephyrKernelCount =
    sizeof(kZephyrTopKernels) / sizeof(kZephyrTopKernels[0]);

/*
 * TinyLlama-1.1B reference (v2 profiler, batch=4):
 *   aten::mm ~45% CUDA, flash attention kernels ~13%, elementwise ~10%
 * See: v2/results/profiles/kernel_table_tinyllama_bs4_tok32.txt
 */

__global__ void benchmark_warmup_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<float>(idx) * 0.001f;
    }
}

// Not linked into serving — reference + minimal warmup stub for native/ layout.
