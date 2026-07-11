/*
 * vLLM / SLA runtime notes (C++ layer under the Python FastAPI servers)
 *
 * Our Python servers (server/vllm_server.py, server/sla_server.py) call vLLM's
 * AsyncLLMEngine. vLLM implements continuous batching and PagedAttention in C++/CUDA.
 * This file documents integration points we rely on — not a fork of vLLM.
 */

#include <cstddef>
#include <cstdint>

namespace inference_lab {

// Env keys read by server/vllm_engine.py before engine construction
struct VllmEngineConfig {
    const char* model_key;       // e.g. "zephyr-7b"
    const char* hf_model_id;     // HuggingFaceH4/zephyr-7b-beta
    float gpu_memory_utilization;
    int max_num_seqs;
    int max_model_len;
};

constexpr VllmEngineConfig kZephyrDefaults = {
    "zephyr-7b",
    "HuggingFaceH4/zephyr-7b-beta",
    0.92f,
    32,
    4096,
};

// SLA admission layer (server/sla_server.py) sits above vLLM:
//   - rolling e2e p95 window (queue wait + inference)
//   - HTTP 503 when budget exceeded (reject_e2e_v2)
//   - stale queue wait rejection
struct SlaPolicy {
    float p95_budget_sec;
    int max_queue_depth;
    int latency_window_size;
};

constexpr SlaPolicy kDefaultSla = {3.0f, 32, 50};

// Hand-rolled HF servers (baseline/batched/dynamic) use PyTorch C++ ATen ops;
// profiler shows aten::mm, flash attention, and elementwise kernels dominate.

}  // namespace inference_lab
