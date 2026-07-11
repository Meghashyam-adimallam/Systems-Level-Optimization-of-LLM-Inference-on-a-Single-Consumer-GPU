# Native / GPU layer reference

Profiler-derived inventory of CUDA kernels and C++ runtime notes for vLLM integration.

| Path | Purpose |
|------|---------|
| `cuda/top_kernels_reference.cu` | Top CUDA kernels from Zephyr + TinyLlama torch.profiler runs |
| `cpp/vllm_runtime_notes.cpp` | vLLM engine + SLA policy constants (mirrors `server/vllm_engine.py`) |

Source data: `v2/results/profiles/`, `v3/results/profiles/kernel_table_*.txt`

Serving code is Python; GPU work runs via PyTorch ATen and vLLM's C++/CUDA backend.
