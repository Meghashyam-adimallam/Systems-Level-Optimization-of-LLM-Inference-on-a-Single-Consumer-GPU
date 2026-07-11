#   Complete Technical Summary — Systems-Level Optimization of LLM Inference on a Single Consumer GPU

**Structured technical report extracted from the full repository analysis.**

---

## 1. PROJECT OVERVIEW

### What problem does this project solve?

The project addresses **low GPU utilization** during LLM inference when requests are processed sequentially. A single-request-at-a-time server leaves the GPU idle between requests, achieving roughly 0.2–0.5 requests per second on a consumer GPU. The goal is to increase throughput (req/s) and improve latency (p50, p95) purely through **systems-level optimization**—without changing the model architecture.

### Why is it important?

- **Cost:** Cloud GPUs cost ~$1/hour. At 0.5 req/s, 1000 requests take ~33 minutes and cost ~$0.55. At 1.7 req/s (dynamic batching), the same workload takes ~10 minutes and costs ~$0.17.
- **Scale:** A 3× throughput gain halves the number of GPUs needed for the same traffic.
- **Accessibility:** Demonstrates that meaningful gains are possible on a single 8 GB consumer GPU with standard tooling (HuggingFace, FastAPI).

### What type of system is this?

This is an **LLM inference optimization lab** and **reproducible benchmarking framework**. It implements and compares:

1. A naive baseline inference server
2. A static batching server (client-driven batch API)
3. A dynamic batching server (server-side request coalescing)
4. A KV cache impact experiment
5. A custom load generator with p50/p95 metrics

---

## 2. BUSINESS MOTIVATION

### Real-world problem

Production LLM serving is dominated by **GPU time cost**. Out-of-the-box sequential inference underutilizes the GPU and inflates per-request cost.

### Addressed dimensions

| Dimension | Baseline behavior | After optimization |
|-----------|-------------------|---------------------|
| **GPU utilization** | ~25% (idle between requests) | ~75% (dynamic batching) |
| **Cost per 1000 req** | ~$0.55 | ~$0.17 |
| **Latency (p50)** | 8.3 s (medium load) | 2.2 s |
| **Throughput** | 0.57 req/s | 1.73 req/s |

### Who benefits

- Teams deploying LLMs on limited GPU capacity
- Startups / labs with consumer hardware (e.g., RTX 4060)
- Engineers learning inference optimization without model changes

---

## 3. SYSTEM ARCHITECTURE

### High-level flow

```
Client → FastAPI (/generate) → [Queue/Scheduler] → model.generate() → GPU → KV Cache → Decode → Response
```

### Request lifecycle by server type

**Baseline (`server/main.py`):**
```
POST /generate → tokenize → model.generate() [blocking] → decode → return
```
Single request blocks the handler until GPU work completes.

**Static batching (`server/batched_server.py`):**
```
POST /generate        → _run_batch([prompt]) → model.generate() → return single
POST /generate_batch → _run_batch(prompts)   → model.generate() → return list
```
Batching is **client-driven**; the server does not coalesce single `/generate` requests.

**Dynamic batching (`server/dynamic_server.py`):**
```
POST /generate → append (req, Future) to pending → await Future
                    ↑
batch_worker (every 20ms): grab pending → _run_batch(prompts) [in executor] → set Future results
```
Server-side coalescing: all requests that arrive within a 20 ms window are batched together.

### Main components

| Component | Responsibility |
|-----------|----------------|
| **FastAPI** | HTTP API, async request handling |
| **lifespan** | Model/tokenizer loading at startup |
| **pending / pending_lock** | Dynamic server: queue + lock for thread-safe batching |
| **batch_worker** | Background coroutine: sleep 20ms → grab batch → run in executor |
| **model.generate()** | HuggingFace Transformers: autoregressive generation, uses KV cache by default |
| **run_in_executor** | Runs blocking GPU work in thread pool so event loop stays responsive |

### GPU interaction

- Model: `AutoModelForCausalLM`, float16, `device_map="auto"` or `.to("cuda")`
- Inference: synchronous `model.generate()` inside `run_in_executor` (dynamic) or directly (baseline)
- KV cache: managed internally by Transformers; `use_cache` passed only in the dedicated KV experiment script

### Where batching happens

- **Baseline:** No batching.
- **Static:** When caller uses `/generate_batch` with multiple prompts.
- **Dynamic:** Inside `batch_worker`; groups all requests in `pending` every 20 ms.

---

## 4. BASELINE IMPLEMENTATION

### Source

`server/main.py` (lines 54–69)

### How inference works without batching

```python
inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.temperature and req.temperature > 0,
        temperature=req.temperature or 0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
generated = outputs[:, inputs["input_ids"].shape[1]:]
```

One request → one `model.generate()` call. No batching, no queue, no explicit KV cache configuration (default is `use_cache=True`).

### Latency characteristics (from `results/`)

| Load | p50 (s) | p95 (s) |
|------|---------|---------|
| Light (10 req, 2 conc, 32 tok) | 2.09 | 3.08 |
| Medium (30 req, 5 conc, 64 tok) | 8.32 | 9.56 |
| Heavy (30 req, 8 conc, 128 tok) | 17.96 | 22.58 |

### Throughput characteristics

| Load | req/s | tok/s |
|------|-------|-------|
| Light | 0.93 | 27.7 |
| Medium | 0.57 | 30.6 |
| Heavy | 0.41 | 31.8 |

### GPU utilization behavior

~25% under load: GPU is busy only during each single generation; idle between requests.

### Baseline test script

`scripts/baseline_test.py` runs a single request and records:

- Latency (s)
- Tokens generated
- Tokens/sec
- Peak GPU memory (GB)

Example output (`results/baseline_20260228_140918.json`): 2.06 GB peak, 32 tokens, 13.11 tok/s.

---

## 5. STATIC BATCHING

### Implementation details

**Source:** `server/batched_server.py`

### API

- `POST /generate`: single request → `_run_batch([req.prompt], ...)` → single response
- `POST /generate_batch`: list of prompts → `_run_batch(prompts, ...)` → list of responses

### Batch formation

Batches are **not** formed from multiple `/generate` calls. They are formed only when the client sends multiple prompts via `/generate_batch`.

### Core logic (`_run_batch`, lines 66–96)

```python
tokenizer.padding_side = "left"
encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
# ...
outputs = model.generate(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
    max_new_tokens=max_new_tokens,
    ...
)
input_lengths = attention_mask.sum(dim=1)
# Decode per sequence using input_lengths to slice
```

- Left-padding for causal LM decoding
- `attention_mask` for variable-length inputs
- Per-sequence slicing for decoding

### Benchmark methodology note

The load generator (`benchmark/load_generator.py`) always sends to `POST /generate` only. It does **not** call `/generate_batch`. Therefore, when benchmarking the batched server:

- Each request goes to `/generate`
- Each `/generate` calls `_run_batch([single_prompt])` → effectively one-at-a-time
- Static batching infrastructure exists but is **not** exercised in the reported benchmarks

### Results (as reported)

| Load | Batched req/s | Batched p50 | Batched tok/s |
|------|---------------|-------------|---------------|
| Light | 1.03 | 2.02 | 29.6 |
| Medium | 0.71 | 6.21 | 31.4 |
| Heavy | 0.30 | 24.28 | 27.0 |

Under heavy load, batched is **slower** than baseline (0.30 vs 0.41 req/s), consistent with padding overhead and fixed-batch assumptions when receiving single requests.

### Limitations

1. No server-side coalescing of `/generate` requests
2. Client must explicitly use `/generate_batch` for batching
3. Padding overhead for highly variable prompt lengths
4. Fixed batch size implied by client behavior; no adaptive sizing

### Trade-offs

- Simplicity vs. utilization: easy to reason about, but no automatic batching
- Padding: left-padding ensures correct causality but adds tokens for short prompts in a batch

---

## 6. DYNAMIC BATCHING

### Source

`server/dynamic_server.py`

### Exact algorithm

```
Constants: BATCH_WINDOW_SEC = 0.02 (20 ms)

Data: pending: List[tuple]  # (GenerateRequest, Future)
      pending_lock: asyncio.Lock

On POST /generate(req):
  1. fut = loop.create_future()
  2. async with pending_lock:
        pending.append((req, fut))
  3. return await fut

batch_worker (runs forever, started at lifespan):
  loop:
    1. await asyncio.sleep(BATCH_WINDOW_SEC)   # 20 ms
    2. async with pending_lock:
         batch = pending
         pending = []
    3. if batch empty: continue
    4. prompts = [r.prompt for r, _ in batch]
    5. max_tok = max(r.max_new_tokens for r in batch)
    6. temp = batch[0].temperature or 0.7
    7. results = await loop.run_in_executor(None, lambda: _run_batch_sync(prompts, max_tok, temp))
    8. for each (req, fut), resp in zip(batch, results):
         if not fut.done(): fut.set_result(resp)
```

### Queue handling logic

- Shared `pending` list protected by `pending_lock`
- Each `/generate` adds `(req, fut)` and awaits its Future
- Worker atomically swaps `pending` with an empty list and processes the old contents

### Batch window logic

- Fixed **20 ms** (`BATCH_WINDOW_SEC = 0.02`)
- Worker wakes every 20 ms; all requests that arrived since the last wake are batched
- No minimum batch size; batch of 1 is allowed
- No maximum batch size in code; limited by GPU memory in practice

### How requests are grouped

All requests present in `pending` when the worker wakes are grouped into one batch. Grouping is by **arrival time**, not by prompt length or priority.

### Scheduling strategy

- Time-based: wake every 20 ms
- No priority; FIFO within the window
- Blocking GPU work offloaded to default thread pool via `run_in_executor`

### Edge case handling

| Case | Behavior |
|------|----------|
| Empty batch | `continue`; no GPU call |
| Exception in _run_batch_sync | All batch Futures get `set_exception(e)` |
| Single request | Batch size 1; processed normally |
| Variable max_new_tokens | Uses `max(r.max_new_tokens for r in batch)` — some requests get more tokens than requested |
| Variable temperature | Uses first request’s temperature for the whole batch |

### Batch execution (`_run_batch_sync`)

Same pattern as static batching: left-padding, `attention_mask`, per-sequence decode. Uses `tokenizer.pad_token_id` (set at startup).

---

## 7. KV CACHE IMPLEMENTATION

### Important clarification

This project does **not** implement a custom KV cache. It uses the built-in KV cache in HuggingFace Transformers’ `model.generate()`. The KV cache experiment measures the **performance impact** of enabling vs. disabling it.

### Where cache is controlled

- **Servers:** Do not pass `use_cache`; Transformers default is `True`
- **Experiment:** `scripts/kv_cache_test.py` explicitly sets `use_cache=True` or `use_cache=False`

### How the cache works (Transformers/PyTorch)

- Each transformer layer stores key/value tensors for previously generated tokens
- New token: compute K, V only for the new position; concatenate with cached K, V
- Avoids recomputing attention over all previous tokens

### Memory structure

Handled internally by the model. No explicit cache layout or management in this codebase.

### Performance impact (from `results/kv_cache_20260228_164607.json`)

| max_new_tokens | use_cache=True (tok/s) | use_cache=False (tok/s) | Improvement |
|----------------|------------------------|-------------------------|-------------|
| 32 | 29.28 | 25.18 | ~16% |
| 64 | 28.12 | 21.98 | ~28% |
| 128 | 28.92 | 19.98 | ~45% |

### Memory complexity

- With cache: O(seq_len × batch × layers × hidden) for K and V
- Longer sequences and larger batches increase cache size

### Experiment script

`scripts/kv_cache_test.py`:

- Loads model once
- For each token length (32, 64, 128): runs with `use_cache=True` and `use_cache=False`
- Records latency, tokens, tok/s
- Writes `results/kv_cache_<timestamp>.json`

---

## 8. LOAD GENERATION

### Source

`benchmark/load_generator.py`

### How traffic is simulated

- Creates `num_requests` async tasks
- Each task calls `send_request()` which POSTs to `{url}/generate` with a fixed prompt and `max_new_tokens`
- Uses `asyncio.gather()` to run all tasks concurrently
- Concurrency limited by `asyncio.Semaphore(concurrency)`

### Concurrency model

```python
sem = asyncio.Semaphore(concurrency)

async def one_request():
    async with sem:
        lat, tokens = await send_request(client, url, max_new_tokens=max_new_tokens)
        latencies.append(lat)
        token_counts.append(tokens)

tasks = [one_request() for _ in range(num_requests)]
await asyncio.gather(*tasks)
```

- Up to `concurrency` requests in flight at once
- No explicit pacing; requests are sent as fast as the semaphore allows
- No think time; continuous burst

### Request distribution

- Single prompt: `"Hello, world!"`
- Fixed `max_new_tokens` per run
- Requests issued as soon as semaphore allows (no Poisson or other arrival model)

### Benchmark methodology

1. Wall-clock timing: `t_start`, `t_end` around `asyncio.gather()`
2. Per-request latency from `time.perf_counter()` around `client.post()`
3. Metrics: `num_requests / wall_clock_sec`, `np.percentile(latencies, 50)`, `np.percentile(latencies, 95)`, `sum(tokens) / wall_clock_sec`
4. Retries: up to 3 on `ReadError`, `ReadTimeout`, `ConnectError` with 2 s delay
5. Timeout: `httpx.Timeout(30.0, read=600.0)` to handle long generations

### Load configurations (`scripts/benchmark_configs.py`, `run_benchmark_suite.py`)

| Config | num_requests | concurrency | max_new_tokens |
|--------|--------------|-------------|----------------|
| Light | 10 | 2 | 32 |
| Medium | 30 | 5 | 64 |
| Heavy | 30 | 8 | 128 |

### Output

JSON written to `results/load_<label>_<timestamp>.json` with:

- `num_requests`, `concurrency`, `wall_clock_sec`, `req_per_sec`, `p50_latency_sec`, `p95_latency_sec`, `tokens_per_sec`

---

## 9. METRICS & RESULTS

### Extracted numbers (from `results/` JSON files)

### Light (10 req, 2 conc, 32 tok)

| Strategy | req/s | p50 (s) | p95 (s) | tok/s |
|----------|-------|---------|---------|-------|
| Baseline | 0.926 | 2.087 | 3.080 | 27.69 |
| Batched | 1.026 | 2.020 | 2.632 | 29.65 |
| Dynamic | **1.685** | **1.021** | **1.804** | **53.92** |

### Medium (30 req, 5 conc, 64 tok)

| Strategy | req/s | p50 (s) | p95 (s) | tok/s | wall_clock (s) |
|----------|-------|---------|---------|-------|----------------|
| Baseline | 0.569 | 8.323 | 9.559 | 30.56 | 52.75 |
| Batched | 0.711 | 6.215 | 9.986 | 31.41 | 42.22 |
| Dynamic | **1.733** | **2.228** | **4.218** | **107.89** | 17.31 |

### Heavy (30 req, 8 conc, 128 tok)

| Strategy | req/s | p50 (s) | p95 (s) | tok/s |
|----------|-------|---------|---------|-------|
| Baseline | 0.413 | 17.957 | 22.579 | 31.81 |
| Batched | 0.298 | 24.279 | 33.705 | 27.00 |
| Dynamic | **1.113** | **6.833** | **7.898** | **140.17** |

### KV cache (single request, from kv_cache JSON)

| max_new_tokens | use_cache=True (tok/s) | use_cache=False (tok/s) |
|----------------|------------------------|-------------------------|
| 32 | 29.28 | 25.18 |
| 64 | 28.12 | 21.98 |
| 128 | 28.92 | 19.98 |

### Baseline single-request (from baseline_test)

- Latency: 2.44 s (32 tok), ~13 tok/s
- Peak GPU memory: 2.06 GB

### Cost framing (from README)

| Scenario | Throughput | Time for 1000 req | Cost per 1000 (@ $1/hr) |
|----------|------------|-------------------|--------------------------|
| Baseline | 0.5 req/s | ~33 min | ~$0.55 |
| Dynamic | 1.7 req/s | ~10 min | ~$0.17 |

### GPU utilization (from README, estimated)

- Baseline: ~25%
- Batched: ~60%
- Dynamic: ~75%

---

## 10. SCALING CHARACTERISTICS

### Under high load

- **Baseline:** Throughput drops (0.41 req/s heavy vs 0.93 light); p50 rises to ~18 s; requests queue at the API layer.
- **Batched:** Under single-request benchmark flow, performance degrades vs baseline on heavy load (0.30 req/s) due to overhead when not using `/generate_batch`.
- **Dynamic:** Best scaling; 1.11 req/s and 140 tok/s on heavy load.

### Bottlenecks

1. **GPU:** Main bottleneck; single GPU, no model parallelism
2. **Batch size:** No cap; very large batches could cause OOM
3. **20 ms window:** Under very high request rate, batches could grow large and increase memory/latency
4. **Thread pool:** Default executor; no tuning for CPU-bound/blocking work

### Memory growth

- Linear in batch size (more prompts → more padding, larger tensors)
- KV cache grows with sequence length
- No explicit memory monitoring or backpressure

### Failure modes

- GPU OOM on very large batches
- No timeout or cancellation for long-running generations
- No circuit breaker or health checks beyond `/health`
- No graceful degradation under overload

---

## 11. ENGINEERING TRADE-OFFS

### Latency vs throughput

- **20 ms window:** Lower window → better latency, worse batching; higher window → better throughput, worse tail latency
- **Dynamic batching:** Optimized for throughput; some requests wait up to 20 ms before processing

### Memory vs speed

- **KV cache:** More memory for cached K/V, fewer FLOPs per token
- **Batching:** More memory per batch, better GPU utilization

### Simplicity vs optimization

- Baseline: simplest; poor utilization
- Dynamic: more complex (queue, worker, Futures); best utilization
- No continuous batching or speculative decoding

### Fairness vs efficiency

- FIFO within 20 ms; no priority or fairness
- `max_new_tokens` taken as max over batch; shorter requests get unnecessary tokens

---

## 12. LIMITATIONS

### What this system does NOT handle

1. **Multi-GPU:** Single GPU only
2. **Long context:** Experiments use ≤128 new tokens; no long-context testing
3. **Continuous batching:** Fixed `max_new_tokens` per batch
4. **Streaming:** No token-by-token streaming
5. **Autoscaling:** No Kubernetes, replicas, or load-based scaling
6. **Rate limiting:** No per-user or per-tenant limits
7. **Observability:** No Prometheus, distributed tracing, or structured metrics

### Production readiness gaps

- No health checks beyond basic `/health`
- No request timeouts or cancellation
- No graceful shutdown with drain
- No authentication or API keys
- Single process; no multi-worker Uvicorn
- Results vary with system load; no isolation

### Missing features

- vLLM / TensorRT-LLM comparison
- Configurable batching window
- Batch size limits
- A/B testing or canary routing
- Model versioning or hot-swap

---

## 13. IF EXTENDED TO PRODUCTION

### Infrastructure

- **Kubernetes:** Deploy Uvicorn workers as pods; use HPA for scaling
- **Ingress/LB:** Distribute traffic across replicas
- **Monitoring:** Prometheus + Grafana for req/s, latency, GPU utilization
- **Logging:** Structured logs (e.g., JSON) for debugging and analysis

### Triton / vLLM

- **Triton Inference Server:** Would require converting the model to a Triton model repository format
- **vLLM:** PagedAttention, continuous batching, higher throughput; would replace the custom batching layer
- **Comparison:** Add a vLLM backend and run the same load generator for direct comparison

### Horizontal scaling

- Stateless API; scale by adding replicas
- Need shared or distributed queue if batching is centralized
- Or per-replica batching (current design) with client-side load balancing

### Other considerations

- Model warming and preloading
- GPU memory sharing (MPS, multi-process)
- Quantization (INT8/INT4) for larger models

---

## 14. KEY TECHNICAL INSIGHTS

### Most interesting implementation details

1. **Future-based response coupling:** Each request gets a Future; the worker resolves it when the batch completes. Clean async design.

2. **run_in_executor for GPU:** Blocking `model.generate()` runs in a thread so the FastAPI event loop stays responsive for new requests.

3. **Atomic batch swap:** `async with pending_lock: batch = pending; pending = []` ensures a clear snapshot with minimal lock time.

4. **Left-padding for causal LM:** `padding_side="left"` plus `attention_mask` keeps generation correct for variable-length prompts in a batch.

5. **max_new_tokens = max(batch):** Simple way to batch heterogeneous requests; trades some extra generation for implementation simplicity.

### Non-obvious design choices

1. **20 ms window:** Empirically chosen; not derived from queuing theory. Could be made configurable.

2. **No batch size cap:** Relying on GPU memory; large bursts could cause OOM.

3. **Single temperature per batch:** First request’s temperature used for all; could be extended to per-request sampling.

4. **Batched server benchmark:** Load generator uses `/generate` only, so static batching is not exercised in benchmarks.

### Why this approach works

- Coalescing requests amortizes GPU launch and kernel overhead
- Better GPU utilization increases tokens per second
- 20 ms keeps tail latency reasonable while allowing useful batching under typical traffic
- Simplicity makes the pipeline reproducible and easy to reason about

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `server/main.py` | Baseline server |
| `server/batched_server.py` | Static batching server |
| `server/dynamic_server.py` | Dynamic batching server |
| `benchmark/load_generator.py` | Load test client |
| `scripts/baseline_test.py` | Single-request baseline metrics |
| `scripts/kv_cache_test.py` | KV cache True vs False experiment |
| `scripts/run_benchmark_suite.py` | Full benchmark (light/medium/heavy) |
| `scripts/benchmark_configs.py` | Load config constants |
| `report/generate_charts.py` | Matplotlib charts from results JSON |
| `report/final_report.md` | Written summary |
| `docs/llm_benchmark_viz_2d.html` | Interactive 2D visualization |
| `docs/project_flow.html` | Mermaid flow diagrams |
| `requirements.txt` | Dependencies |

---

*Report generated from full repository analysis. No features were invented; all statements are grounded in the codebase and result files.*
