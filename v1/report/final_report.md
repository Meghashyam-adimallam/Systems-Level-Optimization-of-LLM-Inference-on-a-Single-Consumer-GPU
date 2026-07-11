
# Systems-Level Optimization of LLM Inference on a Single Consumer GPU — Final Benchmark Report

## Summary

- **Baseline** (single-request), **static batching**, and **dynamic batching** (20 ms window) were compared under the same load.
- **KV cache** was compared: `use_cache=True` vs `use_cache=False`.
- Charts were generated with `report/generate_charts.py` from `results/load_*.json`.

**Setup:** TinyLlama-1.1B, NVIDIA RTX 4060 Laptop GPU (8 GB).

**Load configs (multiple):**

| Config  | Requests | Concurrency | Max tokens |
|---------|----------|-------------|------------|
| Light   | 10       | 2           | 32         |
| Medium  | 30       | 5           | 64         |
| Heavy   | 30       | 8           | 128        |

**KV cache:** tested at 32, 64, and 128 tokens (use_cache True vs False).

---

## Load test results — single run (30 req, concurrency 5, 64 tokens)

| Strategy   | p50 Latency | p95 Latency | Req/s | Tok/s | Wall clock |
|------------|-------------|-------------|-------|-------|------------|
| Baseline   | 14.9 s      | 39.0 s      | 0.22  | 10.7  | 138.1 s    |
| Batched    | 9.4 s       | 12.3 s      | 0.53  | 27.7  | 56.9 s     |
| Dynamic    | 11.9 s      | 23.3 s      | 0.36  | 23.0  | 83.6 s     |

- **Batched** gave the best throughput and latency (about 2.4× req/s and 2.6× tok/s vs baseline).
- **Dynamic** sat between baseline and batched, improving over baseline by batching requests that arrive within the 20 ms window.

---

## Load test results — multiple loads (light / medium / heavy)

**Light (10 req, 2 conc, 32 tok)**

| Metric   | Baseline | Batched | Dynamic |
|----------|----------|---------|---------|
| Req/s    | 0.93     | 1.03    | **1.69** |
| p50 (s)  | 2.09     | 2.02    | **1.02** |
| Tok/s    | 27.7     | 29.6    | **53.9** |

**Medium (30 req, 5 conc, 64 tok)**

| Metric   | Baseline | Batched | Dynamic |
|----------|----------|---------|---------|
| Req/s    | 0.57     | 0.71    | **1.73** |
| p50 (s)  | 8.32     | 6.21    | **2.23** |
| Tok/s    | 30.6     | 31.4    | **107.9** |

**Heavy (30 req, 8 conc, 128 tok)**

| Metric   | Baseline | Batched | Dynamic |
|----------|----------|---------|---------|
| Req/s    | 0.41     | 0.30    | **1.11** |
| p50 (s)  | 17.96    | 24.28   | **6.83** |
| Tok/s    | 31.8     | 27.0    | **140.2** |

---

## KV cache experiment (32, 64, 128 tokens)

**Tok/s by token length**

| Setting           | 32 tok  | 64 tok  | 128 tok |
|-------------------|---------|---------|---------|
| use_cache=True    | 29.3    | 28.1    | 28.9    |
| use_cache=False   | 25.2    | 22.0    | 20.0    |

With cache on, throughput is ~16–45% higher; the gain is largest at 128 tokens.

---

## Charts & visual report

- **Bar charts:** Run `python report/generate_charts.py` to create:
  - `report/benchmark_comparison.png` — throughput, p50/p95, tok/s (light / medium / heavy)
  - `report/kv_cache_comparison.png` — KV cache tok/s by token length
- **One-page visual report:** Open `report/benchmark_report.html` in your browser to see both charts and all tables together.

---

## Conclusions

1. **Dynamic batching** gave the best results in this run: up to **1.73 req/s** (medium) and **140 tok/s** (heavy) vs 0.57 req/s and 31 tok/s for baseline — about **3× req/s** and **3–4× tok/s**.
2. **Static batching** helped on light and medium (e.g. 1.03 req/s and 0.71 req/s) but was slower than baseline on heavy (0.30 req/s, 24.3 s p50) due to padding and batch overhead.
3. **KV cache** (use_cache=True) is important: ~16–45% higher tok/s; the gain grows with longer sequences (20.0 vs 28.9 tok/s at 128 tokens).
4. The project showed that **dynamic batching** and **KV cache** improve LLM serving on a single consumer GPU (RTX 4060 8 GB) without changing the model.

---

## Reproducibility and follow-up improvements

The original report above used single-run benchmark numbers. The repository now includes a stricter benchmark path:

- `benchmark/load_generator.py` supports fixed prompt seeds and warmup requests.
- `scripts/run_benchmark_suite.py` supports `--runs N` and reports mean ± standard deviation.
- `scripts/gpu_monitor.py` records real `nvidia-smi` GPU utilization, VRAM, and power samples while benchmarks run.
- `report/generate_gpu_charts.py` turns GPU traces into timeline charts.
- `server/vllm_server.py` provides an optional vLLM comparison backend using the same `/generate` API contract.

Recommended rerun:

```bash
python scripts/run_benchmark_suite.py --url http://127.0.0.1:8000 --strategy baseline --runs 3 --monitor-gpu
python scripts/run_benchmark_suite.py --url http://127.0.0.1:8000 --strategy batched  --runs 3 --monitor-gpu
python scripts/run_benchmark_suite.py --url http://127.0.0.1:8000 --strategy dynamic  --runs 3 --monitor-gpu
python report/generate_charts.py
python report/generate_gpu_charts.py
```

---

## Honest failure analysis: static batching

The first static batching implementation exposed an important systems lesson: batching should live in the server scheduler, not depend on the client. Earlier, `/generate_batch` could process a list of prompts, but normal `/generate` traffic was still processed one request at a time. Under heavy load this made the "static batching" strategy underperform because the benchmark load generator used `/generate`, not `/generate_batch`.

The server now coalesces normal `/generate` calls into fixed-size server-side batches of 4 with a short timeout. This turns static batching from a client-driven API feature into a real serving strategy and makes the comparison against dynamic batching fairer.

---

## Production comparison: vLLM

The project also includes an optional vLLM backend for a more credible ceiling comparison. The goal is not to claim the hand-written scheduler beats vLLM; the goal is to show where hand-rolled batching helps, where it hits limits, and why production inference servers use continuous batching and PagedAttention.

Run on a Linux/CUDA machine:

```bash
pip install -r requirements-vllm.txt
uvicorn server.vllm_server:app --host 127.0.0.1 --port 8000
python scripts/run_benchmark_suite.py --url http://127.0.0.1:8000 --strategy vllm --runs 3 --monitor-gpu
```
