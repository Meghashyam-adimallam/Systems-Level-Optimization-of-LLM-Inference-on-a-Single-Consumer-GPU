# Tail Latency Analysis (v1 + v2)

This doc explains **why p95/p99 matter** and what our saved JSON shows — using real files in `v1/results/` and `v2/results/`.

## What we measure

The load generator records per-request wall time and reports **p50**, **p95**, and **p99** (v2/v3 multi-run JSON includes mean ± std).

Throughput (req/s) alone hides **tail latency**: a few slow requests dominate user experience under contention.

## v1 — RTX 4060, medium load (30 req, concurrency 5)

| Strategy | req/s | p50 | p95 |
|----------|-------|-----|-----|
| Baseline | 0.57 | 8.32 s | 9.56 s |
| Batched | 0.71 | 6.21 s | 9.99 s |
| Dynamic | **1.73** | **2.23 s** | **4.22 s** |

**Takeaway:** On medium load, dynamic batching improves **both** throughput and tail latency vs baseline and static batching.

## v1 — heavy load (30 req, concurrency 8, 128 tokens)

| Strategy | req/s | p50 | p95 |
|----------|-------|-----|-----|
| Batched | 0.30 | 24.28 s | **33.70 s** |
| Dynamic | 1.11 | 6.83 s | **7.90 s** |

**Takeaway:** Under heavier contention, **static batching** suffers the worst tails: requests wait for a full batch and pay padding cost on variable prompts. Dynamic coalescing (20 ms window) reduces queueing delay and keeps p95 much lower.

## v2 — A100, heavy load (3 runs, mean)

| Strategy | req/s | p50 | p95 |
|----------|-------|-----|-----|
| Batched | 1.24 ± 0.14 | 6.31 s | 7.87 s |
| Dynamic | **2.08 ± 0.18** | **3.51 s** | **3.90 s** |

**Takeaway:** The same pattern holds with rigorous multi-run stats on A100 — dynamic improves throughput **and** tightens the tail vs static batching under heavy load.

## v3 — why SLA p95 diverges from vLLM

On medium load (saved v3 JSON):

| Engine | req/s | p50 | p99 |
|--------|-------|-----|-----|
| vLLM | 5.65 | 0.88 s | 0.89 s |
| SLA | 1.19 | 4.25 s | 4.29 s |

vLLM optimizes for **throughput** with PagedAttention + continuous batching. The SLA layer **throttles intake** (queue cap, p95 budget, 503 backpressure) so p99 stays near p50 — a policy choice, not a broken scheduler.

## Chart

Regenerate: `python scripts/plot_tail_latency.py --results-dir v2/results --out-dir v2/report --tier v2`

Output: `v2/report/tail_latency_comparison.png`
