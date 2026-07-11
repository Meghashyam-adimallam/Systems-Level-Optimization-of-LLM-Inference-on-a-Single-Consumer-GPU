# v3 SLA findings

## Model

All v3 load JSON files use env key `mistral-7b` or `zephyr-7b` but load:

**`HuggingFaceH4/zephyr-7b-beta`**

Official `mistralai/Mistral-7B-Instruct-v0.3` was not used (gated). Relabel all v3 tables as **Zephyr-7B-beta**.

---

## Current results (`reject_e2e_v2`)

Budget: **p95 < 3.0 s** (`SLA_P95_BUDGET_SEC=3.0`)

| Load | SLA req/s | SLA p95 | 503 fails | vs budget |
|------|-----------|---------|-----------|-----------|
| light | 2.33 | 0.86 s | 0 | ✅ under |
| medium | 1.16 | 3.30 s | 52 | sheds load; successful p95 ~budget |
| heavy | 0.58 | 3.43 s | 54 | sheds load; successful p95 ~budget |
| long_context | 0.41 | 4.37 s | 34 | ❌ still overshoots |

vLLM medium for comparison: **5.65 req/s**, p95 **0.89 s**.

Policy: `sla_policy: "reject_e2e_v2"` — HTTP **503** when rolling e2e p95 > budget; stale queue waits rejected.

---

## Old failure case (broken policy)

Saved before the fix; kept here as the honest debugging story.

| Load | SLA req/s | SLA p95 | fails |
|------|-----------|---------|-------|
| light | 2.35 | 0.86 s | 0 |
| medium | 1.19 | **4.29 s** | 0 |
| heavy | 0.63 | **13.50 s** | 0 |
| long_context | 0.72 | **9.19 s** | 0 |

`failed_requests: 0` while p95 far above budget — policy was not enforcing.

### Root cause (old `sla_server.py`)

1. **No hard reject** — when p95 > budget, server only `sleep(0.05)` then accepted the request anyway.
2. **Wrong latency window** — tracked inference time only, not queue wait (load generator measures end-to-end).

### Fix applied (repo + re-run)

- Reject with **HTTP 503** when rolling p95 > budget (`stats.rejected_budget`)
- Record **queue wait + inference** in rolling window
- `POST /admin/reset_window` between benchmark configs/runs
- `/metrics` exposes `rejected_budget`, `budget_exceeded`, `model_id`

JSON: `v3/results/load_sla_*.json`
