"""
Optional v3 second engine: llama.cpp (GGUF) medium-load benchmark.

Run on A100 Colab after: pip install llama-cpp-python huggingface_hub

  python scripts/bench_llamacpp.py --out-dir v3/results --runs 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

DEFAULT_REPO = "bartowski/zephyr-7b-beta-GGUF"
DEFAULT_FILE = "zephyr-7b-beta-Q4_K_M.gguf"

PROMPTS = [
    "Summarize the tradeoff between throughput and tail latency in LLM serving.",
    "Why does PagedAttention help vLLM batch variable-length requests?",
    "Explain KV cache memory growth during autoregressive decoding.",
] * 10


def download_gguf(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename)


def run_once(llm, num_requests: int, concurrency: int, max_tokens: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]
    latencies = []
    token_counts = []

    # llama.cpp python binding is sync; approximate load with sequential requests
    # (conservative vs vLLM async server — note in README)
    t0 = time.perf_counter()
    for prompt in prompts:
        req_start = time.perf_counter()
        out = llm.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=0.7)
        latencies.append(time.perf_counter() - req_start)
        text = out["choices"][0]["text"]
        token_counts.append(max(len(text.split()), 1))
    wall = time.perf_counter() - t0

    arr = np.array(latencies)
    return {
        "num_requests": num_requests,
        "successful_requests": len(latencies),
        "failed_requests": 0,
        "concurrency": concurrency,
        "warmup_requests": 0,
        "seed": seed,
        "wall_clock_sec": wall,
        "req_per_sec": len(latencies) / wall if wall > 0 else 0,
        "p50_latency_sec": float(np.percentile(arr, 50)),
        "p95_latency_sec": float(np.percentile(arr, 95)),
        "p99_latency_sec": float(np.percentile(arr, 99)),
        "tokens_per_sec": sum(token_counts) / wall if wall > 0 else 0,
        "prompt_set": "default",
        "total_tokens": sum(token_counts),
        "engine": "llama.cpp",
        "note": "sequential GGUF requests; not async server — compare trends only",
    }


def aggregate(runs: list) -> dict:
    out = {"runs": runs, "num_runs": len(runs), "engine": "llama.cpp"}
    for key in ("req_per_sec", "p50_latency_sec", "p95_latency_sec", "p99_latency_sec", "tokens_per_sec"):
        vals = [r[key] for r in runs]
        out[f"{key}_mean"] = statistics.mean(vals)
        out[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--gguf-file", default=DEFAULT_FILE)
    parser.add_argument("--num-requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=5, help="documented; llama.cpp runs sequential")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="v3/results")
    args = parser.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit("pip install llama-cpp-python") from exc

    print(f"Downloading {args.repo} / {args.gguf_file} ...")
    path = download_gguf(args.repo, args.gguf_file)
    print(f"Loading {path} ...")
    llm = Llama(model_path=path, n_gpu_layers=-1, verbose=False)

    runs = []
    for i in range(args.runs):
        print(f"Run {i + 1}/{args.runs} ...")
        runs.append(
            run_once(llm, args.num_requests, args.concurrency, args.max_new_tokens, args.seed + i)
        )

    result = aggregate(runs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"load_llamacpp_medium_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("req_per_sec_mean", "p50_latency_sec_mean", "p95_latency_sec_mean")}, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
