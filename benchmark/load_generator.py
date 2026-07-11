import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import List, Sequence

import httpx
import numpy as np


DEFAULT_PROMPTS = [
    "Explain GPU batching in one short paragraph.",
    "Summarize why KV cache improves autoregressive decoding.",
    "Write a concise explanation of request latency.",
    "Describe how an async queue helps an inference server.",
    "Give one practical reason to measure p95 latency.",
    "Explain why padding can hurt static batching.",
    "Describe the tradeoff between latency and throughput.",
    "Summarize how a FastAPI LLM server handles a prompt.",
]

LONG_CONTEXT_PROMPTS = [
    "Explain how GPU memory, KV cache, and request batching interact when serving a 7B language model on one GPU.",
    "Compare hand-rolled HuggingFace batching with vLLM continuous batching for throughput and tail latency.",
    "Describe how a serving queue should throttle requests when p95 latency exceeds a 3 second budget.",
    "Walk through the path from HTTP POST /generate to GPU execution, including queue saturation and timeouts.",
]

PROMPT_SETS = {
    "default": DEFAULT_PROMPTS,
    "long_context": LONG_CONTEXT_PROMPTS,
}


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_new_tokens: int = 64,
    max_retries: int = 3,
    retry_reject: bool = True,
) -> tuple[float, int]:
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
    url = f"{url.rstrip('/')}/generate"
    last_err = None
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = time.perf_counter() - t0
            num_tokens = data.get("num_tokens", 0)
            return latency, num_tokens
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503 and not retry_reject:
                raise
            if e.response.status_code in (503, 504) and attempt < max_retries - 1:
                last_err = e
                await asyncio.sleep(2.0)
                continue
            raise
        except (httpx.ReadError, httpx.ReadTimeout, httpx.ConnectError) as e:
            last_err = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2.0)
            continue
    raise last_err


async def run_load_test(
    url: str,
    num_requests: int,
    concurrency: int,
    max_new_tokens: int = 64,
    warmup_requests: int = 2,
    seed: int = 42,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    prompt_set: str = "default",
    save_latencies: bool = False,
    retry_reject: bool = True,
) -> dict:
    latencies: List[float] = []
    token_counts: List[int] = []
    errors: List[str] = []
    active_prompts = list(prompts)
    if prompt_set in PROMPT_SETS:
        active_prompts = list(PROMPT_SETS[prompt_set])
    rng = random.Random(seed)
    request_prompts = [rng.choice(active_prompts) for _ in range(num_requests)]

    timeout = httpx.Timeout(60.0, read=600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(warmup_requests):
            try:
                await send_request(
                    client,
                    url,
                    prompt=active_prompts[i % len(active_prompts)],
                    max_new_tokens=max_new_tokens,
                    retry_reject=True,  # warmup is not measured
                )
            except Exception:
                pass  # SLA may reject warmup if prior load left window hot

        sem = asyncio.Semaphore(concurrency)

        async def one_request(prompt: str):
            async with sem:
                try:
                    lat, tokens = await send_request(
                        client,
                        url,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        retry_reject=retry_reject,
                    )
                    latencies.append(lat)
                    token_counts.append(tokens)
                except Exception as exc:
                    errors.append(str(exc))

        tasks = [one_request(prompt) for prompt in request_prompts]
        t_start = time.perf_counter()
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    total_sec = t_end - t_start
    arr = np.array(latencies)
    successful = len(latencies)
    result = {
        "num_requests": num_requests,
        "successful_requests": successful,
        "failed_requests": len(errors),
        "concurrency": concurrency,
        "warmup_requests": warmup_requests,
        "seed": seed,
        "wall_clock_sec": total_sec,
        "req_per_sec": successful / total_sec if total_sec > 0 else 0,
        "p50_latency_sec": float(np.percentile(arr, 50)) if successful else 0,
        "p95_latency_sec": float(np.percentile(arr, 95)) if successful else 0,
        "p99_latency_sec": float(np.percentile(arr, 99)) if successful else 0,
        "tokens_per_sec": sum(token_counts) / total_sec if total_sec > 0 else 0,
        "prompt_set": prompt_set,
        "total_tokens": sum(token_counts),
        "errors": errors[:5],
    }
    if save_latencies:
        result["latency_sec"] = [float(x) for x in latencies]
    return result


def main():
    parser = argparse.ArgumentParser(description="Load generator for LLM API")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=str, default="", help="e.g. baseline or batched; used in saved filename")
    parser.add_argument("--out-dir", type=str, default="results", help="directory to save JSON results")
    parser.add_argument("--save-latencies", action="store_true", help="include per-request latency_sec list in JSON")
    args = parser.parse_args()

    result = asyncio.run(
        run_load_test(
            args.url,
            args.num_requests,
            args.concurrency,
            args.max_new_tokens,
            warmup_requests=args.warmup_requests,
            seed=args.seed,
            save_latencies=args.save_latencies,
        )
    )
    print("Results:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"load_{args.label}_{ts}.json" if args.label else f"load_{ts}.json"
    out_path = out_dir / name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
