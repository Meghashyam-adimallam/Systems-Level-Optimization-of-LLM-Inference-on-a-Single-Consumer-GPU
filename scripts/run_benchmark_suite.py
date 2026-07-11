import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root))
from benchmark.load_generator import run_load_test

BENCHMARK_LOADS = [
    {"name": "light", "num_requests": 10, "concurrency": 2, "max_new_tokens": 32},
    {"name": "medium", "num_requests": 30, "concurrency": 5, "max_new_tokens": 64},
    {"name": "heavy", "num_requests": 30, "concurrency": 8, "max_new_tokens": 128},
]

CAPSTONE_LOADS = BENCHMARK_LOADS + [
    {
        "name": "long_context",
        "num_requests": 20,
        "concurrency": 4,
        "max_new_tokens": 256,
        "prompt_set": "long_context",
    },
]

METRICS_FOR_AGGREGATE = [
    "req_per_sec",
    "p50_latency_sec",
    "p95_latency_sec",
    "p99_latency_sec",
    "tokens_per_sec",
    "wall_clock_sec",
]


def aggregate_runs(runs):
    aggregate = {"runs": runs, "num_runs": len(runs)}
    for key in METRICS_FOR_AGGREGATE:
        vals = [r[key] for r in runs if key in r]
        aggregate[f"{key}_mean"] = statistics.mean(vals) if vals else 0
        aggregate[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0
    if runs:
        aggregate.update(
            {
                "num_requests": runs[0]["num_requests"],
                "concurrency": runs[0]["concurrency"],
                "max_new_tokens": runs[0].get("max_new_tokens"),
                "warmup_requests": runs[0].get("warmup_requests"),
                "prompt_set": runs[0].get("prompt_set", "default"),
                "successful_requests_mean": statistics.mean(
                    r.get("successful_requests", 0) for r in runs
                ),
                "failed_requests_total": sum(r.get("failed_requests", 0) for r in runs),
            }
        )
    return aggregate


def start_gpu_monitor(label: str, out_dir: Path, enabled: bool):
    if not enabled:
        return None
    cmd = [
        sys.executable,
        str(_root / "scripts" / "gpu_monitor.py"),
        "--label",
        label,
        "--out-dir",
        str(out_dir / "gpu_traces"),
    ]
    try:
        return subprocess.Popen(cmd)
    except OSError as exc:
        print(f"  GPU monitor not started: {exc}")
        return None


def stop_gpu_monitor(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def reset_sla_window(url: str) -> None:
    import urllib.request

    req = urllib.request.Request(
        f"{url.rstrip('/')}/admin/reset_window",
        data=b"",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(f"  SLA window reset: {resp.read().decode()[:120]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["baseline", "batched", "dynamic", "vllm", "sla"],
    )
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor-gpu", action="store_true")
    parser.add_argument("--capstone", action="store_true")
    parser.add_argument(
        "--loads",
        type=str,
        default="",
        help="comma-separated load names to run (e.g. medium,heavy). Default: all",
    )
    args = parser.parse_args()

    loads = CAPSTONE_LOADS if args.capstone else BENCHMARK_LOADS
    if args.loads.strip():
        wanted = {x.strip() for x in args.loads.split(",") if x.strip()}
        loads = [cfg for cfg in loads if cfg["name"] in wanted]
        if not loads:
            raise SystemExit(f"No matching loads in {wanted}")
    model_key = os.environ.get("VLLM_MODEL", "tinyllama")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for cfg in loads:
        label = f"{args.strategy}_{cfg['name']}"
        try:
            print(
                f"Running {label} ({cfg['num_requests']} req, {cfg['concurrency']} conc, "
                f"{cfg['max_new_tokens']} tok)..."
            )
            if args.strategy == "sla":
                try:
                    reset_sla_window(args.url)
                except Exception as exc:
                    print(f"  WARN: SLA window reset failed ({exc})")
            monitor = start_gpu_monitor(label, out_dir, args.monitor_gpu)
            runs = []
            try:
                for run_idx in range(args.runs):
                    if args.strategy == "sla":
                        try:
                            reset_sla_window(args.url)
                        except Exception as exc:
                            print(f"  WARN: SLA reset before run failed ({exc})")
                    try:
                        result = asyncio.run(
                            run_load_test(
                                args.url,
                                num_requests=cfg["num_requests"],
                                concurrency=cfg["concurrency"],
                                max_new_tokens=cfg["max_new_tokens"],
                                warmup_requests=args.warmup_requests,
                                seed=args.seed + run_idx,
                                prompt_set=cfg.get("prompt_set", "default"),
                                retry_reject=args.strategy != "sla",
                            )
                        )
                    except Exception as exc:
                        print(f"  run {run_idx + 1}/{args.runs} FAILED: {exc}")
                        continue
                    result["run_idx"] = run_idx + 1
                    runs.append(result)
                    print(
                        f"  run {run_idx + 1}/{args.runs}: "
                        f"req/s {result['req_per_sec']:.3f}  "
                        f"p50 {result['p50_latency_sec']:.2f}s  "
                        f"p99 {result.get('p99_latency_sec', 0):.2f}s  "
                        f"failures {result.get('failed_requests', 0)}"
                    )
            finally:
                stop_gpu_monitor(monitor)

            if not runs:
                print(f"  ERROR: all runs failed for {label}")
                continue

            result = aggregate_runs(runs)
            result.update(
                {
                    "strategy": args.strategy,
                    "load_config": cfg["name"],
                    "vllm_model": model_key if args.strategy in ("vllm", "sla") else None,
                    "sla_policy": "reject_e2e_v2" if args.strategy == "sla" else None,
                    "tier": "capstone" if args.capstone else "base",
                    "config": cfg,
                }
            )
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"load_{label}_{ts}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            saved += 1
            print(
                f"  mean req/s: {result['req_per_sec_mean']:.3f} ± {result['req_per_sec_std']:.3f}  "
                f"mean p50: {result['p50_latency_sec_mean']:.2f}s  "
                f"mean p99: {result['p99_latency_sec_mean']:.2f}s  "
                f"saved {out_path.name}"
            )
        except Exception as exc:
            import traceback

            print(f"  FATAL for {label}: {exc}")
            traceback.print_exc()
            continue

    print(f"Done. Ran {len(loads)} configs for {args.strategy}. Saved {saved} JSON files.")
    if saved == 0:
        sys.exit(1)
    if saved < len(loads):
        print(f"WARN: only {saved}/{len(loads)} configs saved")


if __name__ == "__main__":
    main()
