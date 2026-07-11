"""
Generate benchmark bar charts per tier (v1 / v2 / v3) and a combined summary.

  python scripts/generate_tier_charts.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COMBINED_DIR = ROOT / "combined"

COLORS = {
    "baseline": "#2d7d46",
    "batched": "#1a5fb4",
    "dynamic": "#c64600",
    "vllm": "#613583",
    "sla": "#7d3c98",
    "llamacpp": "#1c71d8",
}

METRICS = [
    ("req_per_sec", "Throughput", "Requests / sec"),
    ("p50_latency_sec", "p50 Latency", "Seconds"),
    ("p95_latency_sec", "p95 Latency", "Seconds"),
    ("tokens_per_sec", "Token throughput", "Tokens / sec"),
]


def metric_value(data: dict, key: str) -> float:
    if f"{key}_mean" in data:
        return float(data[f"{key}_mean"])
    return float(data.get(key, 0))


def metric_std(data: dict, key: str) -> float:
    return float(data.get(f"{key}_std", 0))


def parse_label(stem: str) -> Tuple[str, str]:
    raw = stem.replace("load_", "")
    raw = re.sub(r"_\d{8}_\d{6}$", "", raw)
    for load in ("long_context", "light", "medium", "heavy", "full"):
        if raw.endswith(f"_{load}"):
            strategy = raw[: -(len(load) + 1)]
            return strategy, load
    return raw, "unknown"


def load_tier_results(results_dir: Path) -> Dict[str, Dict[str, dict]]:
    by_load: Dict[str, Dict[str, dict]] = {}
    if not results_dir.exists():
        return by_load

    latest: Dict[Tuple[str, str], Path] = {}
    for path in sorted(results_dir.glob("load_*.json")):
        strategy, load = parse_label(path.stem)
        if load == "full":
            continue
        key = (strategy, load)
        latest[key] = path

    for (strategy, load), path in latest.items():
        with path.open() as f:
            data = json.load(f)
        by_load.setdefault(load, {})[strategy] = data
    return by_load


def plot_tier_comparison(
    by_load: Dict[str, Dict[str, dict]],
    strategies: List[str],
    load_order: List[str],
    title: str,
    out_path: Path,
) -> bool:
    configs = [c for c in load_order if c in by_load]
    if not configs:
        return False

    n_configs = len(configs)
    n_strat = len(strategies)
    x = np.arange(n_configs)
    width = 0.8 / max(n_strat, 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    for i, (metric_key, metric_title, ylabel) in enumerate(METRICS):
        ax = axes[i // 2, i % 2]
        for j, strat in enumerate(strategies):
            vals = [metric_value(by_load[c].get(strat, {}), metric_key) for c in configs]
            errs = [metric_std(by_load[c].get(strat, {}), metric_key) for c in configs]
            if not any(vals):
                continue
            ax.bar(
                x + j * width,
                vals,
                width,
                yerr=errs if any(errs) else None,
                capsize=3,
                label=strat.capitalize(),
                color=COLORS.get(strat, "gray"),
            )
        ax.set_ylabel(ylabel)
        ax.set_title(metric_title)
        ax.set_xticks(x + width * (n_strat - 1) / 2)
        ax.set_xticklabels([c.replace("_", " ").title() for c in configs])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return True


def plot_kv_cache(results_dir: Path, out_path: Path) -> bool:
    kv_files = sorted(results_dir.glob("kv_cache_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not kv_files:
        return False
    with kv_files[0].open() as f:
        kv_data = json.load(f)
    by_tokens = kv_data.get("by_tokens") or {}
    if not by_tokens:
        return False

    tokens_list = [k for k in ("32", "64", "128") if k in by_tokens]
    x = np.arange(len(tokens_list))
    bar_w = 0.35
    true_vals = [by_tokens[t]["use_cache_true"]["tokens_per_sec"] for t in tokens_list]
    false_vals = [by_tokens[t]["use_cache_false"]["tokens_per_sec"] for t in tokens_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - bar_w / 2, true_vals, bar_w, label="use_cache=True", color="#2d7d46")
    ax.bar(x + bar_w / 2, false_vals, bar_w, label="use_cache=False", color="#7d2525")
    ax.set_ylabel("Tokens / sec")
    ax.set_title("KV cache: use_cache True vs False")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} tok" for t in tokens_list])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return True


def plot_gpu_traces(trace_dir: Path, report_dir: Path, prefix: str = "") -> int:
    if not trace_dir.exists():
        return 0
    count = 0
    for trace in sorted(trace_dir.glob("gpu_*.csv")):
        times, utils, mems = [], [], []
        with trace.open() as f:
            import csv

            for row in csv.DictReader(f):
                times.append(float(row["wall_time_sec"]))
                utils.append(float(row["gpu_util_pct"]))
                mems.append(float(row["memory_used_mb"]))
        if not times:
            continue

        label = trace.stem.replace("gpu_", "")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"GPU telemetry — {label}", fontweight="bold")
        axes[0].plot(times, utils, color="#c64600", linewidth=1.8)
        axes[0].set_ylabel("GPU util (%)")
        axes[0].set_ylim(0, 100)
        axes[0].grid(alpha=0.3)
        axes[1].plot(times, mems, color="#1a5fb4", linewidth=1.8)
        axes[1].set_ylabel("VRAM used (MB)")
        axes[1].set_xlabel("Time (sec)")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        out = report_dir / f"{prefix}{trace.stem}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        count += 1
    return count


def headline_medium(by_load: Dict[str, Dict[str, dict]], strategy: str) -> Optional[float]:
    medium = by_load.get("medium", {}).get(strategy)
    if not medium:
        return None
    return metric_value(medium, "req_per_sec")


def plot_all_tiers_summary(tier_headlines: List[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Project achievements by tier (medium load throughput)", fontweight="bold", y=1.02)

    for ax, tier in zip(axes, tier_headlines):
        labels = tier["labels"]
        values = tier["values"]
        colors = [COLORS.get(l.lower(), "gray") for l in labels]
        x = np.arange(len(labels))
        bars = ax.bar(x, values, 0.55, color=colors)
        ax.set_title(tier["title"], fontsize=10)
        ax.set_ylabel("req/s")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(0.02, 0.98, tier["subtitle"], transform=ax.transAxes, va="top", fontsize=8, color="#444")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_progression_chart(tier_headlines: List[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    tiers = [t["tier_name"] for t in tier_headlines]
    baseline = [t.get("baseline_req_s") or 0 for t in tier_headlines]
    best = [t.get("best_req_s") or 0 for t in tier_headlines]
    x = np.arange(len(tiers))
    w = 0.35
    ax.bar(x - w / 2, baseline, w, label="Baseline / vLLM-only ref", color="#2d7d46")
    ax.bar(x + w / 2, best, w, label="Best strategy in tier", color="#c64600")
    ax.set_ylabel("req/s (medium load)")
    ax.set_title("Throughput progression across tiers", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for i, (b, best_v) in enumerate(zip(baseline, best)):
        if b:
            ax.text(i - w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
        if best_v:
            ax.text(i + w / 2, best_v, f"{best_v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    tier_specs = [
        {
            "id": "v1",
            "dir": ROOT / "v1",
            "strategies": ["baseline", "batched", "dynamic"],
            "loads": ["light", "medium", "heavy"],
            "title": "v1 — RTX 4060, TinyLlama 1.1B (baseline / batched / dynamic)",
            "subtitle": "RTX 4060 8GB · TinyLlama 1.1B",
            "tier_name": "v1 Base",
            "baseline_strategy": "baseline",
            "best_strategy": "dynamic",
        },
        {
            "id": "v2",
            "dir": ROOT / "v2",
            "strategies": ["baseline", "batched", "dynamic"],
            "loads": ["light", "medium", "heavy"],
            "title": "v2 — A100, TinyLlama 1.1B reproducibility (3 runs, mean ± std)",
            "subtitle": "A100 40GB · TinyLlama 1.1B",
            "tier_name": "v2 A100 1.1B",
            "baseline_strategy": "baseline",
            "best_strategy": "dynamic",
        },
        {
            "id": "v3",
            "dir": ROOT / "v3",
            "strategies": ["vllm", "sla"],
            "loads": ["light", "medium", "heavy", "long_context"],
            "title": "v3 — A100, Zephyr-7B capstone (vLLM vs SLA scheduler)",
            "subtitle": "A100 40GB · Zephyr-7B · vLLM + SLA",
            "tier_name": "v3 Capstone 7B",
            "baseline_strategy": "vllm",
            "best_strategy": "vllm",
        },
    ]

    headlines = []
    for spec in tier_specs:
        results_dir = spec["dir"] / "results"
        report_dir = spec["dir"] / "report"
        by_load = load_tier_results(results_dir)

        plot_tier_comparison(
            by_load,
            spec["strategies"],
            spec["loads"],
            spec["title"],
            report_dir / "benchmark_comparison.png",
        )
        plot_kv_cache(results_dir, report_dir / "kv_cache_comparison.png")
        n_gpu = plot_gpu_traces(results_dir / "gpu_traces", report_dir)
        if n_gpu:
            print(f"  {spec['id']}: {n_gpu} GPU trace charts")

        medium = by_load.get("medium", {})
        strat_labels = []
        strat_vals = []
        for s in spec["strategies"]:
            if s in medium:
                strat_labels.append(s.capitalize())
                strat_vals.append(metric_value(medium[s], "req_per_sec"))

        headlines.append(
            {
                "tier_name": spec["tier_name"],
                "title": spec["tier_name"],
                "subtitle": spec["subtitle"],
                "labels": strat_labels,
                "values": strat_vals,
                "baseline_req_s": headline_medium(by_load, spec["baseline_strategy"]),
                "best_req_s": headline_medium(by_load, spec["best_strategy"]),
            }
        )

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    plot_all_tiers_summary(headlines, COMBINED_DIR / "tier_achievements_summary.png")
    plot_progression_chart(headlines, COMBINED_DIR / "tier_progression.png")


if __name__ == "__main__":
    main()
