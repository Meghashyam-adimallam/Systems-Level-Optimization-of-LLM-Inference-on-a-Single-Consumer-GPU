"""
Plot p50 / p95 / p99 tail latency by strategy and load from saved benchmark JSON.

  python scripts/plot_tail_latency.py --results-dir v2/results --out-dir v2/report
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

STRATEGIES = ["baseline", "batched", "dynamic", "vllm", "sla", "llamacpp"]
LOADS = ["light", "medium", "heavy", "long_context"]
PERCENTILES = [
    ("p50_latency_sec", "p50"),
    ("p95_latency_sec", "p95"),
    ("p99_latency_sec", "p99"),
]


def metric_from_blob(blob: dict, key: str) -> Optional[float]:
    if f"{key}_mean" in blob:
        return float(blob[f"{key}_mean"])
    if key in blob:
        return float(blob[key])
    return None


def latest_per_strategy_load(results_dir: Path) -> Dict[Tuple[str, str], dict]:
    found: Dict[Tuple[str, str], Tuple[float, dict]] = {}
    for path in sorted(results_dir.glob("load_*.json")):
        m = re.match(r"load_(\w+)_(light|medium|heavy|long_context)_", path.name)
        if not m:
            continue
        strategy, load = m.group(1), m.group(2)
        data = json.loads(path.read_text())
        mtime = path.stat().st_mtime
        key = (strategy, load)
        if key not in found or mtime > found[key][0]:
            found[key] = (mtime, data)
    return {k: v[1] for k, v in found.items()}


def plot_tail_latency(
    by_key: Dict[Tuple[str, str], dict],
    strategies: List[str],
    loads: List[str],
    title: str,
    out_path: Path,
) -> bool:
    available_loads = [load for load in loads if any((s, load) in by_key for s in strategies)]
    if not available_loads:
        return False

    fig, axes = plt.subplots(1, len(available_loads), figsize=(4 * len(available_loads), 4), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    for ax, load in zip(axes[0], available_loads):
        x = np.arange(len(PERCENTILES))
        width = 0.8 / max(len(strategies), 1)
        plotted = 0
        for i, strat in enumerate(strategies):
            blob = by_key.get((strat, load))
            if not blob:
                continue
            vals = [metric_from_blob(blob, k) or 0 for k, _ in PERCENTILES]
            offset = (i - len(strategies) / 2) * width + width / 2
            ax.bar(x + offset, vals, width, label=strat, color=colors[i])
            plotted += 1
        if not plotted:
            continue
        ax.set_title(load)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in PERCENTILES])
        ax.set_ylabel("seconds")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--tier", type=str, default="", help="v1 / v2 / v3 for chart title")
    parser.add_argument(
        "--strategies",
        type=str,
        default="",
        help="comma-separated; default auto from JSON filenames",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    by_key = latest_per_strategy_load(results_dir)

    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    else:
        strategies = sorted({s for s, _ in by_key.keys() if s in STRATEGIES})

    title = f"{args.tier} — tail latency (p50 / p95 / p99)" if args.tier else "Tail latency (p50 / p95 / p99)"
    plot_tail_latency(by_key, strategies, LOADS, title, out_dir / "tail_latency_comparison.png")


if __name__ == "__main__":
    main()
