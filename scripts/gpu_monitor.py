from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

QUERY_FIELDS = [
    "timestamp",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
]


def sample_nvidia_smi() -> Optional[Dict[str, str]]:
    query = ",".join(QUERY_FIELDS)
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    row = out.strip().splitlines()[0]
    values = [v.strip() for v in row.split(",")]
    if len(values) != len(QUERY_FIELDS):
        return None
    return dict(zip(QUERY_FIELDS, values))


def monitor(label: str, out_dir: Path, interval_sec: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"gpu_{label}_{ts}.csv"
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wall_time_sec",
                "label",
                "timestamp",
                "gpu_util_pct",
                "memory_used_mb",
                "memory_total_mb",
                "power_draw_w",
            ],
        )
        writer.writeheader()
        start = time.perf_counter()

        while running:
            sample = sample_nvidia_smi()
            if sample is not None:
                writer.writerow(
                    {
                        "wall_time_sec": round(time.perf_counter() - start, 3),
                        "label": label,
                        "timestamp": sample["timestamp"],
                        "gpu_util_pct": sample["utilization.gpu"],
                        "memory_used_mb": sample["memory.used"],
                        "memory_total_mb": sample["memory.total"],
                        "power_draw_w": sample["power.draw"],
                    }
                )
                f.flush()
            time.sleep(interval_sec)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record GPU telemetry to CSV")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", default="results/gpu_traces")
    parser.add_argument("--interval-sec", type=float, default=0.5)
    args = parser.parse_args()

    if sample_nvidia_smi() is None:
        raise SystemExit("nvidia-smi not available")

    path = monitor(args.label, Path(args.out_dir), args.interval_sec)
    print(f"Saved GPU telemetry to {path}")


if __name__ == "__main__":
    main()
