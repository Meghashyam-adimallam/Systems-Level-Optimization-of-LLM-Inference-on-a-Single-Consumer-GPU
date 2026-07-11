"""
Kernel-level timing with torch.profiler (v2 / v3 evidence).

  # v2 — TinyLlama, batch of 4 (matches static batching shape)
  python scripts/profile_inference.py --model tinyllama --batch-size 4 --out-dir v2/results/profiles

  # v3 — Zephyr-7B single-sequence decode (HF reference; run on A100)
  python scripts/profile_inference.py --model zephyr-7b --batch-size 1 --out-dir v3/results/profiles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_IDS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
}

PROMPT = "Explain why batching improves GPU utilization during LLM decode."


def run_profile(model_key: str, batch_size: int, max_new_tokens: int, out_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for profiler run (use Colab A100 or local NVIDIA GPU).")

    model_id = MODEL_IDS[model_key]
    device = "cuda"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    prompts = [PROMPT] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # Warmup
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("model_generate"):
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_key}_bs{batch_size}_tok{max_new_tokens}"
    trace_path = out_dir / f"chrome_trace_{tag}.json"
    table_path = out_dir / f"kernel_table_{tag}.txt"
    summary_path = out_dir / f"summary_{tag}.json"

    prof.export_chrome_trace(str(trace_path))

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    table_path.write_text(table, encoding="utf-8")

    cuda_events = [
        {
            "name": e.key,
            "cuda_time_us": e.cuda_time_total,
            "cpu_time_us": e.cpu_time_total,
            "count": e.count,
        }
        for e in prof.key_averages()
        if e.cuda_time_total > 0
    ]
    cuda_events.sort(key=lambda x: x["cuda_time_us"], reverse=True)
    top = cuda_events[:15]
    summary = {
        "model": model_key,
        "model_id": model_id,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "top_cuda_kernels": top,
        "chrome_trace": trace_path.name,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(table)
    print(f"\nSaved chrome trace → {trace_path}")
    print(f"Saved kernel table → {table_path}")
    print(f"Saved summary JSON → {summary_path}")
    print("Open chrome_trace_*.json in chrome://tracing or Perfetto UI.")


def main():
    parser = argparse.ArgumentParser(description="torch.profiler decode breakdown")
    parser.add_argument("--model", choices=sorted(MODEL_IDS), default="tinyllama")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="results/profiles")
    args = parser.parse_args()
    run_profile(args.model, args.batch_size, args.max_new_tokens, Path(args.out_dir))


if __name__ == "__main__":
    main()
