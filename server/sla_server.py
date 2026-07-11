from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .vllm_engine import create_async_engine, generate_with_engine, get_model_key, get_model_spec

engine = None
model_spec = None
request_queue: Optional[asyncio.Queue] = None
worker_task: Optional[asyncio.Task] = None
latency_window: Deque[float] = deque(maxlen=int(os.environ.get("SLA_LATENCY_WINDOW", "50")))
stats = {
    "completed": 0,
    "rejected_queue_full": 0,
    "rejected_budget": 0,
    "rejected_stale": 0,
    "rejected_timeout": 0,
    "errors": 0,
}


class SLARejected(Exception):
    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(detail)


@dataclass
class SLAConfig:
    max_queue_depth: int = int(os.environ.get("SLA_MAX_QUEUE", "32"))
    p95_budget_sec: float = float(os.environ.get("SLA_P95_BUDGET_SEC", "3.0"))
    request_timeout_sec: float = float(os.environ.get("SLA_REQUEST_TIMEOUT_SEC", "120"))
    min_samples_for_budget: int = int(os.environ.get("SLA_MIN_SAMPLES", "5"))


sla_config = SLAConfig()


def rolling_percentile(pct: float) -> float:
    if not latency_window:
        return 0.0
    return float(np.percentile(np.array(latency_window), pct))


def record_latency(latency_sec: float) -> None:
    latency_window.append(latency_sec)
    stats["completed"] += 1


def budget_exceeded() -> bool:
    return (
        len(latency_window) >= sla_config.min_samples_for_budget
        and rolling_percentile(95) > sla_config.p95_budget_sec
    )


async def queue_worker() -> None:
    assert engine is not None
    assert request_queue is not None

    while True:
        req, future, t_enqueued = await request_queue.get()
        if future.done():
            request_queue.task_done()
            continue

        queue_wait_sec = time.perf_counter() - t_enqueued
        if queue_wait_sec > sla_config.p95_budget_sec:
            stats["rejected_stale"] += 1
            if not future.done():
                future.set_exception(
                    SLARejected(
                        "stale",
                        f"queue wait {queue_wait_sec:.2f}s > {sla_config.p95_budget_sec}s budget",
                    )
                )
            request_queue.task_done()
            continue

        if budget_exceeded():
            stats["rejected_budget"] += 1
            if not future.done():
                future.set_exception(
                    SLARejected(
                        "budget",
                        f"p95 budget exceeded ({rolling_percentile(95):.2f}s > {sla_config.p95_budget_sec}s)",
                    )
                )
            request_queue.task_done()
            continue

        try:
            text, num_tokens = await generate_with_engine(
                engine,
                req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
            )
            record_latency(time.perf_counter() - t_enqueued)
            if not future.done():
                future.set_result(
                    GenerateResponse(text=text, prompt=req.prompt, num_tokens=num_tokens)
                )
        except Exception as exc:
            stats["errors"] += 1
            if not future.done():
                future.set_exception(exc)
        finally:
            request_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, model_spec, request_queue, worker_task, latency_window

    engine, model_spec = create_async_engine()
    request_queue = asyncio.Queue()
    latency_window = deque(maxlen=int(os.environ.get("SLA_LATENCY_WINDOW", "50")))
    worker_task = asyncio.create_task(queue_worker())
    print(
        f"  >>> SLA+vLLM ready at http://127.0.0.1:8000  "
        f"model={get_model_key()} ({model_spec.model_id})  "
        f"p95_budget={sla_config.p95_budget_sec}s\n",
        flush=True,
    )
    yield

    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
    engine = None
    model_spec = None
    request_queue = None
    worker_task = None


app = FastAPI(title="LLM Inference — SLA + vLLM", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: Optional[float] = 0.7


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    num_tokens: int


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if engine is None or request_queue is None:
        raise HTTPException(status_code=503, detail="SLA server is not ready")

    if request_queue.qsize() >= sla_config.max_queue_depth:
        stats["rejected_queue_full"] += 1
        raise HTTPException(status_code=503, detail="queue full")

    if budget_exceeded():
        stats["rejected_budget"] += 1
        raise HTTPException(
            status_code=503,
            detail=f"p95 budget exceeded ({rolling_percentile(95):.2f}s > {sla_config.p95_budget_sec}s)",
        )

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    t_enqueued = time.perf_counter()
    await request_queue.put((req, future, t_enqueued))

    try:
        return await asyncio.wait_for(future, timeout=sla_config.request_timeout_sec)
    except SLARejected as exc:
        raise HTTPException(status_code=503, detail=exc.detail) from exc
    except asyncio.TimeoutError:
        stats["rejected_timeout"] += 1
        if not future.done():
            future.cancel()
        raise HTTPException(status_code=504, detail="request timeout")


@app.post("/admin/reset_window")
async def reset_window():
    """Clear rolling latency window between benchmark load configs."""
    latency_window.clear()
    return {
        "ok": True,
        "rolling_p95_sec": rolling_percentile(95),
        "latency_window_size": len(latency_window),
    }


@app.get("/health")
async def health():
    spec = get_model_spec() if model_spec else None
    return {
        "status": "ok" if engine else "not loaded",
        "backend": "sla+vllm",
        "model": spec.key if spec else None,
        "model_id": spec.model_id if spec else None,
        "queue_depth": request_queue.qsize() if request_queue else 0,
        "max_queue_depth": sla_config.max_queue_depth,
        "p95_budget_sec": sla_config.p95_budget_sec,
        "rolling_p95_sec": rolling_percentile(95),
    }


@app.get("/metrics")
async def metrics():
    return {
        **stats,
        "queue_depth": request_queue.qsize() if request_queue else 0,
        "rolling_p50_sec": rolling_percentile(50),
        "rolling_p95_sec": rolling_percentile(95),
        "rolling_p99_sec": rolling_percentile(99),
        "latency_window_size": len(latency_window),
        "p95_budget_sec": sla_config.p95_budget_sec,
        "budget_exceeded": budget_exceeded(),
        "model": get_model_key() if model_spec else None,
        "model_id": model_spec.model_id if model_spec else None,
    }
