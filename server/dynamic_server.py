import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_WINDOW_SEC = 0.02

model = None
tokenizer = None
pending: List[tuple] = []
pending_lock = asyncio.Lock()
worker_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, worker_task
    print("Starting... loading model (this may take 1-2 min)...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    worker_task = asyncio.create_task(batch_worker())
    print("\n  >>> Open in browser: http://127.0.0.1:8000\n")
    yield
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Systems-Level Optimization of LLM Inference on a Single Consumer GPU — Dynamic", lifespan=lifespan)


@app.get("/")
async def custom_docs():
    path = Path(__file__).parent / "static" / "custom_docs.html"
    return FileResponse(path)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: Optional[float] = 0.7


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    num_tokens: int


def _run_batch_sync(prompts: List[str], max_new_tokens: int, temperature: float = 0.7) -> List[GenerateResponse]:
    if not prompts:
        return []
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    results = []
    for i in range(len(prompts)):
        start = input_ids.shape[1]
        generated = outputs[i, start:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        results.append(GenerateResponse(text=text.strip(), prompt=prompts[i], num_tokens=len(generated)))
    return results


async def batch_worker():
    global pending
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(BATCH_WINDOW_SEC)
        async with pending_lock:
            batch = pending
            pending = []
        if not batch:
            continue
        reqs = [r for r, _ in batch]
        prompts = [r.prompt for r in reqs]
        max_tok = max(r.max_new_tokens for r in reqs)
        temp = reqs[0].temperature or 0.7
        try:
            results = await loop.run_in_executor(
                None,
                lambda: _run_batch_sync(prompts, max_tok, temp),
            )
        except Exception as e:
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)
            continue
        for (req, fut), resp in zip(batch, results):
            if not fut.done():
                fut.set_result(resp)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    async with pending_lock:
        pending.append((req, fut))
    return await fut


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(next(model.parameters()).device) if model else "not loaded"}
