from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .vllm_engine import create_async_engine, generate_with_engine, get_model_key, get_model_spec

engine = None
model_spec = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, model_spec
    engine, model_spec = create_async_engine()
    print(f"  >>> vLLM server ready at http://127.0.0.1:8000  model={get_model_key()}\n", flush=True)
    yield
    engine = None
    model_spec = None


app = FastAPI(title="LLM Inference — vLLM", lifespan=lifespan)


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
    if engine is None:
        raise HTTPException(status_code=503, detail="vLLM engine is not loaded")

    try:
        text, num_tokens = await generate_with_engine(
            engine,
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateResponse(text=text, prompt=req.prompt, num_tokens=num_tokens)


@app.get("/health")
async def health():
    spec = get_model_spec() if model_spec else None
    return {
        "status": "ok" if engine else "not loaded",
        "backend": "vllm",
        "model": spec.key if spec else None,
        "model_id": spec.model_id if spec else None,
    }
