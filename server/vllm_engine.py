from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    max_model_len: int
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 64


MODELS: dict[str, ModelSpec] = {
    "tinyllama": ModelSpec(
        key="tinyllama",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_model_len=2048,
        max_num_seqs=64,
    ),
    "zephyr-7b": ModelSpec(
        key="zephyr-7b",
        model_id="HuggingFaceH4/zephyr-7b-beta",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
        max_num_seqs=32,
    ),
    "mistral-7b": ModelSpec(
        key="mistral-7b",
        model_id="HuggingFaceH4/zephyr-7b-beta",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
        max_num_seqs=32,
    ),
    "mistral-7b-official": ModelSpec(
        key="mistral-7b-official",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
        max_num_seqs=32,
    ),
}


def get_model_key() -> str:
    key = os.environ.get("VLLM_MODEL", "tinyllama").strip().lower()
    if key not in MODELS:
        valid = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown VLLM_MODEL={key!r}. Choose one of: {valid}")
    return key


def _patch_transformers_tokenizer_compat() -> None:
    """Colab may ship transformers 5.x; vLLM 0.8.x still reads this attribute."""
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(
                lambda self: self.all_special_tokens
            )
    except Exception:
        pass


def get_model_spec() -> ModelSpec:
    return MODELS[get_model_key()]


def create_async_engine():
    _patch_transformers_tokenizer_compat()
    try:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
    except ImportError as exc:
        raise RuntimeError(f"vLLM import failed: {exc}") from exc

    spec = get_model_spec()
    args = AsyncEngineArgs(
        model=spec.model_id,
        dtype="float16",
        gpu_memory_utilization=spec.gpu_memory_utilization,
        max_model_len=spec.max_model_len,
        max_num_seqs=spec.max_num_seqs,
        trust_remote_code=True,
    )
    engine = AsyncLLMEngine.from_engine_args(args)
    print(f"\n  >>> vLLM engine loaded: {spec.key} ({spec.model_id})\n", flush=True)
    return engine, spec


async def generate_with_engine(
    engine,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: Optional[float] = 0.7,
) -> tuple[str, int]:
    import time

    from vllm import SamplingParams

    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if temperature and temperature > 0 else 0.0,
    )
    request_id = f"request-{time.time_ns()}"
    final_output = None
    async for output in engine.generate(prompt, sampling, request_id):
        final_output = output

    if final_output is None:
        raise RuntimeError("vLLM returned no output")

    generated = final_output.outputs[0]
    return generated.text.strip(), len(generated.token_ids)
