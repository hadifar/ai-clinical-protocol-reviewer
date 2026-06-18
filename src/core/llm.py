from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel

from core.config import settings


@lru_cache(maxsize=1)
def get_ollama():
    import ollama

    return ollama.Client(host=settings.ollama_base_url)


def generate_structured[T: BaseModel](prompt: str, schema: type[T]) -> T:
    resp = get_ollama().generate(
        model=settings.ollama_model,
        prompt=prompt,
        options={"num_ctx": settings.ollama_num_ctx},
        format=schema.model_json_schema(),
    )
    return schema.model_validate_json(resp.response)
