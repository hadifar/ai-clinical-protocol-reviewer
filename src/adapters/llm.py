from __future__ import annotations

from functools import lru_cache
from typing import cast

from pydantic import BaseModel

from config import settings


@lru_cache(maxsize=1)
def get_model():

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
        temperature=0,
        seed=settings.ollama_seed,
    )


def generate_structured[T: BaseModel](prompt: str, schema: type[T]) -> T:
    result = get_model().with_structured_output(schema).invoke(prompt)
    return cast(T, result)
