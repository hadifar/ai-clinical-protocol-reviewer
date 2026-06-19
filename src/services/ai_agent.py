from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

from pydantic import BaseModel

from core.config import settings
from core.prompts import IE_PROMPT
from core.vectorstore import get_chunk
from models.schemas import IEAgentResponse

_PREVIEW_N_WORDS = 120


@lru_cache(maxsize=1)
def get_model():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
        temperature=0,
    )


def generate_structured[T: BaseModel](prompt: str, schema: type[T]) -> T:
    result = get_model().with_structured_output(schema).invoke(prompt)
    return cast(T, result)


def _build_tools():
    from langchain_core.tools import tool

    from services.rank import search

    @tool
    def search_chunks(query: str) -> str:
        """Semantic search over the indexed protocol.
        Returns the candidate chunks with a short preview each.
        """
        results = search(query, k=settings.agent_search_k)
        if not results:
            return "No matching chunks found."
        lines = []
        for r in results:
            preview = " ".join((r["original"] or "").split())[:_PREVIEW_N_WORDS]
            lines.append(f"chunk {r['chunk_index']} · {preview}…")
        return "\n".join(lines)

    @tool
    def read_chunk(chunk_index: int) -> str:
        """Return the full text of a chunk by its chunk_index."""
        text = get_chunk(chunk_index)
        if text is None:
            return f"No chunk found with chunk_index {chunk_index}."
        return text

    return [search_chunks, read_chunk]


def _invoke_exception_handler(inputs: Any) -> dict:
    from langchain_core.messages import AIMessage

    return {
        "messages": [
            *inputs["messages"],
            AIMessage(content="FAILED"),
        ]
    }


def invoke_agent(attribute: str) -> tuple[str, list]:
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableLambda

    agent = create_agent(
        model=get_model(),
        tools=_build_tools(),
        system_prompt=IE_PROMPT,
        response_format=IEAgentResponse,
    )

    agent = agent.with_fallbacks([RunnableLambda(_invoke_exception_handler)])

    state = agent.invoke(
        {"messages": [HumanMessage(content=f"attribute: {attribute}")]}
    )
    messages = state["messages"]
    info = (messages[-1].content or "").strip()
    return info, messages
