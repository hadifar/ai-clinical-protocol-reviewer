from __future__ import annotations

from functools import lru_cache
from typing import cast

from pydantic import BaseModel

from core.config import settings
from core.vectorstore import get_chunk
from models.schemas import IEAgentResponse

_PREVIEW_N_WORDS = 120


SYSTEM_PROMPT = """You are an information extraction agent for clinical trial protocols.
You are given a single attribute or query. Your job is to find the precise piece of
information regarding that attribute in the indexed document.

You have two tools:
- search_chunks(query: str): semantic search over the document. Returns candidate chunks
  (chunk_index, short preview).
- read_chunk(chunk_index): returns the full text of the chunk with that chunk_index.

Strategy:
1. Call search_chunks with a focused query derived from the attribute.
2. Look at the previews and call read_chunk on the most promising candidate(s) to read
   the full text before answering.
3. If needed, refine your query and repeat.
4. Be extractive; do not invent details that are not in the document.

When you are done, return the extracted information as JSON (no preamble, no explanation) in the following format:

{{"info": "<extracted text>", "cited_chunk_indices": [<int>, ...]}}

"""


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


def _invoke_exception_handler(inputs: dict) -> dict:
    """Fallback when the agent run raises: return the same state shape as a normal
    run (real message objects + structured_response) so callers don't crash."""
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
        system_prompt=SYSTEM_PROMPT,
        response_format=IEAgentResponse,
    )

    agent = agent.with_fallbacks([RunnableLambda(_invoke_exception_handler)])

    state = agent.invoke(
        {"messages": [HumanMessage(content=f"attribute: {attribute}")]}
    )
    messages = state["messages"]
    info = (messages[-1].content or "").strip()
    return info, messages
