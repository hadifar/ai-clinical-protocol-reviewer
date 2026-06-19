from __future__ import annotations

from typing import Any

from langchain.agents.structured_output import ToolStrategy

from core.config import settings
from core.llm import get_model
from core.prompts import IE_PROMPT
from core.vectorstore import get_chunk
from models.schemas import IEAgentResponse

_PREVIEW_N_WORDS = 120


def _build_tools():
    from langchain_core.tools import tool

    from services.ranking_service import search

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


def _exception_handler(inputs: Any) -> dict:
    from langchain_core.messages import AIMessage

    return {
        "messages": [
            *inputs["messages"],
            AIMessage(content="Process failed..."),
        ],
        "structured_response": IEAgentResponse(info="", cited_chunk_indices=[]),
    }


def invoke_agent(attribute: str) -> tuple[str, list]:
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableLambda

    agent = (
        create_agent(
            model=get_model(),
            tools=_build_tools(),
            system_prompt=IE_PROMPT,
            response_format=ToolStrategy(IEAgentResponse),
            debug=True,
        )
        .with_retry(stop_after_attempt=3)
        .with_fallbacks([RunnableLambda(_exception_handler)])
    )

    state = agent.invoke(
        {"messages": [HumanMessage(content=f"attribute: {attribute}")]}
    )

    return state["structured_response"], state["messages"]
