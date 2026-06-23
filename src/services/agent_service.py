from __future__ import annotations

from typing import Any

from langchain.agents.structured_output import ProviderStrategy

from core.config import settings
from core.constants import TARGET_ATTRIBUTES
from core.llm import get_model
from core.prompts import build_ie_prompt
from core.vectorstore import get_chunk
from models.ai_types import IEAgentResponse

_PREVIEW_N_WORDS = 250


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
            header = (r.get("title") or "").strip()
            body = (r.get("summary") or r.get("section") or "").strip()
            preview = " ".join(f"{header} {body}".split())[:_PREVIEW_N_WORDS]
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


def invoke_agent(attribute_key: str) -> tuple[dict, list]:
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableLambda

    attribute = TARGET_ATTRIBUTES.get(attribute_key, attribute_key)

    agent = (
        create_agent(
            model=get_model(),
            tools=_build_tools(),
            system_prompt=build_ie_prompt(attribute_key),
            response_format=ProviderStrategy(IEAgentResponse),
            middleware=[
                ToolCallLimitMiddleware(
                    tool_name="search_chunks",
                    run_limit=2,
                ),
                ToolCallLimitMiddleware(
                    tool_name="read_chunk",
                    thread_limit=6,
                ),
            ],
            debug=settings.agent_debug,
        )
        .with_retry(stop_after_attempt=3)
        .with_fallbacks([RunnableLambda(_exception_handler)])
    )

    state = agent.invoke(
        {"messages": [HumanMessage(content=f"attribute: {attribute}")]}
    )

    # TODO: issue with low capability models
    if state.get("structured_response") is None:
        state["structured_response"] = IEAgentResponse(info="", cited_chunk_indices=[])

    response = state["structured_response"]
    result = {
        attribute_key: response.info,
        "cited_chunk_indices": response.cited_chunk_indices,
    }
    return result, state["messages"]
