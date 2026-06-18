from __future__ import annotations

from core.config import settings
from core.vectorstore import get_chunk
from services.rank import search

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
3. Be extractive; do not invent details that are not in the document.

When you are done, reply with ONLY the extracted information as plain text (no preamble,
no markdown). If the information cannot be found in the document, reply with exactly:
NOT FOUND
"""

NOT_FOUND = "NOT FOUND"
_PREVIEW_CHARS = 120


def _build_tools():
    from langchain_core.tools import tool

    @tool
    def search_chunks(query: str) -> str:
        """Semantic search over the indexed protocol.

        Returns the candidate chunks with a short preview each. Use
        read_chunk(chunk_index) to read the full text of a promising candidate.
        """
        results = search(query, k=settings.agent_search_k)
        if not results:
            return "No matching chunks found."
        lines = []
        for r in results:
            preview = " ".join((r["original"] or "").split())[:_PREVIEW_CHARS]
            lines.append(
                f"chunk {r['chunk_index']} · {preview}…"
            )
        return "\n".join(lines)

    @tool
    def read_chunk(chunk_index: int) -> str:
        """Return the full text of a chunk by its chunk_index, fetched from the index."""
        text = get_chunk(chunk_index)
        if text is None:
            return f"No chunk found with chunk_index {chunk_index}."
        return text

    return [search_chunks, read_chunk]


def invoke_agent(attribute: str) -> tuple[str, list]:
    """Run the extraction agent for an attribute.

    Returns a tuple of (extracted_info, full message trace).
    """
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama

    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
        temperature=0,
    )
    agent = create_agent(
        model=model, tools=_build_tools(), system_prompt=SYSTEM_PROMPT
    )
    state = agent.invoke(
        {"messages": [HumanMessage(content=f"attribute: {attribute}")]}
    )
    messages = state["messages"]
    info = (messages[-1].content or "").strip()
    return info, messages


def extract(attribute: str) -> dict:
    """Extract information for an attribute from the index.

    Returns {attribute: info}.
    """
    info, _ = invoke_agent(attribute)
    return {attribute: info}
