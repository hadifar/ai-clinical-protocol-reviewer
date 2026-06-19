QUERY_GEN_PROMPT = """You are tasked with generating a single short query for a given chunk of text from a clinical trial protocol document.
The qu
ery MUST be concise and reflect the main idea of the chunk.
Ensure query reflects the title (denoted by ## <title>).

Requirements:
- Do not add, infer, or assume any new information.
- Do not include explanations, comments, or extra text.

Return ONLY valid JSON in the following format:
{{"query": "<generated query>"}}

CHUNK:
{section}


GENERATED QUERY:
"""

RERANK_PROMPT = """You are reranking agent.

Rate how well the SECTION below lets to extract the information about the QUERY, using an integer score from 0 to 10:
- 0  = unrelated; garbage or noisy text; no information is here.
- 3  = same general topic, but the specific information is missing.
- 7  = the information is present but partial, implicit, or mixed with unrelated content.
- 10 = the section explicitly and completely states the information about the query.

Judge only whether the answer is present, not how well written the section is.

Return ONLY valid JSON in the following format:
{{"relevance": <integer 0-10>}}

QUERY:
{query}

SECTION:
{section}

RELEVANCE:
"""

IE_PROMPT = """You are an information extraction agent for clinical trial protocols.
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
