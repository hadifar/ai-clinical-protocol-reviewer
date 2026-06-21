DOC2QUERY_PROMPT = """You are tasked with generating a single sentence summary.
The summary MUST be concise and mainly reflect the title.

Requirements:
- Do not add any new information.
- Do not include explanations, comments, or extra text.
- Always start the summary with 'The section details'


Return ONLY valid JSON in the following format:
{{"summary": "<generated summary>"}}

TITLE:
{title}

TEXT CHUNK:
{section}


GENERATED SUMMARY:
"""


RERANK_PROMPT = """You are a reranking agent.

Rate how well the SECTION below lets to extract the information about the QUERY, using an integer score from 0 to 10:
- 0  = unrelated; garbage or noisy content; no information is here.
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
You are given a single attribute to collect information about. Your job is to find the precise piece of
information regarding that attribute in the indexed document.

Strategy:
1. Use search_chunks with a focused query derived from the attribute.
2. Look at the previews and call read_chunk on the most promising candidate(s) to read
   the full text before answering.
3. Be extractive; do not invent details that are not in the document.
"""
