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
2. Look at the previews and call read_chunk on the promising candidates to read
   the full text before generating the information.
3. Repeat step 1 and 2 if needed.
4. Be extractive; do not invent details that are not in the document.
5. When you are done, return the extracted information as JSON (no preamble, no explanation) in the following format: {{"info": "<extracted text>", "cited_chunk_indices": [<int>, ...]}}

"""


# Per-attribute guidance appended to IE_PROMPT. Only attributes that benefit from
# extra direction need an entry; the rest fall back to the generic strategy above.
# Keys must match core.constants.TARGET_ATTRIBUTES.
ATTRIBUTE_HINTS = {
    "primary_study_objectives": "Usually stated in the synopsis or an 'Objectives and Endpoints' section. "
    "Return the primary objective(s) as concise prose, verbatim where possible.",
    "secondary_study_objective": "Found alongside the primary objectives in the 'Objectives and Endpoints' section. "
    "Return all secondary objectives as a list, separate from the primary ones.",
    "primary_endpoints": "Found in the 'Objectives and Endpoints' / 'Estimands and Endpoints' section. "
    "Capture the full endpoint definition including the measure and timepoint.",
    "exploratory_endpoints": "Often a distinct subsection after primary/secondary endpoints. "
    "Return only the exploratory endpoints; do not mix in primary or secondary ones.",
    "schedule_of_activities_table": "This lives in the Schedule of Activities (SoA) section. Return it as a structured "
    "table (visits as columns, procedures/assessments as rows), preserving the cell markers "
    "(e.g. X) and footnotes. Do not summarise it into prose.",
    "inclusion_criteria": "Found in the eligibility / study population section. Return the complete numbered or "
    "bulleted list of inclusion criteria verbatim; do not paraphrase or omit items.",
    "exclusion_criteria": "Found in the eligibility / study population section, immediately after the inclusion "
    "criteria. Return the complete list verbatim; do not include inclusion criteria.",
    "visit_definition": "Look for definitions of the study visits (e.g. Screening, Baseline, Treatment, "
    "Follow-up). Return what each visit is and what it covers.",
    "visit_timing": "Look for the timing/windows of each visit (study days/weeks and allowed windows, "
    "e.g. 'Day 1 ± 3 days'). Return the timing per visit.",
    "Key_assessments_and_procedures": "Look across the SoA and procedure sections for the assessments and procedures performed "
    "(e.g. labs, imaging, PK sampling, questionnaires). Return them as a list.",
    "safety_monitoring_rules": "Often spread across safety, dose-modification, and DSMB/committee sections. "
    "Capture stopping rules, dose modifications, and oversight/monitoring rules.",
}


def build_ie_prompt(attribute_key: str) -> str:
    hint = ATTRIBUTE_HINTS.get(attribute_key, "")
    if not hint:
        return IE_PROMPT
    return f"{IE_PROMPT}\nAttribute-specific guidance:\n{hint}\n"
