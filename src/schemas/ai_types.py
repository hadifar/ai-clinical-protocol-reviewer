from pydantic import BaseModel, Field


class GeneratedSummaryResponse(BaseModel):
    summary: str


class RelevanceScoreResponse(BaseModel):
    relevance: int = Field(ge=0, le=10)


class IEAgentResponse(BaseModel):
    info: str
    cited_chunk_indices: list[int]
