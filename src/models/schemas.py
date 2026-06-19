from pydantic import BaseModel, Field


class Query(BaseModel):
    query: str


class Relevance(BaseModel):
    relevance: int = Field(ge=0, le=10)


class IEAgentResponse(BaseModel):
    info: str
    cited_chunk_indices: list[int]
