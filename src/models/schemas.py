from pydantic import BaseModel, Field


class Summary(BaseModel):
    summary: str


class Relevance(BaseModel):
    relevance: int = Field(ge=0, le=10)


class IEAgentResponse(BaseModel):
    "The IE agent should returned format"

    info: str
    cited_chunk_indices: list[int]
