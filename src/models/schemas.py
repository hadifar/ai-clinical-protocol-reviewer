from pydantic import BaseModel, Field


class Query(BaseModel):
    query: str


class Relevance(BaseModel):
    relevance: int = Field(ge=0, le=10)
