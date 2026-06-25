from __future__ import annotations

from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    embed_model: str = "BAAI/bge-large-en-v1.5"
    embed_dimension: int = 1024
    max_tokens: int = 6000
    sparse_model: str = "Qdrant/bm25"

    ollama_model: str = "gemma4"
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192
    ollama_seed: int = 42

    qdrant_collection: str = "clinical_chunks"

    apply_reranking: bool = False
    apply_query_expansion: bool = True

    agent_search_k: int = 5
    agent_debug: bool = False

    # Langfuse LLM observability — points at a self-hosted (local) Langfuse by
    # default; tracing turns on only when both keys are set.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    data_dir: Path = ROOT / "data"

    @computed_field
    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @computed_field
    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @computed_field
    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @computed_field
    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    def ensure_dirs(self) -> None:
        for d in (self.upload_dir, self.output_dir, self.qdrant_path.parent):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
