"""Central configuration, loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    start_url: str = Field(default="https://books.toscrape.com/")
    max_pages: int = Field(default=500)
    max_depth: int = Field(default=10)
    crawl_concurrency: int = Field(default=8)
    crawl_delay_seconds: float = Field(default=0.2)
    request_timeout: float = Field(default=20.0)
    user_agent: str = Field(default="internal-linker/1.0 (+SEO internal audit bot)")

    use_mock_data: bool = Field(default=True)
    launch_date: Optional[date] = Field(default=None)
    matomo_base_url: str = Field(default="")
    matomo_site_id: str = Field(default="1")
    matomo_token: str = Field(default="")
    matomo_period_days: int = Field(default=30)

    @field_validator("launch_date", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    at_risk_max_inbound: int = Field(default=2)
    bm25_shortlist_size: int = Field(default=20)
    rrf_k: int = Field(default=60)
    placement_floor: int = Field(default=10)
    placement_ceiling: int = Field(default=30)
    anchor_match_threshold: float = Field(default=80.0)
    orphan_boost: float = Field(default=1.15)

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    chroma_path: str = Field(default="./chroma_store")
    chroma_collection: str = Field(default="page_embeddings")

    decay_traffic_weight: float = Field(default=0.6)
    decay_staleness_weight: float = Field(default=0.4)
    evergreen_min_words: int = Field(default=1500)
    evergreen_trend_band: float = Field(default=10.0)
    evergreen_min_visits: int = Field(default=50)

    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    snapshot_dir: str = Field(default="./snapshots")
    report_dir: str = Field(default="./reports")
    slack_webhook_url: str = Field(default="")


settings = Settings()
