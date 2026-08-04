"""Central configuration, loaded from environment / .env via pydantic-settings.

Every tunable lives here. Modules read settings from this object and never from
raw os.environ. The one switch that matters most operationally is USE_MOCK_DATA.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Crawl target -------------------------------------------------------
    start_url: str = Field(
        default="https://books.toscrape.com/",
        description="Root URL to crawl. Pre-launch this is the dummy test site.",
    )
    max_pages: int = Field(default=500, description="Safety cap on crawl size.")
    max_depth: int = Field(default=10, description="Max link depth from start URL.")
    crawl_concurrency: int = Field(default=8, description="Parallel in-flight requests.")
    crawl_delay_seconds: float = Field(default=0.2, description="Politeness delay per request.")
    request_timeout: float = Field(default=20.0)
    user_agent: str = Field(default="internal-linker/1.0 (+SEO internal audit bot)")

    # --- Data signal --------------------------------------------------------
    use_mock_data: bool = Field(
        default=True,
        description="True pre-launch (mock traffic). Flip to False at launch for real Matomo.",
    )
    launch_date: Optional[date] = Field(
        default=None,
        description="SS launch date. If set and today is past it, a mock-data run warns.",
    )
    matomo_base_url: str = Field(default="", description="Self-hosted Matomo URL.")
    matomo_site_id: str = Field(default="1")
    matomo_token: str = Field(default="", description="Matomo API auth token.")
    matomo_period_days: int = Field(default=30)

    # --- Orphan thresholds --------------------------------------------------
    at_risk_max_inbound: int = Field(
        default=2,
        description="Pages with inbound links <= this (and > 0) are at-risk.",
    )

    # --- Opportunity engine -------------------------------------------------
    bm25_shortlist_size: int = Field(default=20, description="Top-N per source before rerank.")
    rrf_k: int = Field(default=60, description="RRF damping constant.")
    placement_floor: int = Field(default=10, description="Below this, the run warns.")
    placement_ceiling: int = Field(default=30, description="Site-wide max placements per run.")
    anchor_match_threshold: float = Field(
        default=80.0,
        description="rapidfuzz score (0-100) required for an anchor to count as present.",
    )
    orphan_boost: float = Field(
        default=1.15,
        description="Multiplier applied to opportunities whose target is orphan/at-risk.",
    )

    # --- Embeddings ---------------------------------------------------------
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    chroma_path: str = Field(default="./chroma_store")
    chroma_collection: str = Field(default="page_embeddings")

    # --- Decay scoring ------------------------------------------------------
    decay_traffic_weight: float = Field(default=0.6)
    decay_staleness_weight: float = Field(default=0.4)
    evergreen_min_words: int = Field(default=1500)
    evergreen_trend_band: float = Field(default=10.0, description="+/- percent trend to count as stable.")
    evergreen_min_visits: int = Field(
        default=50,
        description="Traffic floor. Below this a page is NOT auto-protected as evergreen.",
    )

    # --- LLM (Groq) ---------------------------------------------------------
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    # --- Paths --------------------------------------------------------------
    snapshot_dir: str = Field(default="./snapshots")
    report_dir: str = Field(default="./reports")

    # --- Notifications ------------------------------------------------------
    slack_webhook_url: str = Field(default="")


settings = Settings()
