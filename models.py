"""Shared Pydantic models.

This module is the single data contract across the whole pipeline. Every other
module imports its types from here. No module defines its own ad-hoc shapes.

The `EvidenceBlock` attached to each `LinkOpportunity` is the handoff contract
with Utkarsh's Stage 3 module. It is a typed model, not a loose dict, so the
fields and their types are guaranteed rather than guessed.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Orphan severity tiers."""

    CRITICAL = "critical"      # zero inbound internal links
    AT_RISK = "at_risk"        # 1-2 inbound internal links


class Page(BaseModel):
    """A single crawled page, after nav/footer stripping."""

    url: str
    title: str = ""
    h1: str = ""
    meta_description: str = ""
    body_text: str = ""
    internal_links: list[str] = Field(default_factory=list)
    word_count: int = 0

    def content_hash(self) -> str:
        """MD5 of title + body, used for ChromaDB cache invalidation."""
        import hashlib

        payload = (self.title + "\n" + self.body_text).encode("utf-8", "ignore")
        return hashlib.md5(payload).hexdigest()


class EvidenceBlock(BaseModel):
    """Typed evidence for a single link suggestion.

    This is the Stage 3 handoff schema. Utkarsh consumes exactly these fields.
    """

    rrf_score: float
    similarity_score: float
    bm25_score: float
    anchor_match_score: float = 0.0   # rapidfuzz confidence the anchor exists in source
    target_inbound_links: int = 0     # how starved the target currently is


class LinkOpportunity(BaseModel):
    """A recommended internal link: source should link to target."""

    source_url: str
    target_url: str
    suggested_anchor_text: str
    evidence: EvidenceBlock


class OrphanPage(BaseModel):
    """A page with too few inbound internal links."""

    url: str
    title: str = ""
    inbound_link_count: int = 0
    severity: Severity


class OrphanReport(BaseModel):
    """Full orphan output, split by tier."""

    critical: list[OrphanPage] = Field(default_factory=list)
    at_risk: list[OrphanPage] = Field(default_factory=list)
    total_pages: int = 0


class DecayScore(BaseModel):
    """Decay assessment for one page."""

    url: str
    title: str = ""
    score: float = 0.0
    traffic_score: float = 0.0
    content_staleness_score: float = 0.0
    is_evergreen: bool = False
    reason: str = ""


class RefreshBrief(BaseModel):
    """A content refresh brief, rendered into Content Factory format downstream."""

    url: str
    title: str = ""
    decay_score: float = 0.0
    reason: str = ""
    suggested_actions: list[str] = Field(default_factory=list)
    traffic_trend: float = 0.0   # percent change over the period


class TrafficStats(BaseModel):
    """Per-page traffic signals, matching the shape Matomo returns.

    The mock generator emits this exact shape so the launch swap is a flag flip.
    """

    url: str
    nb_visits: int = 0
    nb_hits: int = 0
    bounce_rate: float = 0.0          # 0.0 - 1.0
    avg_time_on_page: float = 0.0     # seconds
    trend_pct: float = 0.0            # percent change vs prior period, signed


class PipelineResult(BaseModel):
    """Everything a single monthly run produces. Passed to the output layer."""

    opportunities: list[LinkOpportunity] = Field(default_factory=list)
    orphans: OrphanReport = Field(default_factory=OrphanReport)
    refresh_briefs: list[RefreshBrief] = Field(default_factory=list)
    total_pages: int = 0
