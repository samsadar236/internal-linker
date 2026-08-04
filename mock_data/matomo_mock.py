"""Mock traffic data for pre-launch testing.

Emits TrafficStats in the exact shape the real Matomo fetch returns, so moving
to production is a single flag flip (USE_MOCK_DATA=false), no code changes.

Each page is deterministically assigned one of four archetypes by hashing its
URL, so runs are reproducible:
    healthy    - rising traffic, low bounce
    declining  - falling traffic, medium bounce  (the ones we want to catch)
    low_traffic- few visits, high bounce
    evergreen  - steady traffic, low bounce
"""

from __future__ import annotations

import hashlib

from models import Page, TrafficStats

_ARCHETYPES = ("healthy", "declining", "low_traffic", "evergreen")


def _bucket(url: str) -> str:
    h = int(hashlib.md5(url.encode("utf-8")).hexdigest(), 16)
    return _ARCHETYPES[h % len(_ARCHETYPES)]


def _stats_for(url: str, archetype: str) -> TrafficStats:
    if archetype == "healthy":
        return TrafficStats(url=url, nb_visits=1200, nb_hits=1500,
                            bounce_rate=0.28, avg_time_on_page=95.0, trend_pct=18.0)
    if archetype == "declining":
        return TrafficStats(url=url, nb_visits=420, nb_hits=520,
                            bounce_rate=0.61, avg_time_on_page=44.0, trend_pct=-37.0)
    if archetype == "low_traffic":
        return TrafficStats(url=url, nb_visits=18, nb_hits=22,
                            bounce_rate=0.82, avg_time_on_page=15.0, trend_pct=-6.0)
    # evergreen
    return TrafficStats(url=url, nb_visits=960, nb_hits=1100,
                        bounce_rate=0.33, avg_time_on_page=120.0, trend_pct=2.0)


def generate_traffic(pages: list[Page], settings) -> dict[str, TrafficStats]:
    return {p.url: _stats_for(p.url, _bucket(p.url)) for p in pages}
