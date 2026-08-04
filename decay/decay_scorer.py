"""Content decay scoring.

Combines a traffic signal (declining trend, low visits, high bounce) with a
staleness signal (how unchanged the page is versus last month's crawl) into a
single decay score. Evergreen pages are protected first, before any scoring.

The evergreen rule has three conditions, ALL required:
    - word_count >= evergreen_min_words
    - |trend_pct| <= evergreen_trend_band  (traffic roughly flat)
    - nb_visits >= evergreen_min_visits    (the traffic floor)
The traffic floor stops a long-but-ignored page from being silently protected.

Data source switches on settings.use_mock_data:
    True  -> mock_data.matomo_mock
    False -> real self-hosted Matomo Reporting API

Exposes:
    fetch_traffic(pages, settings) -> dict[url, TrafficStats]
    score_decay(pages, traffic, previous_pages, settings) -> list[DecayScore]
"""

from __future__ import annotations

from rapidfuzz import fuzz

from models import DecayScore, Page, TrafficStats


# --- traffic fetch ---------------------------------------------------------

def fetch_traffic(pages: list[Page], settings) -> dict[str, TrafficStats]:
    if settings.use_mock_data:
        from mock_data.matomo_mock import generate_traffic

        return generate_traffic(pages, settings)
    return _fetch_matomo(pages, settings)


def _fetch_matomo(pages: list[Page], settings) -> dict[str, TrafficStats]:
    """Pull real traffic from a self-hosted Matomo Reporting API.

    Queries Actions.getPageUrls (flat) for the current and previous period and
    matches rows to crawled pages by URL path, computing a signed trend.
    Kept defensive: any transport error raises with a clear message so the
    monthly workflow surfaces it in Slack rather than silently emitting zeros.
    """
    import httpx
    from urllib.parse import urlparse

    if not settings.matomo_base_url or not settings.matomo_token:
        raise RuntimeError(
            "USE_MOCK_DATA is false but MATOMO_BASE_URL / MATOMO_TOKEN are unset."
        )

    def _call(date_range: str) -> list[dict]:
        params = {
            "module": "API",
            "method": "Actions.getPageUrls",
            "idSite": settings.matomo_site_id,
            "period": "range",
            "date": date_range,
            "format": "JSON",
            "flat": "1",
            "token_auth": settings.matomo_token,
            "filter_limit": "-1",
        }
        resp = httpx.get(
            settings.matomo_base_url.rstrip("/") + "/index.php",
            params=params,
            timeout=settings.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("result") == "error":
            raise RuntimeError(f"Matomo API error: {data.get('message')}")
        return data if isinstance(data, list) else []

    days = settings.matomo_period_days
    cur = _call(f"last{days}")
    prev = _call(f"previous{days}")

    def _index(rows: list[dict]) -> dict[str, dict]:
        out = {}
        for r in rows:
            label = (r.get("label") or r.get("url") or "").split("?")[0]
            path = urlparse(label).path if label.startswith("http") else label
            out[path.rstrip("/") or "/"] = r
        return out

    cur_idx, prev_idx = _index(cur), _index(prev)

    stats: dict[str, TrafficStats] = {}
    for p in pages:
        path = (urlparse(p.url).path or "/").rstrip("/") or "/"
        c = cur_idx.get(path, {})
        pr = prev_idx.get(path, {})
        cur_visits = int(c.get("nb_visits", 0) or 0)
        prev_visits = int(pr.get("nb_visits", 0) or 0)
        trend = ((cur_visits - prev_visits) / prev_visits * 100.0) if prev_visits else 0.0
        stats[p.url] = TrafficStats(
            url=p.url,
            nb_visits=cur_visits,
            nb_hits=int(c.get("nb_hits", 0) or 0),
            bounce_rate=float(c.get("bounce_rate", 0) or 0) / 100.0
            if float(c.get("bounce_rate", 0) or 0) > 1 else float(c.get("bounce_rate", 0) or 0),
            avg_time_on_page=float(c.get("avg_time_on_page", 0) or 0),
            trend_pct=round(trend, 1),
        )
    return stats


# --- scoring ---------------------------------------------------------------

def _is_evergreen(page: Page, t: TrafficStats, settings) -> bool:
    return (
        page.word_count >= settings.evergreen_min_words
        and abs(t.trend_pct) <= settings.evergreen_trend_band
        and t.nb_visits >= settings.evergreen_min_visits
    )


def _traffic_score(t: TrafficStats, settings) -> float:
    decline = min(1.0, max(0.0, -t.trend_pct) / 100.0)       # steeper drop -> higher
    bounce = min(1.0, max(0.0, t.bounce_rate))
    low = 1.0 if t.nb_visits < settings.evergreen_min_visits else 0.0
    score = 0.6 * decline + 0.3 * bounce + 0.1 * low
    return min(1.0, score)


def _staleness_score(page: Page, previous: dict[str, Page]) -> float:
    """1.0 = identical to last month (maximally stale). 0.0 = no prior / rewritten."""
    prev = previous.get(page.url)
    if prev is None:
        return 0.0  # first sighting: cannot judge staleness from history
    if not prev.body_text and not page.body_text:
        return 0.0
    return fuzz.ratio(prev.body_text, page.body_text) / 100.0


def score_decay(
    pages: list[Page],
    traffic: dict[str, TrafficStats],
    previous_pages: list[Page] | None,
    settings,
) -> list[DecayScore]:
    previous = {p.url: p for p in (previous_pages or [])}
    titles = {p.url: p.title for p in pages}

    results: list[DecayScore] = []
    for p in pages:
        t = traffic.get(p.url)
        if t is None:
            continue  # no traffic data -> cannot assess decay

        if _is_evergreen(p, t, settings):
            results.append(DecayScore(
                url=p.url, title=titles.get(p.url, ""),
                score=0.0, traffic_score=0.0, content_staleness_score=0.0,
                is_evergreen=True,
                reason=f"Evergreen: {p.word_count} words, trend {t.trend_pct:+.0f}%, "
                       f"{t.nb_visits} visits (>= floor {settings.evergreen_min_visits}).",
            ))
            continue

        ts = _traffic_score(t, settings)
        ss = _staleness_score(p, previous)
        final = settings.decay_traffic_weight * ts + settings.decay_staleness_weight * ss

        drivers = []
        if t.trend_pct < 0:
            drivers.append(f"traffic {t.trend_pct:+.0f}%")
        if t.bounce_rate >= 0.6:
            drivers.append(f"bounce {t.bounce_rate*100:.0f}%")
        if t.nb_visits < settings.evergreen_min_visits:
            drivers.append(f"low volume ({t.nb_visits} visits)")
        if ss >= 0.98:
            drivers.append("content unchanged since last run")
        reason = "; ".join(drivers) if drivers else "mild decay signal"

        results.append(DecayScore(
            url=p.url, title=titles.get(p.url, ""),
            score=round(final, 4),
            traffic_score=round(ts, 4),
            content_staleness_score=round(ss, 4),
            is_evergreen=False,
            reason=reason,
        ))

    results.sort(key=lambda d: d.score, reverse=True)
    return results
