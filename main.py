"""Orchestrator. The single entry point for a monthly run.

Order:
    0. mock-data guardrail (warn if USE_MOCK_DATA is true past launch)
    1. crawl (or resume from today's snapshot if one exists)
    2. link graph -> orphans + PageRank + inbound counts
    3. BM25 shortlist
    4. embeddings (cached)
    5. opportunity finding (RRF + grounded anchors, site-wide ceiling)
    6. traffic fetch (mock or Matomo) + decay scoring (uses last month's snapshot)
    7. refresh briefs (Groq w/ fallback) -> rendered HTML
    8. Excel report (committed to /reports as the permanent record)
    9. sanity checks -> Slack summary

Each module exposes one clean function; the core pipeline stays decoupled,
sharing only models.py and config.py.
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import date

from config import settings
from models import PipelineResult


# --- helpers ---------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[internal-linker] {msg}", flush=True)


def _mock_data_guardrail() -> None:
    if settings.use_mock_data and settings.launch_date and date.today() > settings.launch_date:
        _log(
            f"WARNING: USE_MOCK_DATA is true but launch date {settings.launch_date} "
            f"has passed. This run is on MOCK traffic, not real Matomo data."
        )


def _find_previous_snapshot(run_date: date) -> str | None:
    """Most recent snapshot strictly before run_date, for staleness comparison."""
    from crawler.scraper import snapshot_path_for  # noqa
    pattern = os.path.join(settings.snapshot_dir, "*.json")
    best_path, best_date = None, None
    for path in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            d = date.fromisoformat(stem)
        except ValueError:
            continue
        if d < run_date and (best_date is None or d > best_date):
            best_date, best_path = d, path
    return best_path


def notify(text: str) -> None:
    """Post a short summary to Slack if a webhook is configured. Never raises."""
    if not settings.slack_webhook_url:
        return
    try:
        import httpx

        httpx.post(settings.slack_webhook_url, json={"text": text}, timeout=10.0)
    except Exception:
        pass


# --- run -------------------------------------------------------------------

def run(settings) -> PipelineResult:
    run_date = date.today()
    _mock_data_guardrail()

    # 1. crawl or resume
    from crawler.scraper import crawl_site, load_snapshot, snapshot_path_for

    today_snapshot = snapshot_path_for(settings, run_date)
    if os.path.exists(today_snapshot):
        _log(f"Resuming from existing snapshot: {today_snapshot}")
        pages = load_snapshot(today_snapshot)
    else:
        _log(f"Crawling {settings.start_url} ...")
        pages = crawl_site(settings)
    _log(f"{len(pages)} pages.")

    # 2. graph
    from graph.orphan_detector import build_graph, compute_pagerank, detect_orphans, inbound_counts

    graph = build_graph(pages)
    orphans = detect_orphans(graph, pages, settings)
    compute_pagerank(graph)  # available for future weighting; not required downstream
    counts = inbound_counts(graph)
    _log(f"orphans: {len(orphans.critical)} critical, {len(orphans.at_risk)} at-risk.")

    # 3. BM25 shortlist
    from engine.bm25_ranker import bm25_shortlist

    shortlist = bm25_shortlist(pages, settings)

    # 4. embeddings
    from engine.embeddings import EmbeddingIndex

    index = EmbeddingIndex(settings)
    vectors = index.embed_pages(pages)

    # 5. opportunities
    from engine.opportunity_finder import find_opportunities

    opportunities = find_opportunities(pages, shortlist, vectors, counts, settings)
    _log(f"{len(opportunities)} link placements.")

    # 6. traffic + decay
    from decay.decay_scorer import fetch_traffic, score_decay

    traffic = fetch_traffic(pages, settings)
    prev_path = _find_previous_snapshot(run_date)
    previous_pages = load_snapshot(prev_path) if prev_path else None
    if prev_path:
        _log(f"staleness baseline: {prev_path}")
    decay_scores = score_decay(pages, traffic, previous_pages, settings)
    flagged = [d for d in decay_scores if not d.is_evergreen and d.score > 0.0]
    _log(f"{len(flagged)} decay candidates ({len(decay_scores) - len(flagged)} evergreen/clean).")

    # 7. briefs
    from output.brief_generator import generate_refresh_briefs, render_briefs

    briefs = generate_refresh_briefs(decay_scores, pages, settings, run_date)
    # thread the real traffic trend onto each brief for the template
    trend_by_url = {u: t.trend_pct for u, t in traffic.items()}
    for b in briefs:
        b.traffic_trend = trend_by_url.get(b.url, 0.0)
    brief_paths = render_briefs(briefs, settings, run_date)
    _log(f"{len(brief_paths)} refresh briefs rendered.")

    # 8. report
    from output.report_builder import build_report

    report_path = build_report(opportunities, orphans, decay_scores, settings, run_date)
    _log(f"report: {report_path}")

    # 9. sanity checks
    warnings = _sanity_checks(opportunities, orphans, len(pages))
    for w in warnings:
        _log(f"SANITY: {w}")

    result = PipelineResult(
        opportunities=opportunities,
        orphans=orphans,
        refresh_briefs=briefs,
        total_pages=len(pages),
    )

    summary = (
        f"internal-linker {run_date.isoformat()}: {len(pages)} pages, "
        f"{len(opportunities)} placements, {len(orphans.critical)} critical orphans, "
        f"{len(flagged)} refresh candidates."
    )
    if warnings:
        summary += " ⚠ " + " | ".join(warnings)
    notify(summary)
    _log("done.")
    return result


def _sanity_checks(opportunities, orphans, total_pages: int) -> list[str]:
    warnings: list[str] = []
    if len(opportunities) < settings.placement_floor:
        warnings.append(
            f"placements {len(opportunities)} below floor {settings.placement_floor}"
        )
    if total_pages > 0 and len(orphans.critical) > 0.5 * total_pages:
        warnings.append(
            f"critical orphans {len(orphans.critical)} exceed 50% of {total_pages} pages "
            f"(possible crawl/link-extraction fault)"
        )
    return warnings


if __name__ == "__main__":
    try:
        run(settings)
    except Exception as exc:  # noqa: BLE001
        notify(f"internal-linker FAILED: {type(exc).__name__}: {exc}")
        _log(f"FAILED: {exc}")
        raise SystemExit(1)
