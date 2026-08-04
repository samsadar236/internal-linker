"""Excel report builder.

Produces a single dated workbook with three sheets:
    - Link Placements    : recommended internal links, prioritized
    - Orphan Pages       : critical + at-risk, with severity
    - Refresh Candidates : decay-ranked pages, evergreen flagged and sorted last

The workbook is written to reports/YYYY-MM-DD.xlsx and committed to the repo as
the permanent monthly record (the GitHub Actions artifact expires; this does not).

Exposes:
    build_report(opportunities, orphans, decay_scores, settings, run_date) -> path
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from models import DecayScore, LinkOpportunity, OrphanReport


def _placements_df(opportunities: list[LinkOpportunity]) -> pd.DataFrame:
    rows = []
    for rank, o in enumerate(opportunities, start=1):
        e = o.evidence
        rows.append({
            "rank": rank,
            "source_url": o.source_url,
            "target_url": o.target_url,
            "suggested_anchor_text": o.suggested_anchor_text,
            "rrf_score": e.rrf_score,
            "similarity_score": e.similarity_score,
            "bm25_score": e.bm25_score,
            "anchor_match_score": e.anchor_match_score,
            "target_inbound_links": e.target_inbound_links,
        })
    cols = ["rank", "source_url", "target_url", "suggested_anchor_text", "rrf_score",
            "similarity_score", "bm25_score", "anchor_match_score", "target_inbound_links"]
    return pd.DataFrame(rows, columns=cols)


def _orphans_df(orphans: OrphanReport) -> pd.DataFrame:
    rows = []
    for o in orphans.critical + orphans.at_risk:
        rows.append({
            "url": o.url,
            "title": o.title,
            "inbound_link_count": o.inbound_link_count,
            "severity": o.severity.value,
        })
    cols = ["url", "title", "inbound_link_count", "severity"]
    df = pd.DataFrame(rows, columns=cols)
    # critical first, then by inbound count
    if not df.empty:
        df["_sev"] = df["severity"].map({"critical": 0, "at_risk": 1})
        df = df.sort_values(["_sev", "inbound_link_count", "url"]).drop(columns="_sev")
    return df


def _refresh_df(decay_scores: list[DecayScore]) -> pd.DataFrame:
    rows = []
    for d in decay_scores:
        rows.append({
            "url": d.url,
            "title": d.title,
            "decay_score": d.score,
            "traffic_score": d.traffic_score,
            "content_staleness_score": d.content_staleness_score,
            "is_evergreen": d.is_evergreen,
            "reason": d.reason,
        })
    cols = ["url", "title", "decay_score", "traffic_score",
            "content_staleness_score", "is_evergreen", "reason"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        # evergreen sorted last, then by decay score desc
        df = df.sort_values(["is_evergreen", "decay_score"], ascending=[True, False])
    return df


def build_report(
    opportunities: list[LinkOpportunity],
    orphans: OrphanReport,
    decay_scores: list[DecayScore],
    settings,
    run_date: date | None = None,
) -> str:
    run_date = run_date or date.today()
    os.makedirs(settings.report_dir, exist_ok=True)
    path = os.path.join(settings.report_dir, f"{run_date.isoformat()}.xlsx")

    placements = _placements_df(opportunities)
    orphan_df = _orphans_df(orphans)
    refresh = _refresh_df(decay_scores)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        placements.to_excel(writer, sheet_name="Link Placements", index=False)
        orphan_df.to_excel(writer, sheet_name="Orphan Pages", index=False)
        refresh.to_excel(writer, sheet_name="Refresh Candidates", index=False)
        _autosize(writer)

    return path


def _autosize(writer) -> None:
    """Widen columns to fit content, capped, for a readable workbook."""
    for sheet in writer.sheets.values():
        for column_cells in sheet.columns:
            length = 10
            letter = column_cells[0].column_letter
            for cell in column_cells:
                val = "" if cell.value is None else str(cell.value)
                length = max(length, len(val))
            sheet.column_dimensions[letter].width = min(length + 2, 70)
