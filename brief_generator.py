"""Refresh briefs and anchor text drafting.

Groq drafts two things:
    1. suggested refresh actions for each decaying page
    2. optional alternate anchor phrasings for a link placement
Both paths degrade safely. If GROQ_API_KEY is missing or a call fails, briefs
fall back to a deterministic template and the pipeline does NOT error out.

Anchor variants are re-grounded: any phrase Groq proposes is kept only if it
actually appears in the source page body. If none survive, the original
verbatim-grounded anchor stands. A suggestion can never cite text that is not
on the page.

Briefs are rendered to Content Factory HTML via the Jinja2 template.

Exposes:
    generate_refresh_briefs(decay_scores, pages, settings, run_date) -> list[RefreshBrief]
    render_briefs(briefs, settings, run_date) -> list[str]   # written html paths
    draft_anchor_variants(opportunity, source_page, settings) -> list[str]
"""

from __future__ import annotations

import os
from datetime import date

from jinja2 import Environment, FileSystemLoader, select_autoescape

from models import DecayScore, LinkOpportunity, Page, RefreshBrief

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


# --- Groq access (isolated, always safe) -----------------------------------

def _groq_chat(prompt: str, settings, max_tokens: int = 400) -> str | None:
    """Return Groq's text reply, or None on any failure / missing key."""
    if not settings.groq_api_key:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception:
        # network, auth, rate limit, SDK missing - all handled as "no LLM"
        return None


# --- suggested actions ------------------------------------------------------

def _fallback_actions(decay: DecayScore) -> list[str]:
    """Deterministic actions used when Groq is unavailable."""
    actions = [
        "Review the introduction and update any figures, dates, or claims that have aged.",
        "Check that the primary keyword still matches current search intent.",
        "Add or refresh internal links to and from related pages.",
    ]
    if decay.content_staleness_score >= 0.98:
        actions.append("Content is unchanged since last month; add a substantive update or new section.")
    if "bounce" in decay.reason:
        actions.append("High bounce rate: improve above-the-fold clarity and page load.")
    if "low volume" in decay.reason:
        actions.append("Low traffic: reassess whether to expand, consolidate, or retire this page.")
    return actions


def _draft_actions(decay: DecayScore, page: Page, settings) -> list[str]:
    excerpt = " ".join(page.body_text.split()[:120])
    prompt = (
        "You are an SEO content strategist. A page is losing search performance. "
        "Give 4 to 6 short, concrete refresh actions as a plain list, one per line, "
        "no numbering, no preamble.\n\n"
        f"Page title: {page.title}\n"
        f"Why flagged: {decay.reason}\n"
        f"Content excerpt: {excerpt}\n"
    )
    reply = _groq_chat(prompt, settings)
    if not reply:
        return _fallback_actions(decay)
    lines = [ln.strip(" -*\t") for ln in reply.splitlines() if ln.strip()]
    lines = [ln for ln in lines if len(ln) > 3]
    return lines[:6] if lines else _fallback_actions(decay)


def generate_refresh_briefs(
    decay_scores: list[DecayScore],
    pages: list[Page],
    settings,
    run_date: date | None = None,
) -> list[RefreshBrief]:
    by_url = {p.url: p for p in pages}
    briefs: list[RefreshBrief] = []
    # only decaying (non-evergreen) pages with a real signal become briefs
    for d in decay_scores:
        if d.is_evergreen or d.score <= 0.0:
            continue
        page = by_url.get(d.url)
        if page is None:
            continue
        actions = _draft_actions(d, page, settings)
        briefs.append(RefreshBrief(
            url=d.url,
            title=d.title,
            decay_score=d.score,
            reason=d.reason,
            suggested_actions=actions,
            traffic_trend=0.0,  # populated by caller if traffic stats are threaded through
        ))
    return briefs


# --- anchor variants (grounded) --------------------------------------------

def draft_anchor_variants(opportunity: LinkOpportunity, source_page: Page, settings) -> list[str]:
    """Groq proposes anchor phrasings; only those present in source body survive."""
    grounded = opportunity.suggested_anchor_text
    prompt = (
        "Suggest 3 short natural anchor-text phrases (2-5 words) for a link, "
        "each of which must be a phrase that already appears in the source text. "
        "One per line, no numbering.\n\n"
        f"Source text: {' '.join(source_page.body_text.split()[:200])}\n"
        f"Current anchor: {grounded}\n"
    )
    reply = _groq_chat(prompt, settings)
    body_norm = " ".join(source_page.body_text.lower().split())
    variants = [grounded]
    if reply:
        for ln in reply.splitlines():
            cand = ln.strip(" -*\t").strip()
            if cand and cand.lower() in body_norm and cand not in variants:
                variants.append(cand)
    return variants[:3]


# --- rendering --------------------------------------------------------------

def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )


def render_briefs(briefs: list[RefreshBrief], settings, run_date: date | None = None) -> list[str]:
    run_date = run_date or date.today()
    env = _env()
    template = env.get_template("brief_template.j2")
    out_dir = os.path.join(settings.report_dir, "briefs", run_date.isoformat())
    os.makedirs(out_dir, exist_ok=True)

    paths: list[str] = []
    for i, brief in enumerate(briefs, start=1):
        html = template.render(brief=brief)
        fname = f"brief_{i:02d}.html"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        paths.append(path)
    return paths
