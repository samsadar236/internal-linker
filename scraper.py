"""Async site crawler.

Follows internal links from a start URL, strips chrome (nav/header/footer/
aside/script/style) before extracting anything, and returns a list of `Page`
objects. On completion it writes a dated JSON snapshot to disk so a failed run
can be resumed without re-crawling.

Exposes:
    crawl_site(settings) -> list[Page]        # crawl and snapshot
    save_snapshot(pages, path)                # write snapshot json
    load_snapshot(path) -> list[Page]         # read snapshot json
    snapshot_path_for(settings, run_date)     # canonical snapshot path
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import date
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from models import Page

# Extensions we never treat as crawlable HTML pages.
_NON_HTML_EXT = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".css", ".js", ".json", ".xml", ".rss", ".zip", ".gz", ".tar",
    ".mp4", ".mp3", ".wav", ".avi", ".mov", ".woff", ".woff2", ".ttf",
    ".eot", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".csv",
}

_STRIP_TAGS = ("head", "nav", "header", "footer", "aside", "script", "style", "noscript")


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Normalize a URL for dedup and comparison.

    - resolve relative against base
    - drop fragment
    - lowercase scheme + host
    - remove trailing slash (except root)
    - reject non-http(s) schemes and non-HTML file extensions
    Returns None if the URL should be skipped.
    """
    if base:
        url = urljoin(base, url)
    url, _ = urldefrag(url)
    parts = urlparse(url)

    if parts.scheme not in ("http", "https"):
        return None

    path = parts.path or "/"
    # reject obvious file downloads
    lower = path.lower()
    dot = lower.rfind(".")
    if dot != -1 and lower[dot:] in _NON_HTML_EXT:
        return None

    # collapse directory index files so /foo/index.html == /foo == /foo/
    segments = path.split("/")
    if segments and re.match(r"^(index|default)\.(html?|php|aspx?)$", segments[-1], re.I):
        segments[-1] = ""
        path = "/".join(segments) or "/"

    # strip trailing slash except for root
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    rebuilt = urlunparse((
        parts.scheme.lower(),
        parts.netloc.lower(),
        path,
        "",                # params
        parts.query,       # keep query (can be meaningful)
        "",                # fragment already stripped
    ))
    return rebuilt


def _same_site(url: str, root_netloc: str) -> bool:
    return urlparse(url).netloc.lower() == root_netloc


def _parse_page(url: str, html: str) -> tuple[Page, list[str], list[str]]:
    """Extract a Page plus two href lists from HTML.

    Returns (page, all_hrefs, body_hrefs):
    - all_hrefs   : every internal link on the page, INCLUDING nav/footer.
                    Used only to DISCOVER pages during the crawl, so that a
                    page reachable solely via navigation still becomes a node
                    (and can therefore be flagged as an orphan).
    - body_hrefs  : links found AFTER chrome is stripped. These are the links
                    that actually count - they populate page.internal_links and
                    drive the graph, orphan counts, and "already linked" checks.

    body_text is likewise extracted from the stripped document so similarity is
    not polluted by menu and footer boilerplate.
    """
    soup = BeautifulSoup(html, "lxml")

    all_hrefs = [a.get("href") for a in soup.find_all("a", href=True)]

    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(strip=True) if h1_tag else ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_tag.get("content", "").strip() if meta_tag else ""

    # strip chrome, then collect body links and body text from what remains
    for tag_name in _STRIP_TAGS:
        for t in soup.find_all(tag_name):
            t.decompose()
    body_hrefs = [a.get("href") for a in soup.find_all("a", href=True)]
    body_text = " ".join(soup.get_text(separator=" ").split())
    word_count = len(body_text.split())

    page = Page(
        url=url,
        title=title,
        h1=h1,
        meta_description=meta_description,
        body_text=body_text,
        internal_links=[],          # filled by caller from body_hrefs
        word_count=word_count,
    )
    return page, all_hrefs, body_hrefs


async def _fetch(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore, delay: float):
    async with sem:
        try:
            resp = await client.get(url)
        except (httpx.HTTPError, httpx.InvalidURL):
            return url, None
        await asyncio.sleep(delay)
        ctype = resp.headers.get("content-type", "")
        if resp.status_code != 200 or "html" not in ctype.lower():
            return url, None
        return url, resp.text


async def _crawl(settings) -> list[Page]:
    start = normalize_url(settings.start_url)
    if start is None:
        raise ValueError(f"start_url is not crawlable: {settings.start_url}")
    root_netloc = urlparse(start).netloc.lower()

    seen: set[str] = {start}
    pages: dict[str, Page] = {}
    frontier: list[tuple[str, int]] = [(start, 0)]

    sem = asyncio.Semaphore(settings.crawl_concurrency)
    headers = {"User-Agent": settings.user_agent}

    async with httpx.AsyncClient(
        headers=headers,
        timeout=settings.request_timeout,
        follow_redirects=True,
    ) as client:
        while frontier and len(pages) < settings.max_pages:
            batch = frontier
            frontier = []
            tasks = [_fetch(client, u, sem, settings.crawl_delay_seconds) for u, _ in batch]
            depths = {u: d for u, d in batch}
            results = await asyncio.gather(*tasks)

            for url, html in results:
                if html is None:
                    continue
                page, all_hrefs, body_hrefs = _parse_page(url, html)
                depth = depths.get(url, 0)

                # DISCOVERY: expand the frontier using every internal link,
                # including nav/footer, so nav-only pages still become nodes.
                for href in all_hrefs:
                    n = normalize_url(href, base=url)
                    if n is None or not _same_site(n, root_netloc):
                        continue
                    if n not in seen and depth + 1 <= settings.max_depth:
                        seen.add(n)
                        frontier.append((n, depth + 1))

                # GRAPH: internal_links holds body-only links (nav/footer excluded).
                body_links: list[str] = []
                for href in body_hrefs:
                    n = normalize_url(href, base=url)
                    if n is None or not _same_site(n, root_netloc):
                        continue
                    if n == url:
                        continue  # ignore self-links
                    body_links.append(n)

                page.internal_links = sorted(set(body_links))
                pages[url] = page

    return list(pages.values())


def crawl_site(settings) -> list[Page]:
    """Crawl the site synchronously (wraps the async crawler) and snapshot it."""
    pages = asyncio.run(_crawl(settings))
    path = snapshot_path_for(settings, date.today())
    save_snapshot(pages, path)
    return pages


# --- snapshot persistence --------------------------------------------------

def snapshot_path_for(settings, run_date: date) -> str:
    os.makedirs(settings.snapshot_dir, exist_ok=True)
    return os.path.join(settings.snapshot_dir, f"{run_date.isoformat()}.json")


def save_snapshot(pages: list[Page], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = [p.model_dump() for p in pages]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def load_snapshot(path: str) -> list[Page]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [Page(**row) for row in data]
