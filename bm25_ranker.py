"""BM25 first-pass shortlisting.

For each source page, rank all other pages by topical relevance (BM25 over body
text) and keep the top N as link-target candidates. Pairs the source already
links to (in body) are skipped, as are self-pairs. This shortlist is what the
embedding rerank and RRF steps operate on, so we never embed the full N-squared
cross product.

Exposes:
    tokenize(text) -> list[str]
    bm25_shortlist(pages, settings) -> dict[source_url, list[Candidate]]

Candidate is a tuple: (target_url, bm25_score, rank)  where rank is 1-based.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from models import Page

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# small, cheap stopword list - enough to stop BM25 fixating on filler
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "at", "by", "as", "it",
    "this", "that", "these", "those", "from", "we", "you", "your", "our", "i",
    "he", "she", "they", "them", "his", "her", "its", "their", "if", "then",
    "so", "not", "no", "can", "will", "would", "should", "could", "do", "does",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def bm25_shortlist(pages: list[Page], settings) -> dict[str, list[tuple[str, float, int]]]:
    """Return {source_url: [(target_url, bm25_score, rank), ...]} per source."""
    if len(pages) < 2:
        return {p.url: [] for p in pages}

    urls = [p.url for p in pages]
    # index each page on title + h1 + body so short pages still carry signal
    corpus_tokens = [
        tokenize(f"{p.title} {p.h1} {p.body_text}") for p in pages
    ]
    bm25 = BM25Okapi(corpus_tokens)

    already_linked = {p.url: set(p.internal_links) for p in pages}
    idx_by_url = {u: i for i, u in enumerate(urls)}

    out: dict[str, list[tuple[str, float, int]]] = {}
    for i, src in enumerate(pages):
        query = corpus_tokens[i]
        if not query:
            out[src.url] = []
            continue
        scores = bm25.get_scores(query)

        ranked = sorted(
            (
                (urls[j], float(scores[j]))
                for j in range(len(urls))
                if j != i
                and urls[j] not in already_linked[src.url]
                and scores[j] > 0.0
            ),
            key=lambda x: x[1],
            reverse=True,
        )[: settings.bm25_shortlist_size]

        out[src.url] = [(url, score, rank) for rank, (url, score) in enumerate(ranked, start=1)]

    return out
