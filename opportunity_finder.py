"""The opportunity engine: turn shortlists into confirmed link placements.

Pipeline per source page:
    1. take the BM25 shortlist of candidate targets
    2. rerank those candidates by embedding cosine similarity
    3. fuse the two rankings with Reciprocal Rank Fusion (RRF)
    4. boost candidates whose target is starved (orphan / at-risk)
Then, site-wide:
    5. sort every candidate by fused score
    6. walk down the list grounding an anchor in the source body (rapidfuzz);
       keep a placement only if a real phrase in the source matches the target
    7. stop once the site-wide ceiling is reached

Anchor text is a verbatim whole-word phrase lifted from the source body, so a
suggestion can never point at wording that is not actually on the page. Groq
polishes anchors later, but this grounded phrase is the anchor of record.

Exposes:
    find_opportunities(pages, shortlist, vectors, inbound_counts, settings)
        -> list[LinkOpportunity]
    ground_anchor(query, body, threshold) -> (phrase | None, score)
"""

from __future__ import annotations

from rapidfuzz import fuzz, process

from models import EvidenceBlock, LinkOpportunity, Page


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def ground_anchor(query: str, body: str, threshold: float) -> tuple[str | None, float]:
    """Find the whole-word phrase in `body` that best matches `query`.

    Returns (phrase, score) if score >= threshold, else (None, best_score).
    Scans word windows of size len(query_words) +/- 1 so the returned anchor is
    always a clean run of real words, never a ragged character slice.
    """
    q = " ".join(query.split()).strip()
    if not q or not body.strip():
        return None, 0.0

    words = body.split()
    if not words:
        return None, 0.0

    qlen = max(1, len(q.split()))
    sizes = {max(1, qlen - 1), qlen, qlen + 1}

    choices: list[str] = []
    for w in sizes:
        if w > len(words):
            continue
        for i in range(0, len(words) - w + 1):
            choices.append(" ".join(words[i : i + w]))
    if not choices:
        return None, 0.0

    match = process.extractOne(q, choices, scorer=fuzz.ratio)
    if match is None:
        return None, 0.0
    phrase, score, _ = match
    if score >= threshold:
        return phrase, float(score)
    return None, float(score)


def find_opportunities(
    pages: list[Page],
    shortlist: dict[str, list[tuple[str, float, int]]],
    vectors: dict[str, list[float]],
    inbound_counts: dict[str, int],
    settings,
) -> list[LinkOpportunity]:
    by_url = {p.url: p for p in pages}

    # ---- steps 1-4: build scored candidates ------------------------------
    candidates: list[dict] = []
    for src_url, cands in shortlist.items():
        if not cands or src_url not in vectors:
            continue
        src_vec = vectors[src_url]

        # cosine of source vs each shortlisted target that has a vector
        cos = []
        for target_url, bm25_score, bm25_rank in cands:
            if target_url not in vectors:
                continue
            sim = _dot(src_vec, vectors[target_url])
            cos.append((target_url, sim, bm25_score, bm25_rank))

        # cosine ranking (1-based)
        cos_sorted = sorted(cos, key=lambda x: x[1], reverse=True)
        cos_rank = {t[0]: r for r, t in enumerate(cos_sorted, start=1)}

        k = settings.rrf_k
        for target_url, sim, bm25_score, bm25_rank in cos:
            rrf = 1.0 / (k + bm25_rank) + 1.0 / (k + cos_rank[target_url])
            target_inbound = inbound_counts.get(target_url, 0)
            # boost starved targets (orphan or at-risk)
            if target_inbound <= settings.at_risk_max_inbound:
                rrf *= settings.orphan_boost
            candidates.append({
                "source_url": src_url,
                "target_url": target_url,
                "rrf": rrf,
                "similarity": sim,
                "bm25": bm25_score,
                "target_inbound": target_inbound,
            })

    # ---- steps 5-7: global sort, ground anchors, fill to ceiling ---------
    candidates.sort(key=lambda c: c["rrf"], reverse=True)

    placements: list[LinkOpportunity] = []
    for c in candidates:
        if len(placements) >= settings.placement_ceiling:
            break
        src = by_url.get(c["source_url"])
        tgt = by_url.get(c["target_url"])
        if src is None or tgt is None:
            continue

        query = (tgt.title or tgt.h1 or "").strip()
        anchor, anchor_score = ground_anchor(query, src.body_text, settings.anchor_match_threshold)
        if anchor is None:
            continue  # no natural place to put this link in the source body

        placements.append(LinkOpportunity(
            source_url=c["source_url"],
            target_url=c["target_url"],
            suggested_anchor_text=anchor,
            evidence=EvidenceBlock(
                rrf_score=round(c["rrf"], 6),
                similarity_score=round(c["similarity"], 6),
                bm25_score=round(c["bm25"], 6),
                anchor_match_score=round(anchor_score, 2),
                target_inbound_links=c["target_inbound"],
            ),
        ))

    return placements
