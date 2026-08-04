"""Internal link graph, orphan detection, and PageRank.

Builds a directed graph from body links only (nav/footer already excluded by
the crawler), then:
- flags orphans in two tiers (critical = 0 inbound, at-risk = 1..N inbound)
- computes PageRank, reused later to boost starved targets in scoring

Exposes:
    build_graph(pages) -> networkx.DiGraph
    detect_orphans(graph, pages, settings) -> OrphanReport
    compute_pagerank(graph) -> dict[url, float]
    inbound_counts(graph) -> dict[url, int]
"""

from __future__ import annotations

import networkx as nx

from models import OrphanPage, OrphanReport, Page, Severity


def build_graph(pages: list[Page]) -> nx.DiGraph:
    """Directed graph. Nodes = crawled pages. Edges = body links between them.

    Links pointing at URLs we did not crawl are ignored, so inbound counts only
    reflect verified internal pages.
    """
    g = nx.DiGraph()
    known = {p.url for p in pages}
    for p in pages:
        g.add_node(p.url, title=p.title, word_count=p.word_count)
    for p in pages:
        for target in p.internal_links:
            if target in known and target != p.url:
                g.add_edge(p.url, target)
    return g


def inbound_counts(graph: nx.DiGraph) -> dict[str, int]:
    return {node: graph.in_degree(node) for node in graph.nodes}


def detect_orphans(graph: nx.DiGraph, pages: list[Page], settings) -> OrphanReport:
    """Split pages into critical (0 inbound) and at-risk (1..N inbound)."""
    titles = {p.url: p.title for p in pages}
    counts = inbound_counts(graph)

    critical: list[OrphanPage] = []
    at_risk: list[OrphanPage] = []

    for url, count in counts.items():
        if count == 0:
            critical.append(OrphanPage(
                url=url, title=titles.get(url, ""),
                inbound_link_count=0, severity=Severity.CRITICAL,
            ))
        elif count <= settings.at_risk_max_inbound:
            at_risk.append(OrphanPage(
                url=url, title=titles.get(url, ""),
                inbound_link_count=count, severity=Severity.AT_RISK,
            ))

    critical.sort(key=lambda o: o.url)
    at_risk.sort(key=lambda o: (o.inbound_link_count, o.url))

    return OrphanReport(critical=critical, at_risk=at_risk, total_pages=len(pages))


def compute_pagerank(graph: nx.DiGraph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    # networkx handles dangling nodes; alpha is the standard 0.85 damping factor
    return nx.pagerank(graph, alpha=0.85)
