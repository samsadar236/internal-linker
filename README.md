# Internal Linking & Content Refresh Recommender

Monthly SEO audit for the SS site. Crawls the site, then produces:

1. **Link placements** — internal links that should exist (RRF over BM25 + embeddings, anchor text grounded verbatim in the source page).
2. **Orphan pages** — critical (0 inbound body links) and at-risk (1–2).
3. **Refresh candidates** — pages decaying on traffic + content staleness, with a refresh brief for each.

Output is one Excel workbook per run plus one HTML brief per refresh candidate. Everything feeds the Content Factory pipeline; this tool never edits pages directly (D-S4-9).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env
```

## Run

```bash
python main.py
```

Outputs:
- `reports/YYYY-MM-DD.xlsx` — three sheets: Link Placements, Orphan Pages, Refresh Candidates
- `reports/briefs/YYYY-MM-DD/brief_NN.html` — one refresh brief per candidate
- `snapshots/YYYY-MM-DD.json` — crawl snapshot (also the staleness baseline for next month)

Re-running on the same day resumes from that day's snapshot instead of re-crawling.

## Configuration

All knobs live in `.env` (see `.env.example` for the full list). The ones that matter most:

- `START_URL` — the site to crawl.
- `USE_MOCK_DATA` — `true` before launch (synthetic traffic), `false` after (real Matomo).
- `LAUNCH_DATE` — once set, a mock-data run after this date prints a loud warning.
- `PLACEMENT_FLOOR` / `PLACEMENT_CEILING` — site-wide min (warn if under) / max links per run.
- `ANCHOR_MATCH_THRESHOLD` — how strict anchor grounding is (0–100). Lower = more placements, looser anchors.
- `EVERGREEN_MIN_WORDS` / `EVERGREEN_TREND_BAND` / `EVERGREEN_MIN_VISITS` — evergreen protection; all three must hold, including the visit floor.
- `GROQ_API_KEY` — optional. Without it, briefs use deterministic template actions.

## Launch swap (mock → real Matomo)

1. Set `MATOMO_BASE_URL`, `MATOMO_TOKEN`, `MATOMO_SITE_ID`.
2. Set `USE_MOCK_DATA=false`.
3. Set `START_URL` to the live SS site.
4. Run once manually and confirm the Refresh Candidates sheet reflects real traffic.

No code changes — the mock emits the same shape the Matomo fetch returns.

## Automation (GitHub Actions)

`.github/workflows/monthly_run.yml` runs on the 1st of each month (and on demand via **workflow_dispatch**). It installs, runs, commits the report + that month's snapshot back to the repo (the permanent record + next month's staleness baseline), uploads the report as a 90-day artifact, and pings Slack on failure.

Set these in the repo:
- **Secrets:** `MATOMO_BASE_URL`, `MATOMO_TOKEN`, `GROQ_API_KEY`, `SLACK_WEBHOOK_URL`
- **Variables:** `START_URL`, `USE_MOCK_DATA`, `LAUNCH_DATE`, `MATOMO_SITE_ID`

The embedding cache (`chroma_store/`) is ephemeral in CI by design; embeddings recompute each monthly run, which is cheap for BGE-small.

## Layout

```
config.py                  all settings (pydantic-settings)
models.py                  shared Pydantic types (incl. EvidenceBlock = Stage-3 handoff)
crawler/scraper.py         async crawl, chrome-stripping, URL normalization, snapshots
graph/orphan_detector.py   link graph, two-tier orphans, PageRank
engine/bm25_ranker.py      BM25 shortlist
engine/embeddings.py       Sentence-Transformers + ChromaDB cache
engine/opportunity_finder.py  cosine rerank + RRF + grounded anchors
decay/decay_scorer.py      traffic + staleness decay, evergreen protection
mock_data/matomo_mock.py   pre-launch traffic (matches Matomo shape)
output/report_builder.py   Excel report
output/brief_generator.py  Groq briefs (template fallback) + Jinja2 render
main.py                    orchestrator
```
