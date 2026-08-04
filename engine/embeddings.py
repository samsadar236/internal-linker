"""Semantic embeddings with a persistent ChromaDB cache.

Each page is embedded with a Sentence-Transformers model (BGE-small by default)
and stored in ChromaDB keyed by URL, with the page's content hash in metadata.
On the next monthly run, a page whose hash is unchanged reuses its stored vector
instead of being re-encoded, so only new or edited pages cost compute.

Vectors are L2-normalized, so cosine similarity is a plain dot product.

Heavy dependencies (sentence_transformers, chromadb) are imported lazily inside
the class, so importing this module never requires them to be installed. That
keeps the rest of the pipeline testable in a light environment.

Exposes:
    EmbeddingIndex(settings)                      # holds model + chroma handle
    EmbeddingIndex.embed_pages(pages) -> dict     # {url: normalized vector}
"""

from __future__ import annotations

from models import Page


def _normalize(vec: list[float]) -> list[float]:
    import math

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class EmbeddingIndex:
    """Wraps the model and the Chroma collection. One instance per run."""

    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._collection = None

    # --- lazy resource loaders --------------------------------------------
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=self.settings.chroma_path)
            self._collection = client.get_or_create_collection(
                name=self.settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @staticmethod
    def _embed_text(page: Page) -> str:
        # embed only meaningful content; title + h1 + body (chrome already gone)
        return f"{page.title}\n{page.h1}\n{page.body_text}".strip()

    # --- main entry point --------------------------------------------------
    def embed_pages(self, pages: list[Page]) -> dict[str, list[float]]:
        """Return {url: normalized vector}, reusing cached vectors when hashes match."""
        if not pages:
            return {}

        collection = self._get_collection()
        urls = [p.url for p in pages]
        hashes = {p.url: p.content_hash() for p in pages}

        # pull whatever we already have for these urls
        existing = collection.get(ids=urls, include=["embeddings", "metadatas"])
        cached_vecs: dict[str, list[float]] = {}
        cached_hashes: dict[str, str] = {}
        got_ids = existing.get("ids") or []
        got_embs = existing.get("embeddings"); got_embs = [] if got_embs is None else got_embs
        got_meta = existing.get("metadatas") or []
        for i, uid in enumerate(got_ids):
            meta = got_meta[i] or {}
            cached_hashes[uid] = meta.get("content_hash", "")
            if i < len(got_embs) and got_embs[i] is not None:
                cached_vecs[uid] = list(got_embs[i])

        # decide which pages need re-encoding
        to_encode = [
            p for p in pages
            if p.url not in cached_vecs or cached_hashes.get(p.url) != hashes[p.url]
        ]

        vectors: dict[str, list[float]] = {}
        for p in pages:
            if p not in to_encode:
                vectors[p.url] = cached_vecs[p.url]

        if to_encode:
            model = self._get_model()
            texts = [self._embed_text(p) for p in to_encode]
            raw = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            new_ids, new_embs, new_meta = [], [], []
            for p, vec in zip(to_encode, raw):
                v = _normalize([float(x) for x in vec])
                vectors[p.url] = v
                new_ids.append(p.url)
                new_embs.append(v)
                new_meta.append({"content_hash": hashes[p.url], "url": p.url})
            collection.upsert(ids=new_ids, embeddings=new_embs, metadatas=new_meta)

        return vectors
