"""Five comparable retrieval methods over Day 6 chunks and vectors."""

import math
import re
import time
import unicodedata
from typing import Any, Callable, Protocol

from baseline import TfidfIndex, tokenize


class DenseStore(Protocol):
    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...


class Retriever(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        ...

    def search_with_trace(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        ...


def _validate_search(query: str, filters: dict[str, Any] | None, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if filters:
        unsupported = set(filters) - {"section"}
        if unsupported:
            raise ValueError(f"Unsupported filter fields: {sorted(unsupported)}")
        for key, value in filters.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Filter {key!r} must be a non-empty string.")


def _apply_filter(
    chunks: list[dict[str, Any]], filters: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not filters:
        return chunks
    return [
        chunk
        for chunk in chunks
        if all(chunk.get("metadata", {}).get(key) == value for key, value in filters.items())
    ]


def _ranked(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for rank, result in enumerate(results, start=1):
        ranked.append({**result, "rank": rank})
    return ranked


def normalize_day6_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Map the Day 6 store response to the Day 7 chunk contract."""
    metadata = hit.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("Day 6 chunk metadata must be an object.")
    section = hit.get("section")
    if section is not None:
        metadata = {**metadata, "section": section}
    return {
        "id": hit["chunk_id"],
        "source_id": hit["document_id"],
        "source": hit["source"],
        "page": hit["page"],
        "text": hit["text"],
        "metadata": metadata,
        "score": float(hit["score"]),
    }


class DenseRetriever:
    """Exact cosine search against the Day 6 pgvector index."""

    name = "dense"

    def __init__(
        self,
        store: DenseStore,
        embed_query: Callable[[str], list[float]],
        *,
        strategy: str = "B",
        dimensions: int = 768,
    ) -> None:
        if not strategy.strip():
            raise ValueError("strategy must not be empty.")
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")
        self.store = store
        self.embed_query = embed_query
        self.strategy = strategy
        self.dimensions = dimensions

    def search_with_trace(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        _validate_search(query, filters, top_k)
        started = time.perf_counter()
        vector = self.embed_query(query)
        if len(vector) != self.dimensions or not all(
            math.isfinite(float(value)) for value in vector
        ):
            raise ValueError(
                f"Query embedding must contain {self.dimensions} finite dimensions."
            )
        raw = self.store.search(
            self.strategy, vector, top_k=top_k, filters=filters
        )
        normalized = [normalize_day6_hit(hit) for hit in raw]
        normalized.sort(key=lambda hit: (-hit["score"], hit["id"]))
        results = _ranked(normalized)
        return {
            "method": self.name,
            "original_query": query,
            "retrieval_queries": [query],
            "filters": filters,
            "parameters": {
                "strategy": self.strategy,
                "top_k": top_k,
                "embedding_dimensions": self.dimensions,
                "metric": "exact cosine",
            },
            "candidate_count": len(results),
            "results": results,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        return self.search_with_trace(query, filters=filters, top_k=top_k)["results"]


class TfidfRetriever:
    """Offline lexical retrieval using the Day 7 TF-IDF baseline."""

    name = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("TF-IDF requires at least one chunk.")
        self.chunks = chunks
        self.index = TfidfIndex(chunks)

    def search_with_trace(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        _validate_search(query, filters, top_k)
        started = time.perf_counter()
        results = _ranked(
            self.index.search(query, top_k=top_k, filters=filters)
        )
        return {
            "method": self.name,
            "original_query": query,
            "retrieval_queries": [query],
            "filters": filters,
            "parameters": {"top_k": top_k, "ranking": "TF-IDF cosine"},
            "candidate_count": len(results),
            "results": results,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        return self.search_with_trace(query, filters=filters, top_k=top_k)["results"]


def _bm25_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"\w+", normalized)


class _Bm25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk.")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "Hybrid retrieval requires the Day 6 rank-bm25 dependency."
            ) from exc
        self.chunks = chunks
        tokenized = [_bm25_tokens(chunk["text"]) for chunk in chunks]
        if any(not terms for terms in tokenized):
            raise ValueError("Every BM25 chunk must contain searchable tokens.")
        self.index = BM25Okapi(tokenized)

    def rank(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        terms = _bm25_tokens(query)
        if not terms:
            raise ValueError("query contains no searchable BM25 tokens.")
        eligible = _apply_filter(self.chunks, filters)
        eligible_ids = {chunk["id"] for chunk in eligible}
        scores = self.index.get_scores(terms)
        indexed = [
            (chunk, float(scores[index]))
            for index, chunk in enumerate(self.chunks)
            if chunk["id"] in eligible_ids
        ]
        indexed.sort(key=lambda item: (-item[1], item[0]["id"]))
        return [
            {**chunk, "score": score}
            for chunk, score in indexed
        ]


class HybridRetriever:
    """Fuse full-corpus dense and BM25 rankings with reciprocal-rank fusion."""

    name = "hybrid_rrf"

    def __init__(
        self,
        dense: DenseRetriever,
        chunks: list[dict[str, Any]],
        *,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")
        self.dense = dense
        self.chunks = chunks
        self.bm25 = _Bm25Index(chunks)
        self.rrf_k = rrf_k

    def search_with_trace(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        _validate_search(query, filters, top_k)
        started = time.perf_counter()
        eligible_count = len(_apply_filter(self.chunks, filters))
        if not eligible_count:
            raise ValueError("The requested section filter matches no chunks.")
        dense_trace = self.dense.search_with_trace(
            query, filters=filters, top_k=eligible_count
        )
        dense_hits = dense_trace["results"]
        bm25_hits = self.bm25.rank(query, filters)
        dense_by_id = {hit["id"]: hit for hit in dense_hits}
        bm25_by_id = {hit["id"]: hit for hit in bm25_hits}
        if dense_by_id.keys() != bm25_by_id.keys():
            raise RuntimeError("Dense and BM25 rankings have different candidates.")
        bm25_ranks = {
            hit["id"]: rank for rank, hit in enumerate(bm25_hits, start=1)
        }
        fused = []
        for dense_rank, hit in enumerate(dense_hits, start=1):
            bm25_rank = bm25_ranks[hit["id"]]
            rrf_score = 1 / (self.rrf_k + dense_rank) + 1 / (
                self.rrf_k + bm25_rank
            )
            fused.append(
                {
                    **hit,
                    "dense_rank": dense_rank,
                    "bm25_rank": bm25_rank,
                    "dense_score": hit["score"],
                    "bm25_score": bm25_by_id[hit["id"]]["score"],
                    "score": rrf_score,
                    "rrf_score": rrf_score,
                }
            )
        fused.sort(key=lambda hit: (-hit["rrf_score"], hit["id"]))
        results = _ranked(fused[:top_k])
        return {
            "method": self.name,
            "original_query": query,
            "retrieval_queries": [query],
            "filters": filters,
            "parameters": {
                "top_k": top_k,
                "candidate_count_per_retriever": eligible_count,
                "rrf_k": self.rrf_k,
                "fusion": "1/(rrf_k+dense_rank) + 1/(rrf_k+bm25_rank)",
                "bm25_tokenizer": "Unicode NFC, casefold, Unicode word boundaries",
            },
            "candidate_count": len(fused),
            "results": results,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        return self.search_with_trace(query, filters=filters, top_k=top_k)["results"]


class TfidfReranker:
    """Retrieve a dense candidate pool, then rerank it by TF-IDF query relevance."""

    name = "dense_tfidf_rerank"

    def __init__(
        self,
        dense: DenseRetriever,
        chunks: list[dict[str, Any]],
        *,
        candidate_k: int = 30,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")
        self.dense = dense
        self.chunks = chunks
        self.candidate_k = candidate_k
        self.lexical_index = TfidfIndex(chunks)
        self.by_id = {chunk["id"]: chunk for chunk in chunks}

    def search_with_trace(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        _validate_search(query, filters, top_k)
        started = time.perf_counter()
        eligible_count = len(_apply_filter(self.chunks, filters))
        candidate_k = min(self.candidate_k, eligible_count)
        if candidate_k == 0:
            raise ValueError("The requested section filter matches no chunks.")
        dense_trace = self.dense.search_with_trace(
            query, filters=filters, top_k=candidate_k
        )
        candidates = dense_trace["results"]
        reranked = []
        for dense_rank, candidate in enumerate(candidates, start=1):
            score = self.lexical_index.score(query, self.by_id[candidate["id"]])
            reranked.append(
                {
                    **candidate,
                    "dense_rank": dense_rank,
                    "dense_score": candidate["score"],
                    "rerank_score": score,
                    "score": score,
                }
            )
        reranked.sort(
            key=lambda hit: (-hit["rerank_score"], hit["dense_rank"], hit["id"])
        )
        results = _ranked(reranked[:top_k])
        return {
            "method": self.name,
            "original_query": query,
            "retrieval_queries": [query],
            "filters": filters,
            "parameters": {
                "top_k": top_k,
                "dense_candidate_k": candidate_k,
                "reranker": "TF-IDF cosine over dense candidate pool",
            },
            "candidate_count": len(candidates),
            "results": results,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        return self.search_with_trace(query, filters=filters, top_k=top_k)["results"]


_QUESTION_FILLERS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "can",
    "could", "does", "do", "for", "from", "how", "in", "into", "is",
    "it", "of", "on", "or", "should", "that", "the", "their", "this",
    "to", "under", "what", "when", "where", "which", "who", "why",
    "would",
}


def keyword_rewrite(question: str) -> str:
    """Remove interrogative/function words while retaining question content terms."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    terms = [term for term in tokenize(question) if term not in _QUESTION_FILLERS]
    if not terms:
        raise ValueError("Question rewrite produced no searchable terms.")
    return " ".join(terms)


class QueryRewriteRetriever:
    """Dense retrieval with a transparent, deterministic keyword-focused rewrite."""

    name = "dense_keyword_rewrite"

    def __init__(self, dense: DenseRetriever) -> None:
        self.dense = dense

    def search_with_trace(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        _validate_search(query, filters, top_k)
        started = time.perf_counter()
        rewritten = keyword_rewrite(query)
        dense_trace = self.dense.search_with_trace(
            rewritten, filters=filters, top_k=top_k
        )
        return {
            **dense_trace,
            "method": self.name,
            "original_query": query,
            "retrieval_queries": [rewritten],
            "parameters": {
                **dense_trace["parameters"],
                "rewrite": "drop interrogative/function words; retain content and negation terms",
            },
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        return self.search_with_trace(query, filters=filters, top_k=top_k)["results"]
