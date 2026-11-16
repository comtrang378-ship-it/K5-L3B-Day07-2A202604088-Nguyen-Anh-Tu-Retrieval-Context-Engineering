"""Offline tests for retrieval method composition and trace fields."""

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from retrieval import (
    DenseRetriever,
    HybridRetriever,
    QueryRewriteRetriever,
    TfidfReranker,
    TfidfRetriever,
    keyword_rewrite,
)
from retrieval_benchmark import _run_method
from tracing import sanitize_error_message


def chunk(chunk_id, text, section="5.1 GOVERN"):
    return {
        "id": chunk_id,
        "source_id": "doc",
        "source": "doc.pdf",
        "page": 1,
        "text": text,
        "metadata": {"section": section},
    }


class FakeStore:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, strategy, query_vector, top_k=3, filters=None):
        self.calls.append((strategy, query_vector, top_k, filters))
        rows = self.rows
        if filters:
            rows = [
                row for row in rows
                if row["metadata"]["section"] == filters.get("section")
            ]
        return [
            {
                "chunk_id": row["id"],
                "document_id": row["source_id"],
                "source": row["source"],
                "page": row["page"],
                "section": row["metadata"]["section"],
                "text": row["text"],
                "metadata": row["metadata"],
                "score": row["dense_score"],
            }
            for row in rows[:top_k]
        ]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk("a", "banana yellow fruit", "5.1 GOVERN"),
            chunk("b", "apple apple red fruit", "5.2 MAP"),
            chunk("c", "apple green vegetable", "5.1 GOVERN"),
        ]

    def dense(self, rows=None):
        return DenseRetriever(
            FakeStore(rows or [
                {**self.chunks[0], "dense_score": 0.9},
                {**self.chunks[1], "dense_score": 0.8},
                {**self.chunks[2], "dense_score": 0.7},
            ]),
            lambda query: [0.25, 0.75],
            dimensions=2,
        )

    def test_dense_reuses_day6_strategy_and_normalizes_chunk_contract(self):
        dense = self.dense()
        trace = dense.search_with_trace("fruit", filters={"section": "5.1 GOVERN"})
        self.assertEqual(trace["method"], "dense")
        self.assertEqual(trace["results"][0]["id"], "a")
        self.assertEqual(trace["results"][0]["rank"], 1)
        self.assertEqual(dense.store.calls[0][0], "B")
        self.assertEqual(dense.store.calls[0][3], {"section": "5.1 GOVERN"})

    def test_tfidf_filter_is_applied_before_ranking(self):
        trace = TfidfRetriever(self.chunks).search_with_trace(
            "apple", filters={"section": "5.1 GOVERN"}
        )
        self.assertEqual({row["id"] for row in trace["results"]}, {"c"})

    def test_hybrid_fuses_dense_and_bm25_ranks(self):
        dense_rows = [
            {**self.chunks[0], "dense_score": 0.9},
            {**self.chunks[1], "dense_score": 0.8},
            {**self.chunks[2], "dense_score": 0.7},
        ]
        bm25_rows = [
            {**self.chunks[1], "score": 4.0},
            {**self.chunks[2], "score": 2.0},
            {**self.chunks[0], "score": 1.0},
        ]
        fake_index = type("FakeBm25", (), {"rank": lambda _self, _query, _filters: bm25_rows})()
        with patch("retrieval._Bm25Index", return_value=fake_index):
            hybrid = HybridRetriever(self.dense(dense_rows), self.chunks, rrf_k=1)
        trace = hybrid.search_with_trace("apple fruit", top_k=3)
        self.assertEqual(trace["results"][0]["id"], "b")
        self.assertIn("dense_rank", trace["results"][0])
        self.assertIn("bm25_rank", trace["results"][0])
        self.assertEqual(trace["parameters"]["rrf_k"], 1)

    def test_tfidf_reranker_reorders_only_dense_candidates(self):
        dense_rows = [
            {**self.chunks[0], "dense_score": 0.9},
            {**self.chunks[1], "dense_score": 0.8},
        ]
        reranker = TfidfReranker(
            self.dense(dense_rows), self.chunks, candidate_k=2
        )
        trace = reranker.search_with_trace("apple", top_k=2)
        self.assertEqual(trace["results"][0]["id"], "b")
        self.assertEqual(trace["candidate_count"], 2)
        self.assertIn("rerank_score", trace["results"][0])

    def test_keyword_rewrite_preserves_negation_and_content(self):
        question = "What does the AI RMF not require before deployment?"
        rewritten = keyword_rewrite(question)
        self.assertEqual(rewritten, "ai rmf not require deployment")
        trace = QueryRewriteRetriever(self.dense()).search_with_trace(question)
        self.assertEqual(trace["original_query"], question)
        self.assertEqual(trace["retrieval_queries"], [rewritten])
        self.assertEqual(trace["method"], "dense_keyword_rewrite")

    def test_dense_rejects_embedding_dimension_mismatch(self):
        dense = DenseRetriever(FakeStore([]), lambda _query: [1.0], dimensions=2)
        with self.assertRaisesRegex(ValueError, "2 finite dimensions"):
            dense.search("query")

    def test_filters_reject_untrusted_fields(self):
        with self.assertRaisesRegex(ValueError, "Unsupported filter fields"):
            TfidfRetriever(self.chunks).search(
                "apple", filters={"source_id": "attacker-controlled"}
            )

    def test_trace_error_sanitizer_redacts_common_credentials(self):
        message = (
            "api_key=private Bearer abc123 sk-12345678901234567890 "
            "postgresql://day6:privatepass@127.0.0.1/day6"
        )
        sanitized = sanitize_error_message(message)
        self.assertNotIn("private", sanitized)
        self.assertNotIn("abc123", sanitized)
        self.assertNotIn("sk-12345678901234567890", sanitized)
        self.assertNotIn("privatepass", sanitized)

    def test_benchmark_excludes_unanswerable_cases_from_retrieval_metrics(self):
        answerable = {
            "id": "q1",
            "split": "dev",
            "question": "What fruit is red?",
            "answerable": True,
            "expected_answer": "apple",
            "gold": [{"source_id": "doc", "page": 1, "evidence": "apple apple"}],
        }
        unanswerable = {
            "id": "q2",
            "split": "dev",
            "question": "What is the budget?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "No budget is specified.",
        }

        class StaticRetriever:
            name = "static"

            def search_with_trace(self, query, *, filters, top_k):
                result = {**self_result, "score": 1.0, "rank": 1}
                return {
                    "method": self.name,
                    "original_query": query,
                    "retrieval_queries": [query],
                    "filters": filters,
                    "parameters": {"top_k": top_k},
                    "candidate_count": 1,
                    "results": [result],
                    "elapsed_ms": 1.0,
                }

        self_result = {
            **self.chunks[1],
            "text": "apple apple",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = _run_method(
                StaticRetriever(),
                self.chunks,
                [answerable, unanswerable],
                split="dev",
                output_dir=output,
                top_k=3,
                budget=800,
                database_url="local-only",
            )
            traces = [
                json.loads(line)
                for line in (output / "static" / "traces.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(summary["questions"], 2)
        self.assertEqual(summary["evidence_hit_at_3"]["N"], 1)
        self.assertEqual(summary["evidence_hit_at_3"]["n"], 1)
        self.assertIsNone(traces[1]["evidence_hit_at_3"])
        self.assertNotIn("gold", traces[1])


if __name__ == "__main__":
    unittest.main()
