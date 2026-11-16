"""Compare five retrieval methods over the Day 6 B index."""

import argparse
import json
import os
import sys
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from baseline import hit, pack_context, validate
from retrieval import (
    DenseRetriever,
    HybridRetriever,
    QueryRewriteRetriever,
    TfidfReranker,
    TfidfRetriever,
)
from tracing import append_jsonl, sanitize_error_message

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TABLE = "day6_benchmark_chunks"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON from {path}: {exc}") from exc


def _load_day6(
    day6_starter: Path,
    database_url: str | None,
    model_path: str | None,
    expected_chunks: int,
):
    if not day6_starter.is_dir():
        raise FileNotFoundError(f"Day 6 starter directory not found: {day6_starter}")
    sys.path.insert(0, str(day6_starter))
    try:
        from dotenv import load_dotenv
        from src.embeddings import MyEmbedder
        from src.store import PgVectorStore
    except ImportError as exc:
        raise RuntimeError(
            "Run this script with the Day 6 virtual environment and its requirements."
        ) from exc

    load_dotenv(day6_starter / ".env")
    resolved_database_url = database_url or os.environ.get("DATABASE_URL")
    if not resolved_database_url:
        raise ValueError("Set DATABASE_URL or pass --database-url.")
    resolved_model_path = model_path or os.environ.get("EMBEDDING_MODEL")
    if not resolved_model_path:
        raise ValueError("Set EMBEDDING_MODEL or pass --embedding-model.")

    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(resolved_database_url, connect_timeout=5) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT current_setting('server_version') AS postgres_version,
                       (SELECT extversion FROM pg_extension WHERE extname = 'vector')
                           AS pgvector_version
                """
            )
            backend = cursor.fetchone()
            if not backend["pgvector_version"]:
                raise RuntimeError("The Day 6 pgvector extension is not available.")
            cursor.execute(
                f"""
                SELECT chunk_id, document_id, source, page, section, text, metadata
                FROM {SOURCE_TABLE}
                WHERE strategy = %s
                ORDER BY chunk_id
                """,
                ("B",),
            )
            rows = cursor.fetchall()
    if len(rows) != expected_chunks:
        raise RuntimeError(
            f"Expected {expected_chunks} B chunks; found {len(rows)}."
        )
    if any(row["document_id"] != "nist-ai-rmf-1-0" for row in rows):
        raise RuntimeError("The Day 6 B index contains an unexpected document.")
    device_setting = os.environ.get("EMBEDDING_DEVICE", "auto")
    device = None if device_setting.casefold() == "auto" else device_setting
    embedder = MyEmbedder(
        resolved_model_path,
        batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "8")),
        device=device,
    )
    if embedder.dimensions != 768:
        raise RuntimeError(
            f"Day 6 index expects 768 dimensions; loaded {embedder.dimensions}."
        )
    store = PgVectorStore(resolved_database_url, dimensions=embedder.dimensions)
    chunks = [
        {
            "id": row["chunk_id"],
            "source_id": row["document_id"],
            "source": row["source"],
            "page": row["page"],
            "text": row["text"],
            "metadata": {**row["metadata"], "section": row["section"]},
        }
        for row in rows
    ]
    if not chunks:
        raise RuntimeError("The Day 6 vector table has no strategy B chunks.")
    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("The Day 6 B index contains duplicate chunk IDs.")
    return embedder, store, chunks, backend, resolved_database_url


def _evidence_metrics(
    results: list[dict[str, Any]],
    context_ids: list[str],
    gold: list[dict[str, Any]],
) -> dict[str, bool]:
    hit1 = bool(results and hit(results[0], gold))
    hit3 = any(hit(result, gold) for result in results[:3])
    context_hit = any(
        result["id"] in context_ids and hit(result, gold) for result in results
    )
    return {"hit1": hit1, "evidence_hit_at_3": hit3, "context_hit": context_hit}


def _run_method(
    method: Any,
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    split: str,
    output_dir: Path,
    top_k: int,
    budget: int,
    database_url: str,
    index_build_ms: float = 0.0,
) -> dict[str, Any]:
    traces_path = output_dir / method.name / "traces.jsonl"
    answerable = [question for question in questions if question["answerable"]]
    method_started = time.perf_counter()
    validate(chunks, answerable)
    counters = {
        "hit1": 0,
        "evidence_hit_at_3": 0,
        "context_hit": 0,
        "errors": 0,
    }
    search_times: list[float] = []
    for question in questions:
        started = time.perf_counter()
        try:
            result = method.search_with_trace(
                question["question"],
                filters=question.get("filter"),
                top_k=top_k,
            )
            context_started = time.perf_counter()
            context, context_ids = pack_context(
                result["results"], budget=budget
            )
            context_ms = (time.perf_counter() - context_started) * 1000
            metrics = (
                _evidence_metrics(result["results"], context_ids, question["gold"])
                if question["answerable"]
                else {
                    "hit1": None,
                    "evidence_hit_at_3": None,
                    "context_hit": None,
                }
            )
            trace = {
                "run_id": output_dir.name,
                "query_id": question["id"],
                "split": split,
                "answerable": question["answerable"],
                "method": method.name,
                "question": question["question"],
                "original_query": result["original_query"],
                "retrieval_queries": result["retrieval_queries"],
                "filters": result["filters"],
                "parameters": {
                    **result["parameters"],
                    "context_budget_chars": budget,
                },
                "candidate_count": result["candidate_count"],
                "results": result["results"],
                "context": context,
                "context_ids": context_ids,
                "context_chars": len(context),
                "search_ms": result["elapsed_ms"],
                "context_selection_ms": context_ms,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                **metrics,
                "status": "ok",
            }
            search_times.append(trace["search_ms"])
            if question["answerable"]:
                for key in ("hit1", "evidence_hit_at_3", "context_hit"):
                    counters[key] += metrics[key]
        except Exception as exc:
            counters["errors"] += 1
            message = str(exc).replace(database_url, "[redacted database URL]")
            message = sanitize_error_message(message)
            trace = {
                "run_id": output_dir.name,
                "query_id": question["id"],
                "split": split,
                "answerable": question["answerable"],
                "method": method.name,
                "question": question["question"],
                "original_query": question["question"],
                "retrieval_queries": [],
                "filters": question.get("filter"),
                "parameters": {
                    "top_k": top_k,
                    "context_budget_chars": budget,
                },
                "status": "error",
                "error": {"type": type(exc).__name__, "message": message},
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        append_jsonl(traces_path, trace)

    denominator = len(answerable)
    summary = {
        "method": method.name,
        "split": split,
        "questions": len(questions),
        "answerable_n": denominator,
        "unanswerable_n": len(questions) - denominator,
        "hit1": {
            "n": counters["hit1"],
            "N": denominator,
            "rate": counters["hit1"] / denominator if denominator else None,
        },
        "evidence_hit_at_3": {
            "n": counters["evidence_hit_at_3"],
            "N": denominator,
            "rate": counters["evidence_hit_at_3"] / denominator if denominator else None,
        },
        "context_evidence_rate": {
            "n": counters["context_hit"],
            "N": denominator,
            "rate": counters["context_hit"] / denominator if denominator else None,
        },
        "median_search_ms": statistics.median(search_times) if search_times else None,
        "errors": counters["errors"],
        "local_service_cost_usd": 0,
        "index_build_ms": index_build_ms,
        "elapsed_seconds": time.perf_counter() - method_started,
    }
    summary_path = output_dir / method.name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dense, TF-IDF, hybrid, rerank and query rewrite."
    )
    parser.add_argument("--day6-starter", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "benchmark.json")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "starter" / "results" / "retrieval")
    parser.add_argument("--database-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--expected-chunks", type=int, default=188)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--context-budget", type=int, default=800)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-candidates", type=int, default=30)
    args = parser.parse_args()
    if args.top_k <= 0 or args.context_budget <= 0 or args.expected_chunks <= 0:
        raise ValueError("top-k, context budget, and expected chunk count must be positive.")
    questions = _read_json(args.benchmark)
    if not isinstance(questions, list):
        raise ValueError("benchmark must be a JSON array.")
    selected = [question for question in questions if question.get("split") == args.split]
    if not selected:
        raise ValueError(f"No {args.split} questions in the benchmark.")
    if any(not isinstance(question.get("answerable"), bool) for question in selected):
        raise ValueError("Every benchmark case must include an answerable boolean.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + str(uuid4())[:8]
    output_dir = args.output / f"{args.split}-{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        embedder, store, chunks, backend, database_url = _load_day6(
            args.day6_starter,
            args.database_url,
            args.embedding_model,
            args.expected_chunks,
        )
        if len(chunks) != args.expected_chunks:
            raise RuntimeError(
                f"Expected {args.expected_chunks} B chunks; found {len(chunks)}."
            )
        if any(chunk["source_id"] != "nist-ai-rmf-1-0" for chunk in chunks):
            raise RuntimeError("The Day 6 B index contains an unexpected document.")
        dense = DenseRetriever(
            store, embedder.embed_query, strategy="B", dimensions=embedder.dimensions
        )
        methods = []
        for build in (
            lambda: dense,
            lambda: TfidfRetriever(chunks),
            lambda: HybridRetriever(dense, chunks, rrf_k=args.rrf_k),
            lambda: TfidfReranker(
                dense, chunks, candidate_k=args.rerank_candidates
            ),
            lambda: QueryRewriteRetriever(dense),
        ):
            build_started = time.perf_counter()
            method = build()
            methods.append(
                (method, (time.perf_counter() - build_started) * 1000)
            )
    except Exception as exc:
        message = str(exc)
        database_secret = args.database_url or os.environ.get("DATABASE_URL")
        if database_secret:
            message = message.replace(database_secret, "[redacted database URL]")
        message = sanitize_error_message(message)
        append_jsonl(
            output_dir / "errors.jsonl",
            {
                "run_id": run_id,
                "split": args.split,
                "stage": "initialize",
                "error": {"type": type(exc).__name__, "message": message},
            },
        )
        raise

    summaries = [
        _run_method(
            method,
            chunks,
            selected,
            split=args.split,
            output_dir=output_dir,
            top_k=args.top_k,
            budget=args.context_budget,
            database_url=database_url,
            index_build_ms=index_build_ms,
        )
        for method, index_build_ms in methods
    ]
    metadata = {
        "run_id": run_id,
        "split": args.split,
        "benchmark": str(args.benchmark.resolve()),
        "corpus_source_id": "nist-ai-rmf-1-0",
        "index_strategy": "B",
        "chunk_count": len(chunks),
        "vector_store": {
            "postgres_version": backend["postgres_version"],
            "pgvector_version": backend["pgvector_version"],
            "table": SOURCE_TABLE,
            "search": "exact cosine",
        },
        "embedding": {
            "model": embedder.model_name,
            "dimensions": embedder.dimensions,
            "device": embedder.device,
            "query_prompt": "SearchQuery",
            "normalized": True,
            "cost_usd": 0,
        },
        "config": {
            "top_k": args.top_k,
            "context_budget_chars": args.context_budget,
            "rrf_k": args.rrf_k,
            "rerank_candidates": args.rerank_candidates,
        },
        "methods": [summary["method"] for summary in summaries],
        "summaries": summaries,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if any(summary["errors"] for summary in summaries):
        raise RuntimeError("One or more retrieval queries failed; see trace files.")


if __name__ == "__main__":
    main()
