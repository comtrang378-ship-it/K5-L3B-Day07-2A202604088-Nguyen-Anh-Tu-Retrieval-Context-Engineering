"""Run the local model agent on a benchmark split and save auditable traces."""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentic import run_agent
from baseline import hit, validate
from local_controller import LocalTransformersController
from retrieval import DenseRetriever
from retrieval_benchmark import _load_day6
from tracing import append_jsonl, sanitize_error_message

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON from {path}: {exc}") from exc


def _metrics(
    case: dict[str, Any], trace: dict[str, Any]
) -> dict[str, bool | None]:
    if not case["answerable"]:
        return {"hit1": None, "evidence_hit_at_3": None, "context_hit": None}

    searches = [
        event
        for event in trace["events"]
        if event.get("event") == "search" and event.get("status") == "ok"
    ]
    if not searches:
        return {"hit1": None, "evidence_hit_at_3": None, "context_hit": None}
    first_results = searches[0]["results"] if searches else []
    context_ids = set(trace.get("context_ids", []))
    all_results = [
        result
        for event in searches
        for result in event.get("results", [])
    ]
    context_results = [
        result for result in all_results if result["id"] in context_ids
    ]
    return {
        "hit1": bool(first_results and hit(first_results[0], case["gold"])),
        "evidence_hit_at_3": any(
            hit(result, case["gold"]) for result in first_results[:3]
        ),
        "context_hit": any(
            hit(result, case["gold"]) for result in context_results
        ),
    }


def _rate(successes: int, total: int) -> dict[str, Any]:
    return {
        "n": successes,
        "N": total,
        "rate": successes / total if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the local Transformers agent on dev or test."
    )
    parser.add_argument("--day6-starter", type=Path, required=True)
    parser.add_argument(
        "--benchmark", type=Path, default=ROOT / "data" / "benchmark.json"
    )
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "starter" / "results" / "agent"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-chunks", type=int, default=188)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--context-budget", type=int, default=800)
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N selected cases for smoke checks; not a full benchmark.",
    )
    args = parser.parse_args()
    if min(
        args.expected_chunks,
        args.top_k,
        args.context_budget,
        args.max_searches,
        args.max_new_tokens,
    ) <= 0:
        raise ValueError("All count, budget, and token arguments must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be greater than zero.")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {args.model_path}")

    benchmark = _read_json(args.benchmark)
    if not isinstance(benchmark, list):
        raise ValueError("Benchmark must be a JSON array.")
    selected = [case for case in benchmark if case.get("split") == args.split]
    if not selected:
        raise ValueError(f"No {args.split} questions in the benchmark.")
    if any(not isinstance(case.get("answerable"), bool) for case in selected):
        raise ValueError("Every benchmark case must include an answerable boolean.")
    cases = selected[: args.limit] if args.limit is not None else selected

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + str(uuid4())[:8]
    )
    output_dir = args.output / f"{args.split}-{run_id}"
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir.mkdir()
    trace_path = output_dir / "traces.jsonl"

    embedder, store, chunks, backend, _ = _load_day6(
        args.day6_starter, None, None, args.expected_chunks
    )
    answerable = [case for case in cases if case["answerable"]]
    validate(chunks, answerable)
    retriever = DenseRetriever(
        store,
        embedder.embed_query,
        strategy="B",
        dimensions=embedder.dimensions,
    )
    controller = LocalTransformersController(
        args.model_path, max_new_tokens=args.max_new_tokens
    )

    statuses = {"answered": 0, "abstain": 0, "error": 0}
    metric_counts = {"hit1": 0, "evidence_hit_at_3": 0, "context_hit": 0}
    retrieval_evaluable_n = 0
    answerable_count = len(answerable)
    tool_use = 0
    elapsed_times: list[float] = []
    cost_usd = 0.0
    traces: list[dict[str, Any]] = []
    review_template = []
    error_summaries = []

    for case in cases:
        raw_trace_path = raw_dir / f"{case['id']}.jsonl"
        started = time.perf_counter()
        try:
            run_agent(
                case["question"],
                controller,
                retriever,
                filters=case.get("filter"),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.context_budget,
                trace_path=raw_trace_path,
            )
        except Exception as exc:
            message = sanitize_error_message(str(exc))
            error_summaries.append(
                {"query_id": case["id"], "type": type(exc).__name__, "message": message}
            )
        elapsed_times.append(time.perf_counter() - started)

        raw_lines = raw_trace_path.read_text(encoding="utf-8").splitlines()
        if len(raw_lines) != 1:
            raise RuntimeError(
                f"Expected one persisted trace for {case['id']}; found {len(raw_lines)}."
            )
        trace = json.loads(raw_lines[0])
        trace["query_id"] = case["id"]
        trace["split"] = args.split
        trace["answerable"] = case["answerable"]
        trace["retrieval_metrics"] = _metrics(case, trace)
        append_jsonl(trace_path, trace)
        traces.append(trace)
        raw_trace_path.unlink()
        print(
            f"[{len(traces)}/{len(cases)}] {case['id']}: {trace['status']}",
            flush=True,
        )

        statuses[trace["status"]] += 1
        if trace["status"] != "error":
            tool_use += trace["tool_call_count"] > 0
            cost_usd += trace["reported_cost_usd"] or 0
        if case["answerable"]:
            if trace["retrieval_metrics"]["hit1"] is not None:
                retrieval_evaluable_n += 1
                for name in metric_counts:
                    metric_counts[name] += bool(trace["retrieval_metrics"][name])

        citations = trace.get("citations", [])
        review_template.append(
            {
                "query_id": case["id"],
                "answer_grade": None,
                "citation_support": [None for _ in citations],
                "abstention_correct": None,
            }
        )

    raw_dir.rmdir()
    (output_dir / "cases.json").write_text(
        json.dumps(
            [
                {
                    "id": case["id"],
                    "split": args.split,
                    "partial_split": len(cases) != len(selected),
                    "selected_case_count": len(cases),
                    "full_split_case_count": len(selected),
                    "answerable": case["answerable"],
                }
                for case in cases
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "reviews_template.json").write_text(
        json.dumps(review_template, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "run_id": run_id,
        "split": args.split,
        "corpus_chunk_count": len(chunks),
        "vector_store": {
            "postgres_version": backend["postgres_version"],
            "pgvector_version": backend["pgvector_version"],
            "strategy": "B",
            "search": "exact cosine",
        },
        "controller": {
            "provider": "local_transformers",
            "model": controller.model_name,
            "path": str(args.model_path),
            "device": "cpu",
            "max_new_tokens": args.max_new_tokens,
            "cost_usd": 0,
        },
        "retrieval": {
            "method": "dense",
            "top_k": args.top_k,
            "context_budget_chars": args.context_budget,
            "max_searches": args.max_searches,
            "metric_scope": "first successful search top-3; evidence in final context",
            "retrieval_evaluable_n": retrieval_evaluable_n,
            "hit1": _rate(metric_counts["hit1"], retrieval_evaluable_n),
            "evidence_hit_at_3": _rate(
                metric_counts["evidence_hit_at_3"], retrieval_evaluable_n
            ),
            "context_evidence_rate": _rate(
                metric_counts["context_hit"], retrieval_evaluable_n
            ),
        },
        "agent": {
            "questions": len(cases),
            "answerable_n": answerable_count,
            "unanswerable_n": len(cases) - answerable_count,
            "status_counts": statuses,
            "tool_use": _rate(tool_use, len(cases) - statuses["error"]),
            "median_elapsed_seconds": (
                statistics.median(elapsed_times) if elapsed_times else None
            ),
            "elapsed_seconds": sum(elapsed_times),
            "local_cost_usd": cost_usd,
            "answer_quality": "Requires manual review; see reviews_template.json.",
            "errors": error_summaries,
        },
        "outputs": {
            "traces": "traces.jsonl",
            "traces_json": "traces.json",
            "cases": "cases.json",
            "reviews_template": "reviews_template.json",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if error_summaries:
        raise RuntimeError("Agent benchmark had errors; see the saved traces.")


if __name__ == "__main__":
    main()
