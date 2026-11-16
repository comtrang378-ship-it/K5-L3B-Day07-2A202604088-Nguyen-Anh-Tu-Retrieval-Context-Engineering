"""Agent loop with provider-specific controllers and Day 6-compatible retrievers.

Controller adapters remain provider-specific; never use fake adapters to report
benchmark quality.
"""
from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Any, Protocol
from uuid import uuid4

from baseline import pack_context
from tracing import append_jsonl, sanitize_error_message


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action.

        Search: {'action': 'search', 'query': '...'}.
        Final: {'action': 'answer', 'answer': '...', 'citations': ['chunk-id']}.
        Citations must also appear as [chunk-id] in the answer text.
        """
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the real Day 6 vector store with a compatible query embedding.

        Return each chunk's id, text, source_id, source, page, metadata and score.
        The query embedding model and dimension must match the document index.
        """
        ...


def run_agent(question: str, controller: Controller, retriever: Retriever, *,
              filters: dict[str, Any] | None = None, max_searches: int = 4,
              top_k: int = 3, budget: int = 800,
              trace_path: str | Path | None = None) -> dict[str, Any]:
    """Return answer, citation IDs and trace; no benchmark gold enters this loop.

    A final answer without valid citations, an empty context, and a search limit
    lead to abstention. This only checks citation IDs, not semantic grounding.
    Provider/retriever errors are persisted when trace_path is set, then propagate.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    trace_id = str(uuid4())
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    stage = "controller"

    def finish(
        status: str,
        answer: str,
        citations: list[str],
        *,
        error: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        finished_at = datetime.now(timezone.utc)
        reported_costs = []
        for event in events:
            model_details = event.get("controller_model")
            if model_details is None:
                model_details = event.get("tool_call", {}).get(
                    "controller_model", {}
                )
            cost = model_details.get("cost_usd") if isinstance(model_details, dict) else None
            if isinstance(cost, (int, float)) and math.isfinite(float(cost)) and cost >= 0:
                reported_costs.append(float(cost))
        record = {
            "trace_id": trace_id,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "question": question,
            "filters": filters,
            "retriever": getattr(retriever, "name", type(retriever).__name__),
            "tool_call_count": sum(
                event.get("event") == "search" for event in events
            ),
            "reported_cost_usd": (
                sum(reported_costs) if reported_costs else None
            ),
            "parameters": {
                "max_searches": max_searches,
                "top_k": top_k,
                "context_budget_chars": budget,
            },
            "status": status,
            "events": events,
            "context": next(
                (event["context"] for event in reversed(events)
                 if event.get("event") == "search" and "context" in event),
                "",
            ),
            "context_ids": next(
                (event["context_ids"] for event in reversed(events)
                 if event.get("event") == "search" and "context_ids" in event),
                [],
            ),
            "answer": answer,
            "citations": citations,
            "error": error,
        }
        if trace_path is not None:
            append_jsonl(trace_path, record)
        return {
            "status": status,
            "answer": answer,
            "citations": citations,
            "events": events,
            "trace_id": trace_id,
        }

    try:
        for _ in range(max_searches + 1):
            stage = "controller"
            action = controller.decide(question, history)
            if action.get('action') == 'search':
                if sum(e['event'] == 'search' for e in events) >= max_searches:
                    events.append({'event': 'search_limit'})
                    return finish(
                        'abstain', 'Không đủ bằng chứng để trả lời.', []
                    )
                query = action.get('query')
                if not isinstance(query, str) or not query.strip():
                    raise ValueError('search query phải là chuỗi không rỗng')

                event = {
                    'event': 'search',
                    'status': 'started',
                    'tool_call': {
                        'name': 'retrieval.search',
                        'query': query,
                        'filters': filters,
                        'top_k': top_k,
                        'controller_model': {
                            key: action[key]
                            for key in ('provider', 'model', 'usage', 'cost_usd')
                            if key in action
                        },
                    },
                    'query': query,
                    'filter': filters,
                    'top_k': top_k,
                }
                events.append(event)
                stage = "retriever"
                retrieval_started = time.perf_counter()
                search_with_trace = getattr(retriever, "search_with_trace", None)
                retrieval_trace = None
                if callable(search_with_trace):
                    retrieval_trace = search_with_trace(
                        query, filters=filters, top_k=top_k
                    )
                    results = retrieval_trace.get("results")
                else:
                    results = retriever.search(query, filters=filters, top_k=top_k)
                retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
                if not isinstance(results, list):
                    raise ValueError('retriever phải trả về danh sách kết quả')
                if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                    raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
                candidates.extend(results)
                stage = "context"
                context, ids = pack_context(candidates, budget=budget)
                chunks = {c['id']: c for c in candidates if c['id'] in ids}
                event.update({
                    'status': 'ok',
                    'elapsed_ms': retrieval_ms,
                    'results': results,
                    'context': context,
                    'context_ids': ids,
                })
                if retrieval_trace is not None:
                    event['retrieval_trace'] = {
                        key: value for key, value in retrieval_trace.items()
                        if key != 'results'
                    }
                # Send only the latest cumulative context; avoid multiplying the budget.
                history = [{'tool': 'search', 'queries': [e['query'] for e in events if e['event'] == 'search'],
                            'context': context, 'context_ids': ids}]
            elif action.get('action') == 'answer':
                answer, citations = action.get('answer'), action.get('citations')
                if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                    raise ValueError('answer/citations không hợp lệ')
                valid = bool(answer.strip() and citations and all(i in chunks and f'[{i}]' in answer for i in citations))
                if valid:
                    events.append({
                        'event': 'answer',
                        'model_answer': answer,
                        'citations': citations,
                        'controller_model': {
                            key: action[key]
                            for key in ('provider', 'model', 'usage', 'cost_usd')
                            if key in action
                        },
                    })
                    return finish('answered', answer, citations)
                events.append({
                    'event': 'invalid_or_missing_citations',
                    'model_answer': answer,
                    'attempted_citations': citations,
                    'controller_model': {
                        key: action[key]
                        for key in ('provider', 'model', 'usage', 'cost_usd')
                        if key in action
                    },
                })
                return finish(
                    'abstain', 'Không đủ bằng chứng để trả lời.', []
                )
            else:
                raise ValueError('action phải là search hoặc answer')
        return finish('abstain', 'Không đủ bằng chứng để trả lời.', [])
    except Exception as exc:
        error = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": sanitize_error_message(str(exc)),
        }
        if stage in ("retriever", "context") and events and events[-1].get("event") == "search":
            events[-1]["status"] = "error"
            events[-1]["error"] = error
        try:
            finish("error", "", [], error=error)
        except Exception as trace_exc:
            raise RuntimeError(
                f"Agent failed and its trace could not be persisted: {trace_exc}"
            ) from exc
        raise
