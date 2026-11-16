import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from agent_benchmark import _metrics


class AgentBenchmarkTests(unittest.TestCase):
    def test_context_metric_uses_evidence_from_any_search(self):
        case = {
            "answerable": True,
            "gold": [{"source_id": "doc", "page": 2, "evidence": "needed evidence"}],
        }
        unrelated = {
            "id": "c1",
            "source_id": "doc",
            "page": 1,
            "text": "other text",
        }
        evidence = {
            "id": "c2",
            "source_id": "doc",
            "page": 2,
            "text": "needed evidence here",
        }
        trace = {
            "events": [
                {"event": "search", "status": "ok", "results": [unrelated]},
                {"event": "search", "status": "ok", "results": [evidence]},
            ],
            "context_ids": ["c1", "c2"],
        }
        self.assertEqual(
            _metrics(case, trace),
            {"hit1": False, "evidence_hit_at_3": False, "context_hit": True},
        )

    def test_missing_retrieval_and_unanswerable_cases_are_not_in_denominator(self):
        case = {
            "answerable": True,
            "gold": [{"source_id": "doc", "page": 1, "evidence": "text"}],
        }
        self.assertEqual(
            _metrics(case, {"events": [], "context_ids": []}),
            {"hit1": None, "evidence_hit_at_3": None, "context_hit": None},
        )
        self.assertEqual(
            _metrics({"answerable": False, "gold": []}, {"events": []}),
            {"hit1": None, "evidence_hit_at_3": None, "context_hit": None},
        )


if __name__ == "__main__":
    unittest.main()
