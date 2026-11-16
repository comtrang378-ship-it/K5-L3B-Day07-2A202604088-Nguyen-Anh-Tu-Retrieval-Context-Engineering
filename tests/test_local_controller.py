import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from local_controller import parse_action


class LocalControllerTests(unittest.TestCase):
    def test_parses_search_and_answer_actions(self):
        self.assertEqual(
            parse_action('{"action":"search","query":"  evidence  "}'),
            {"action": "search", "query": "evidence"},
        )
        self.assertEqual(
            parse_action(
                '{"action":"answer","answer":"Fact [c1]","citations":["c1"]}'
            ),
            {
                "action": "answer",
                "answer": "Fact [c1]",
                "citations": ["c1"],
            },
        )
        self.assertEqual(
            parse_action(
                '{"action":"answer","answer":"INSUFFICIENT_EVIDENCE","citations":[]}'
            )["citations"],
            [],
        )

    def test_rejects_malformed_or_unsafe_actions(self):
        for raw in (
            "not JSON",
            "[]",
            '{"action":"search","query":" "}',
            '{"action":"answer","answer":"Fact","citations":[1]}',
            '{"action":"execute","command":"..."}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_action(raw)

    def test_invalid_response_includes_bounded_raw_output_for_tracing(self):
        raw = "not JSON " + ("x" * 600)
        with self.assertRaises(ValueError) as raised:
            parse_action(raw)

        message = str(raised.exception)
        self.assertIn("Raw model response (truncated)", message)
        self.assertIn(repr(raw[:500]), message)
        self.assertEqual(message.count("x"), len(raw[:500]) - len("not JSON "))


if __name__ == "__main__":
    unittest.main()
