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

    def test_extracts_only_explicit_inline_citations_when_array_is_omitted(self):
        self.assertEqual(
            parse_action(
                '{"action":"answer","answer":"Fact [chunk-1] and [chunk-2]"}'
            )["citations"],
            ["chunk-1", "chunk-2"],
        )
        self.assertEqual(
            parse_action(
                '{"action":"answer","answer":"INSUFFICIENT_EVIDENCE"}'
            )["citations"],
            [],
        )
        self.assertEqual(
            parse_action(
                '{"action":"answer","answer":"Unsupported claim"}'
            )["citations"],
            [],
        )

    def test_parses_json_fenced_in_markdown(self):
        self.assertEqual(
            parse_action(
                '```json\n{"action":"answer","answer":"Fact [c1]",'
                '"citations":["c1"]}\n```'
            ),
            {
                "action": "answer",
                "answer": "Fact [c1]",
                "citations": ["c1"],
            },
        )

    def test_rejects_malformed_or_unsafe_actions(self):
        for raw in (
            "not JSON",
            "[]",
            '{"action":"search","query":" "}',
            '{"action":"answer","answer":"Fact","citations":[1]}',
            '{"action":"answer","answer":"Fact","citations":null}',
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
