"""Offline behavioral checks only; fake adapters do not measure model quality."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from agentic import run_agent

CHUNK = {'id': 'c1', 'text': 'Hoàn tiền trong 7 ngày.', 'source_id': 'doc',
         'source': 'doc.pdf', 'page': 1, 'score': 0.8, 'metadata': {}}

class Controller:
    def __init__(self, actions): self.actions = iter(actions)
    def decide(self, question, history): return next(self.actions)

class Retriever:
    def __init__(self, results): self.results = results; self.calls = []
    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k)); return self.results

class AgentTests(unittest.TestCase):
    def test_agent_searches_and_cites_actual_context(self):
        c = Controller([
            {'action': 'search', 'query': 'hoàn tiền', 'provider': 'local',
             'model': 'test-model', 'usage': {'input_tokens': 10}, 'cost_usd': 0.001},
            {'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1'],
             'provider': 'local', 'model': 'test-model', 'cost_usd': 0.002},
        ])
        r = Retriever([CHUNK]); result = run_agent('Hoàn tiền?', c, r, filters={'year': 2025})
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['citations'], ['c1'])
        self.assertEqual(r.calls, [('hoàn tiền', {'year': 2025}, 3)])
        self.assertEqual(result['events'][0]['context_ids'], ['c1'])

    def test_invalid_citation_and_empty_context_abstain(self):
        for rows, cid in (([CHUNK], 'not-present'), ([], 'c1')):
            c = Controller([{'action':'search','query':'q'}, {'action':'answer','answer':f'7 ngày [{cid}]','citations':[cid]}])
            self.assertEqual(run_agent('q', c, Retriever(rows))['status'], 'abstain')

    def test_search_limit_is_enforced(self):
        c = Controller([{'action':'search','query':'one'}, {'action':'search','query':'two'}, {'action':'search','query':'three'}])
        r = Retriever([CHUNK]); result = run_agent('q', c, r, max_searches=2)
        self.assertEqual(result['status'], 'abstain'); self.assertEqual(len(r.calls), 2)

    def test_context_budget_across_multiple_searches(self):
        other = {**CHUNK, 'id': 'c2', 'text': 'Một đoạn bằng chứng khác.'}
        class SequentialRetriever:
            def __init__(self): self.i = 0
            def search(self, query, *, filters, top_k):
                self.i += 1
                return [CHUNK] if self.i == 1 else [other]
        class InspectController:
            def __init__(self): self.i = 0
            def decide(self, question, history):
                self.i += 1
                if history:
                    assert len(history) == 1 and len(history[0]['context']) <= 90
                if self.i <= 2: return {'action': 'search', 'query': str(self.i)}
                return {'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']}
        result = run_agent('q', InspectController(), SequentialRetriever(), budget=90)
        self.assertEqual(result['status'], 'answered')
        self.assertTrue(all(len(e['context']) <= 90 for e in result['events'] if e['event'] == 'search'))

    def test_configurable_retrieval_allows_more_than_two_calls(self):
        c = Controller([{'action': 'search', 'query': str(i)} for i in range(3)] +
                       [{'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']}])
        r = Retriever([CHUNK])
        result = run_agent('q', c, r, max_searches=3, top_k=7, budget=1200)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(r.calls), 3)
        self.assertTrue(all(call[2] == 7 for call in r.calls))

    def test_tool_errors_are_not_hidden(self):
        class Broken:
            def search(self, query, *, filters, top_k): raise ConnectionError('database unavailable')
        with self.assertRaises(ConnectionError):
            run_agent('q', Controller([{'action':'search','query':'q'}]), Broken())

    def test_trace_persists_search_context_answer_and_citations(self):
        c = Controller([
            {'action': 'search', 'query': 'hoàn tiền', 'provider': 'local',
             'model': 'test-model', 'usage': {'input_tokens': 10}, 'cost_usd': 0.001},
            {'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1'],
             'provider': 'local', 'model': 'test-model', 'cost_usd': 0.002},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.jsonl'
            result = run_agent('Hoàn tiền?', c, Retriever([CHUNK]), trace_path=path)
            record = json.loads(path.read_text(encoding='utf-8').splitlines()[0])
        self.assertEqual(record['trace_id'], result['trace_id'])
        self.assertEqual(record['status'], 'answered')
        self.assertEqual(record['tool_call_count'], 1)
        self.assertEqual(record['events'][0]['tool_call']['top_k'], 3)
        self.assertEqual(record['context_ids'], ['c1'])
        self.assertEqual(record['answer'], '7 ngày [c1]')
        self.assertEqual(record['citations'], ['c1'])
        self.assertAlmostEqual(record['reported_cost_usd'], 0.003)

    def test_trace_persists_retriever_errors_before_propagating(self):
        class Broken:
            def search(self, query, *, filters, top_k):
                raise ConnectionError('database unavailable')

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.jsonl'
            with self.assertRaises(ConnectionError):
                run_agent(
                    'q', Controller([{'action': 'search', 'query': 'q'}]),
                    Broken(), trace_path=path,
                )
            record = json.loads(path.read_text(encoding='utf-8').splitlines()[0])
        self.assertEqual(record['status'], 'error')
        self.assertEqual(record['error']['stage'], 'retriever')
        self.assertIn('database unavailable', record['error']['message'])
        self.assertEqual(record['events'][0]['status'], 'error')
        self.assertEqual(record['events'][0]['tool_call']['query'], 'q')

    def test_trace_keeps_last_context_when_later_search_fails(self):
        class FailsOnSecondSearch:
            def __init__(self): self.calls = 0
            def search(self, query, *, filters, top_k):
                self.calls += 1
                if self.calls == 2:
                    raise ConnectionError('second search failed')
                return [CHUNK]

        controller = Controller([
            {'action': 'search', 'query': 'first'},
            {'action': 'search', 'query': 'second'},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.jsonl'
            with self.assertRaises(ConnectionError):
                run_agent('q', controller, FailsOnSecondSearch(), trace_path=path)
            record = json.loads(path.read_text(encoding='utf-8').splitlines()[0])
        self.assertIn('[c1 | doc.pdf | page 1]', record['context'])
        self.assertEqual(record['events'][1]['status'], 'error')
        self.assertEqual(record['tool_call_count'], 2)

if __name__ == '__main__': unittest.main()
