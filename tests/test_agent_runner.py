import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from agentic import Controller, Retriever, run_agent
from agent_runner import TfidfAgentRetriever, load_tasks


def tool_call(name, arguments):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


class FakeChatCompletions:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.response


class FakeBM25:
    def __init__(self, results):
        self.results = results
        self.filters = None

    def search(self, query, *, top_k, filters):
        self.filters = filters
        return self.results[:top_k]


class AgentRunnerTests(unittest.TestCase):
    def test_tfidf_agent_retriever_uses_baseline_without_metadata_filter(self):
        chunks = [
            {
                'id': 'match',
                'source_id': 'doc',
                'source': 'doc.pdf',
                'page': 1,
                'text': 'patent protection filing',
                'metadata': {'section': 'Part I'},
            },
            {
                'id': 'other',
                'source_id': 'doc',
                'source': 'doc.pdf',
                'page': 2,
                'text': 'unrelated material',
                'metadata': {'section': 'Part II'},
            },
        ]
        retriever = TfidfAgentRetriever(chunks)

        results = retriever.search(
            'patent protection',
            filters={'section': 'Part II'},
            top_k=1,
        )

        self.assertEqual([result['id'] for result in results], ['match'])
        self.assertEqual(results[0]['strategy'], 'tfidf')

    def test_controller_parses_search_and_answer_tool_calls(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(tool_calls=[tool_call(
                    'search_evidence', {'query': 'focused query'}
                )])
            )],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        )
        completions = FakeChatCompletions(response)
        controller = Controller(
            SimpleNamespace(chat=SimpleNamespace(
                completions=completions
            )),
            'openai/gpt-4o-mini',
        )

        action = controller.decide(
            'question',
            [{'queries': ['old query'], 'context': 'evidence', 'context_ids': ['c1']}],
        )

        self.assertEqual(action, {'action': 'search', 'query': 'focused query'})
        self.assertIn('Available citation IDs: c1', completions.request['messages'][1]['content'])
        self.assertIn('Searches used: 1 of 2', completions.request['messages'][1]['content'])
        self.assertFalse(completions.request['parallel_tool_calls'])
        self.assertEqual(controller.calls, 1)
        self.assertEqual(controller.prompt_tokens, 12)

        completions.response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(tool_calls=[tool_call(
                    'final_answer', {
                        'answer': 'Supported answer [c1]',
                        'abstain': False,
                        'citations': ['c1'],
                    }
                )])
            )],
            usage=None,
        )
        self.assertEqual(
            controller.decide('question', []),
            {
                'action': 'answer',
                'answer': 'Supported answer [c1]',
                'abstain': False,
                'citations': ['c1'],
            },
        )

    def test_controller_forces_final_answer_after_search_limit(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(tool_calls=[tool_call(
                    'final_answer', {
                        'answer': 'Not enough evidence',
                        'abstain': True,
                        'citations': [],
                    }
                )])
            )],
            usage=None,
        )
        completions = FakeChatCompletions(response)
        controller = Controller(
            SimpleNamespace(chat=SimpleNamespace(
                completions=completions
            )),
            'openai/gpt-4o-mini',
            max_searches=1,
        )

        action = controller.decide(
            'question',
            [{'queries': ['first search'], 'context': 'evidence', 'context_ids': ['c1']}],
        )

        self.assertEqual(action['action'], 'answer')
        self.assertTrue(action['abstain'])
        self.assertEqual(completions.request['tools'][0]['function']['name'], 'final_answer')
        self.assertEqual(
            completions.request['tool_choice'],
            {'type': 'function', 'function': {'name': 'final_answer'}},
        )

    def test_controller_prioritizes_search_when_model_returns_parallel_calls(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(tool_calls=[
                    tool_call('final_answer', {
                        'answer': 'Premature answer',
                        'citations': [],
                    }),
                    tool_call('search_evidence', {'query': 'first search'}),
                    tool_call('search_evidence', {'query': 'extra search'}),
                ])
            )],
            usage=None,
        )
        controller = Controller(
            SimpleNamespace(chat=SimpleNamespace(
                completions=FakeChatCompletions(response)
            )),
            'openai/gpt-4o-mini',
        )

        action = controller.decide('question', [])

        self.assertEqual(action['action'], 'search')
        self.assertEqual(action['query'], 'first search')
        self.assertEqual(len(action['parallel_tool_calls_ignored']), 2)

    def test_agent_records_ignored_parallel_tool_calls(self):
        chunk = {
            'id': 'c1',
            'source_id': 'doc',
            'source': 'doc.pdf',
            'page': 1,
            'text': 'evidence',
            'metadata': {},
            'score': 1.0,
        }

        class FakeController:
            def __init__(self):
                self.actions = iter([
                    {
                        'action': 'search',
                        'query': 'q',
                        'parallel_tool_calls_ignored': [{'name': 'search_evidence'}],
                    },
                    {
                        'action': 'answer',
                        'answer': 'evidence [c1]',
                        'citations': ['c1'],
                    },
                ])

            def decide(self, question, history):
                return next(self.actions)

        class FakeRetriever:
            def search(self, query, *, filters, top_k):
                return [chunk]

        result = run_agent(
            'question',
            FakeController(),
            FakeRetriever(),
            max_searches=1,
        )

        self.assertEqual(result['status'], 'answered')
        self.assertEqual(
            result['events'][0]['event'],
            'parallel_tool_calls_ignored',
        )

    def test_task_loader_does_not_pass_gold_or_absence_reason_to_agent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            questions_path = Path(temp_dir) / 'questions.json'
            missing_path = Path(temp_dir) / 'missing.json'
            questions_path.write_text(json.dumps([{
                'id': 'q1',
                'split': 'dev',
                'question': 'answerable question',
                'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'secret gold'}],
                'expected_answer': 'secret answer',
                'filter': None,
            }]), encoding='utf-8')
            missing_path.write_text(json.dumps([{
                'id': 'q2',
                'split': 'dev',
                'question': 'missing question',
                'answerable': False,
                'absence_reason': 'secret reason',
            }]), encoding='utf-8')

            tasks, cases = load_tasks(questions_path, missing_path, 'dev')

        self.assertEqual([task['id'] for task in tasks], ['q1', 'q2'])
        self.assertNotIn('gold', tasks[0])
        self.assertNotIn('expected_answer', tasks[0])
        self.assertNotIn('absence_reason', tasks[1])
        self.assertEqual([case['answerable'] for case in cases], [True, False])

    def test_hybrid_retriever_applies_filter_and_returns_fused_chunks(self):
        metadata = {'section': 'Part II'}
        dense_rows = [
            {
                'chunk_id': 'dense',
                'document_id': 'doc',
                'source': 'doc.pdf',
                'page': 1,
                'text': 'dense text',
                'metadata': metadata,
                'score': 0.9,
            },
            {
                'chunk_id': 'shared',
                'document_id': 'doc',
                'source': 'doc.pdf',
                'page': 2,
                'text': 'shared text',
                'metadata': metadata,
                'score': 0.8,
            },
        ]

        class FakeConnection:
            def execute(self, query, parameters):
                self.parameters = parameters
                return SimpleNamespace(fetchall=lambda: dense_rows)

        class FakeEmbeddings:
            def create(self, *, model, input):
                return SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.1] * 3072)],
                    usage=SimpleNamespace(prompt_tokens=5),
                )

        bm25_results = [
            {
                'id': 'lexical',
                'source_id': 'doc',
                'source': 'doc.pdf',
                'page': 3,
                'text': 'lexical text',
                'metadata': metadata,
                'score': 9.0,
            },
            {
                'id': 'shared',
                'source_id': 'doc',
                'source': 'doc.pdf',
                'page': 2,
                'text': 'shared text',
                'metadata': metadata,
                'score': 7.0,
            },
        ]
        client = SimpleNamespace(embeddings=FakeEmbeddings())
        conn = FakeConnection()
        bm25 = FakeBM25(bm25_results)
        retriever = Retriever(
            conn, client, 'embedding-model', bm25, candidate_k=2
        )

        results = retriever.search(
            'query', filters={'section': 'Part II'}, top_k=2
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(len({result['id'] for result in results}), 2)
        self.assertTrue(all(
            result['strategy'] == 'hybrid_minmax_dense_bm25'
            for result in results
        ))
        self.assertEqual(bm25.filters, {'section': 'Part II'})
        self.assertEqual(conn.parameters[2], '{"section": "Part II"}')
        self.assertEqual(retriever.embedding_calls, 1)
        self.assertEqual(retriever.embedding_tokens, 5)


if __name__ == '__main__':
    unittest.main()
