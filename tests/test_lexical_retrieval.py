import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from lexical_retrieval import BM25Index, evaluate


def chunk(chunk_id, text):
    return {
        'id': chunk_id,
        'source_id': 'doc',
        'source': 'doc.pdf',
        'page': 1,
        'text': text,
        'metadata': {},
    }


class BM25IndexTests(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index([
            chunk('common-only', 'common common'),
            chunk('rare-and-common', 'rare common'),
            chunk('unrelated', 'other terms'),
        ])

    def test_rare_matching_term_ranks_first(self):
        results = self.index.search('rare common', top_k=2)

        self.assertEqual(
            [result['id'] for result in results],
            ['rare-and-common', 'common-only'],
        )
        self.assertGreater(results[0]['score'], results[1]['score'])

    def test_empty_or_unmatched_query_returns_no_results(self):
        self.assertEqual(self.index.search('!!!'), [])
        self.assertEqual(self.index.search('not-in-corpus'), [])

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BM25Index([chunk('a', 'text')], k1=0)
        with self.assertRaises(ValueError):
            BM25Index([chunk('a', 'text')], b=1.1)
        with self.assertRaises(ValueError):
            self.index.search('common', top_k=0)

    def test_hit3_stays_hit_at_three_when_retrieving_more(self):
        rows = [
            chunk('a', 'wrong one'),
            chunk('b', 'wrong two'),
            chunk('c', 'wrong three'),
            chunk('d', 'evidence'),
        ]

        class FixedIndex:
            def search(self, question, top_k):
                return rows[:top_k]

        question = {
            'id': 'q1',
            'split': 'dev',
            'question': 'question',
            'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'evidence'}],
        }
        trace = evaluate(FixedIndex(), [question], top_k=4, budget=1000)[0]

        self.assertFalse(trace['hit3'])
        self.assertTrue(trace['context_hit'])


if __name__ == '__main__':
    unittest.main()
