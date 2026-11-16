import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from hybrid_rrf import fuse_query


def result(chunk_id, text, page, section='Part II'):
    return {
        'id': chunk_id,
        'source_id': 'doc',
        'source': 'doc.pdf',
        'page': page,
        'text': text,
        'metadata': {'section': section},
        'score': 0.5,
        'hit': text == 'evidence',
    }


class HybridRrfTests(unittest.TestCase):
    def test_fuses_rankings_and_preserves_evidence_hit(self):
        gold = [{'source_id': 'doc', 'page': 1, 'evidence': 'evidence'}]
        dense = {
            'query_id': 'q1', 'split': 'dev', 'question': 'question',
            'gold': gold, 'filter': None, 'database_strategy': 'B',
            'results': [result('shared', 'other', 2), result('gold', 'evidence', 1)],
        }
        lexical = {
            'query_id': 'q1', 'split': 'dev', 'question': 'question',
            'gold': gold,
            'results': [result('shared', 'other', 2), result('noise', 'other', 3)],
        }

        trace = fuse_query(dense, lexical, top_k=2, budget=1000, rrf_k=60)

        self.assertEqual([item['id'] for item in trace['results']], ['shared', 'gold'])
        self.assertFalse(trace['hit1'])
        self.assertTrue(trace['hit3'])
        self.assertTrue(trace['context_hit'])

    def test_applies_question_metadata_filter_to_both_rankings(self):
        gold = [{'source_id': 'doc', 'page': 1, 'evidence': 'evidence'}]
        dense = {
            'query_id': 'q1', 'split': 'dev', 'question': 'question',
            'gold': gold, 'filter': {'section': 'Part II'},
            'results': [result('allowed', 'evidence', 1, 'Part II')],
        }
        lexical = {
            'query_id': 'q1', 'split': 'dev', 'question': 'question',
            'gold': gold,
            'results': [
                result('filtered-out', 'other', 2, 'Part I'),
                result('allowed', 'evidence', 1, 'Part II'),
            ],
        }

        trace = fuse_query(dense, lexical, top_k=3, budget=1000, rrf_k=60)

        self.assertEqual([item['id'] for item in trace['results']], ['allowed'])
        self.assertTrue(trace['context_hit'])


if __name__ == '__main__':
    unittest.main()
