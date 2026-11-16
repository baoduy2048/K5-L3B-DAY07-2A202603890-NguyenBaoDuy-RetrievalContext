import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from hybrid_score_fusion import fuse_query, min_max_scores


def result(chunk_id, score, *, hit=False, section='Part II', text='other'):
    return {
        'id': chunk_id,
        'source_id': 'doc',
        'source': 'doc.pdf',
        'page': 1 if hit else 2,
        'text': 'evidence' if hit else text,
        'metadata': {'section': section},
        'score': score,
        'hit': hit,
    }


class HybridScoreFusionTests(unittest.TestCase):
    def test_min_max_normalizes_scores_and_equal_scores(self):
        normalized = min_max_scores([
            result('high', 10),
            result('low', 2),
        ])
        self.assertEqual(normalized, {'high': 1.0, 'low': 0.0})
        self.assertEqual(
            min_max_scores([result('a', 5), result('b', 5)]),
            {'a': 1.0, 'b': 1.0},
        )

    def test_fuses_differently_scaled_scores_and_respects_filter(self):
        gold = [{'source_id': 'doc', 'page': 1, 'evidence': 'evidence'}]
        dense = {
            'query_id': 'q1',
            'split': 'dev',
            'question': 'question',
            'gold': gold,
            'filter': {'section': 'Part II'},
            'database_strategy': 'B',
            'results': [
                result('dense-only', 0.9),
                result('shared-hit', 0.86, hit=True),
                result('shared-low', 0.4),
            ],
        }
        bm25 = {
            'query_id': 'q1',
            'split': 'dev',
            'question': 'question',
            'gold': gold,
            'results': [
                result('shared-low', 100),
                result('shared-hit', 91, hit=True),
                result('filtered-out', 200, section='Part I'),
                result('bm25-low', 0),
            ],
        }

        trace = fuse_query(dense, bm25, top_k=3, budget=1000)

        self.assertEqual(trace['results'][0]['id'], 'shared-hit')
        self.assertAlmostEqual(trace['results'][0]['fusion_score'], 0.915)
        self.assertTrue(trace['hit1'])
        self.assertTrue(trace['context_hit'])
        self.assertNotIn(
            'filtered-out', [item['id'] for item in trace['results']]
        )


if __name__ == '__main__':
    unittest.main()
