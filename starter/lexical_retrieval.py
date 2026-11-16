import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from baseline import hit, pack_context, tokenize, validate
from convert_day6 import convert_chunk, convert_question


HERE = Path(__file__).resolve().parent


class BM25Index:
    def __init__(self, chunks, *, k1=1.5, b=0.75):
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 phải là số hữu hạn lớn hơn 0")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b phải nằm trong khoảng [0, 1]")
        if not chunks:
            raise ValueError("chunks phải là danh sách không rỗng")

        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / len(self.lengths) or 1
        document_frequency = Counter(
            term for counts in self.term_counts for term in counts
        )
        document_count = len(chunks)
        self.idf = {
            term: math.log(
                1 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(self, question, top_k=3, *, filters=None):
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("filters phải là object hoặc null")
        query_terms = set(tokenize(question))
        if not query_terms:
            return []

        results = []
        for chunk, counts, length in zip(
            self.chunks, self.term_counts, self.lengths
        ):
            if filters and any(
                chunk["metadata"].get(key) != value
                for key, value in filters.items()
            ):
                continue
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency:
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / self.average_length
                    )
                    score += self.idf[term] * (
                        frequency * (self.k1 + 1) / denominator
                    )
            if score > 0:
                results.append({**chunk, "score": score, "strategy": "bm25"})
        return sorted(results, key=lambda chunk: (-chunk["score"], chunk["id"]))[
            :top_k
        ]


def evaluate(index, questions, *, top_k, budget):
    traces = []
    for question in questions:
        started = time.perf_counter()
        results = index.search(question["question"], top_k=top_k)
        context, context_ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - started) * 1000
        marked = [
            {**chunk, "hit": hit(chunk, question["gold"])}
            for chunk in results
        ]
        traces.append({
            "query_id": question["id"],
            "split": question["split"],
            "question": question["question"],
            "strategy": "bm25",
            "top_k": top_k,
            "budget_chars": budget,
            "gold": question["gold"],
            "results": marked,
            "context": context,
            "context_ids": context_ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(chunk["hit"] for chunk in marked[:3]),
            "context_hit": any(
                chunk["hit"] and chunk["id"] in context_ids for chunk in marked
            ),
        })
    return traces


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BM25 lexical retrieval on Day 6 chunks"
    )
    parser.add_argument("--chunks", type=Path, default=HERE / "chunks_B.json")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "dev-bm25-B"
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()
    if args.top_k < 1 or args.budget < 1:
        raise ValueError("top-k và budget phải lớn hơn 0")

    raw_chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    raw_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(raw_chunks, list) or not isinstance(raw_questions, list):
        raise ValueError("chunks và questions phải là JSON arrays")
    chunks = [convert_chunk(chunk) for chunk in raw_chunks]
    questions = [
        convert_question(question)
        for question in raw_questions
        if question.get("split") == args.split
    ]
    validate(chunks, questions)

    index_started = time.perf_counter()
    index = BM25Index(chunks, k1=args.k1, b=args.b)
    index_seconds = time.perf_counter() - index_started
    traces = evaluate(index, questions, top_k=args.top_k, budget=args.budget)
    summary = {
        "strategy": "bm25",
        "tokenizer": "baseline.tokenize (Unicode word tokens)",
        "split": args.split,
        "n": len(traces),
        "top_k": args.top_k,
        "budget_chars": args.budget,
        "k1": args.k1,
        "b": args.b,
        "hit1": sum(trace["hit1"] for trace in traces),
        "hit3": sum(trace["hit3"] for trace in traces),
        "context_hit": sum(trace["context_hit"] for trace in traces),
        "avg_context_chars": statistics.mean(
            trace["context_chars"] for trace in traces
        ),
        "median_ms": statistics.median(
            trace["elapsed_ms"] for trace in traces
        ),
        "index_seconds": index_seconds,
        "service_cost_usd": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, result in (("traces.json", traces), ("summary.json", summary)):
        (args.output / name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
