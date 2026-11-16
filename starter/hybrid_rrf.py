import argparse
import json
import statistics
import time
from pathlib import Path

from baseline import pack_context


HERE = Path(__file__).resolve().parent


def load_traces(path, split):
    traces = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(traces, list):
        raise ValueError(f"{path}: traces phải là JSON array")
    selected = [trace for trace in traces if trace.get("split") == split]
    ids = [trace.get("query_id") for trace in selected]
    if not selected or len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError(f"{path}: split {split} rỗng hoặc query_id bị thiếu/trùng")
    return {trace["query_id"]: trace for trace in selected}


def matches_filter(result, filters):
    metadata = result.get("metadata")
    return isinstance(metadata, dict) and all(
        metadata.get(key) == value for key, value in filters.items()
    )


def fuse_query(dense, lexical, *, top_k, budget, rrf_k):
    if dense["question"] != lexical["question"] or dense["gold"] != lexical["gold"]:
        raise ValueError(f"{dense['query_id']}: dense và TF-IDF không cùng query/gold")

    filters = dense.get("filter") or {}
    rank_scores = {}
    candidates = {}
    source_ranks = {}
    for method, trace in (("dense", dense), ("tfidf", lexical)):
        filtered = [
            result for result in trace["results"]
            if matches_filter(result, filters)
        ]
        for rank, result in enumerate(filtered, start=1):
            chunk_id = result["id"]
            if chunk_id not in candidates:
                candidates[chunk_id] = dict(result)
                rank_scores[chunk_id] = 0.0
                source_ranks[chunk_id] = {}
            rank_scores[chunk_id] += 1.0 / (rrf_k + rank)
            source_ranks[chunk_id][method] = rank

    fused = []
    for chunk_id, result in candidates.items():
        fused.append({
            **result,
            "dense_rank": source_ranks[chunk_id].get("dense"),
            "tfidf_rank": source_ranks[chunk_id].get("tfidf"),
            "rrf_score": rank_scores[chunk_id],
            "score": rank_scores[chunk_id],
        })
    fused.sort(key=lambda result: (-result["rrf_score"], result["id"]))
    fused = fused[:top_k]
    context, context_ids = pack_context(fused, budget=budget)
    gold = dense["gold"]
    for result in fused:
        result["hit"] = any(
            result["source_id"] == item["source_id"]
            and result["page"] == item["page"]
            and " ".join(item["evidence"].split()) in " ".join(result["text"].split())
            for item in gold
        )

    return {
        "query_id": dense["query_id"],
        "split": dense["split"],
        "question": dense["question"],
        "strategy": "hybrid_rrf",
        "database_strategy": dense.get("database_strategy"),
        "rrf_k": rrf_k,
        "top_k": top_k,
        "budget_chars": budget,
        "gold": gold,
        "filter": filters or None,
        "results": fused,
        "context": context,
        "context_ids": context_ids,
        "context_chars": len(context),
        "hit1": bool(fused and fused[0]["hit"]),
        "hit3": any(result["hit"] for result in fused),
        "context_hit": any(
            result["hit"] and result["id"] in context_ids for result in fused
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Fuse saved dense and TF-IDF results with RRF")
    parser.add_argument(
        "--dense-traces", type=Path,
        default=HERE / "results" / "dev-dense-B" / "traces.json",
    )
    parser.add_argument(
        "--tfidf-traces", type=Path,
        default=HERE / "results" / "dev-tfidf-recheck" / "traces.json",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "results" / "dev-hybrid-rrf-B",
    )
    args = parser.parse_args()
    if args.top_k < 1 or args.budget < 1 or args.rrf_k < 1:
        raise ValueError("top-k, budget và rrf-k phải lớn hơn 0")

    dense = load_traces(args.dense_traces, args.split)
    lexical = load_traces(args.tfidf_traces, args.split)
    if dense.keys() != lexical.keys():
        raise ValueError("Dense và TF-IDF phải có cùng query ID trong split")

    started = time.perf_counter()
    traces = [
        fuse_query(
            dense[query_id],
            lexical[query_id],
            top_k=args.top_k,
            budget=args.budget,
            rrf_k=args.rrf_k,
        )
        for query_id in dense
    ]
    elapsed_ms = (time.perf_counter() - started) * 1000
    count = len(traces)
    summary = {
        "strategy": "hybrid_rrf",
        "split": args.split,
        "n": count,
        "candidate_depth_per_method": min(
            min(len(trace["results"]) for trace in dense.values()),
            min(len(trace["results"]) for trace in lexical.values()),
        ),
        "top_k": args.top_k,
        "rrf_k": args.rrf_k,
        "budget_chars": args.budget,
        "hit1": sum(trace["hit1"] for trace in traces),
        "hit3": sum(trace["hit3"] for trace in traces),
        "context_hit": sum(trace["context_hit"] for trace in traces),
        "avg_context_chars": statistics.mean(trace["context_chars"] for trace in traces),
        "fusion_elapsed_ms": elapsed_ms,
        "service_cost_usd": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
