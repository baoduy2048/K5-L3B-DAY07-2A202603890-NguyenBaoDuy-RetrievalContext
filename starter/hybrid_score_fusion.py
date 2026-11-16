import argparse
import json
import math
import statistics
import time
from pathlib import Path

from baseline import pack_context
from hybrid_rrf import load_traces, matches_filter


HERE = Path(__file__).resolve().parent
WEIGHT_DENSE = 0.5
WEIGHT_BM25 = 0.5


def min_max_scores(results):
    scores = {}
    for result in results:
        score = result.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError(f"{result.get('id')}: score phải là số hữu hạn")
        scores[result["id"]] = float(score)
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {chunk_id: 1.0 for chunk_id in scores}
    return {
        chunk_id: (score - low) / (high - low)
        for chunk_id, score in scores.items()
    }


def fuse_results(dense_results, bm25_results, *, top_k):
    if top_k < 1:
        raise ValueError("top_k phải lớn hơn 0")
    for results in (dense_results, bm25_results):
        ids = [result["id"] for result in results]
        if len(ids) != len(set(ids)):
            raise ValueError("Mỗi retriever phải trả về ID chunk duy nhất")

    dense_scores = min_max_scores(dense_results)
    bm25_scores = min_max_scores(bm25_results)
    candidates = {}
    for method, source_results, normalized in (
        ("dense", dense_results, dense_scores),
        ("bm25", bm25_results, bm25_scores),
    ):
        for rank, result in enumerate(source_results, start=1):
            chunk_id = result["id"]
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk": dict(result),
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_norm_score": 0.0,
                    "bm25_norm_score": 0.0,
                }
            candidate = candidates[chunk_id]
            if (
                "hit" in candidate["chunk"]
                and "hit" in result
                and candidate["chunk"]["hit"] != result["hit"]
            ):
                raise ValueError(f"{chunk_id}: retriever không khớp nhãn evidence")
            candidate[f"{method}_rank"] = rank
            candidate[f"{method}_norm_score"] = normalized[chunk_id]

    fused = []
    for candidate in candidates.values():
        dense_score = candidate["dense_norm_score"]
        bm25_score = candidate["bm25_norm_score"]
        fusion_score = WEIGHT_DENSE * dense_score + WEIGHT_BM25 * bm25_score
        fused.append({
            **candidate["chunk"],
            "dense_rank": candidate["dense_rank"],
            "bm25_rank": candidate["bm25_rank"],
            "dense_norm_score": dense_score,
            "bm25_norm_score": bm25_score,
            "fusion_score": fusion_score,
            "score": fusion_score,
        })
    fused.sort(key=lambda result: (-result["fusion_score"], result["id"]))
    return fused[:top_k]


def fuse_query(dense, bm25, *, top_k, budget):
    if dense["question"] != bm25["question"] or dense["gold"] != bm25["gold"]:
        raise ValueError(f"{dense['query_id']}: dense và BM25 không cùng query/gold")

    filters = dense.get("filter") or {}
    dense_results = [
        result for result in dense["results"]
        if matches_filter(result, filters)
    ]
    bm25_results = [
        result for result in bm25["results"]
        if matches_filter(result, filters)
    ]
    fused = fuse_results(dense_results, bm25_results, top_k=top_k)
    context, context_ids = pack_context(fused, budget=budget)
    included = set(context_ids)
    return {
        "query_id": dense["query_id"],
        "split": dense["split"],
        "question": dense["question"],
        "strategy": "hybrid_minmax_dense_bm25",
        "database_strategy": dense.get("database_strategy"),
        "normalization": "per-query min-max",
        "weights": {"dense": WEIGHT_DENSE, "bm25": WEIGHT_BM25},
        "top_k": top_k,
        "budget_chars": budget,
        "gold": dense["gold"],
        "filter": filters or None,
        "results": fused,
        "context": context,
        "context_ids": context_ids,
        "context_chars": len(context),
        "hit1": bool(fused and fused[0]["hit"]),
        "hit3": any(result["hit"] for result in fused[:3]),
        "context_hit": any(
            result["hit"] and result["id"] in included for result in fused
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fuse saved dense and BM25 scores with min-max normalization"
    )
    parser.add_argument(
        "--dense-traces", type=Path,
        default=HERE / "results" / "dev-dense-B-budget3000" / "traces.json",
    )
    parser.add_argument(
        "--bm25-traces", type=Path,
        default=HERE / "results" / "dev-bm25-B-k1-0.75-b-0.75-budget3000" / "traces.json",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "results" / "dev-hybrid-score-fusion-B-budget3000",
    )
    args = parser.parse_args()
    if args.top_k < 1 or args.budget < 1:
        raise ValueError("top-k và budget phải lớn hơn 0")

    dense = load_traces(args.dense_traces, args.split)
    bm25 = load_traces(args.bm25_traces, args.split)
    if dense.keys() != bm25.keys():
        raise ValueError("Dense và BM25 phải có cùng query ID trong split")

    started = time.perf_counter()
    traces = [
        fuse_query(
            dense[query_id],
            bm25[query_id],
            top_k=args.top_k,
            budget=args.budget,
        )
        for query_id in dense
    ]
    fusion_ms = (time.perf_counter() - started) * 1000
    summary = {
        "strategy": "hybrid_minmax_dense_bm25",
        "split": args.split,
        "n": len(traces),
        "candidate_depth_per_method": min(
            min(len(trace["results"]) for trace in dense.values()),
            min(len(trace["results"]) for trace in bm25.values()),
        ),
        "normalization": "per-query min-max",
        "weights": {"dense": WEIGHT_DENSE, "bm25": WEIGHT_BM25},
        "top_k": args.top_k,
        "budget_chars": args.budget,
        "hit1": sum(trace["hit1"] for trace in traces),
        "hit3": sum(trace["hit3"] for trace in traces),
        "context_hit": sum(trace["context_hit"] for trace in traces),
        "avg_context_chars": statistics.mean(
            trace["context_chars"] for trace in traces
        ),
        "fusion_elapsed_ms": fusion_ms,
        "service_cost_usd": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
