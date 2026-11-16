import argparse
import json
import os
import statistics
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from psycopg.rows import dict_row

from agentic import Controller, Retriever, run_agent
from baseline import TfidfIndex
from convert_day6 import convert_chunk
from dense_retrieval import (
    OPENROUTER_BASE_URL,
    STRATEGY,
    get_api_key,
    get_connection_url,
    get_embedding_model,
    verify_index,
)
from lexical_retrieval import BM25Index


HERE = Path(__file__).resolve().parent
DEFAULT_AGENT_MODEL = "openai/gpt-4o-mini"


class TfidfAgentRetriever:
    """TF-IDF baseline adapter; intentionally does not apply metadata filters."""

    def __init__(self, chunks):
        self.index = TfidfIndex(chunks)
        self.embedding_calls = 0
        self.embedding_tokens = 0
        self.database_elapsed_ms = 0.0
        self.fusion_elapsed_ms = 0.0

    def search(self, query, *, filters, top_k):
        return self.index.search(query, top_k=top_k)


def load_tasks(questions_path, missing_cases_path, split):
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    missing_cases = json.loads(missing_cases_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not isinstance(missing_cases, list):
        raise ValueError("questions và missing cases phải là JSON arrays")

    tasks = []
    cases = []
    for question in questions:
        if question.get("split") != split:
            continue
        task = {
            "id": question.get("id"),
            "question": question.get("question"),
            "filters": question.get("filter"),
            "answerable": True,
        }
        tasks.append(task)
        cases.append({
            "id": task["id"],
            "split": split,
            "answerable": True,
        })
    for missing in missing_cases:
        if missing.get("split") != split:
            continue
        task = {
            "id": missing.get("id"),
            "question": missing.get("question"),
            "filters": missing.get("filter"),
            "answerable": False,
        }
        tasks.append(task)
        cases.append({
            "id": task["id"],
            "split": split,
            "answerable": False,
        })
    ids = [task["id"] for task in tasks]
    if not tasks or any(not isinstance(task["question"], str) or not task["question"].strip() for task in tasks):
        raise ValueError(f"split {split} rỗng hoặc có question không hợp lệ")
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"split {split} có ID thiếu hoặc trùng")
    return tasks, cases


def load_chunks(path):
    raw_chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("chunks phải là JSON array không rỗng")
    chunks = [convert_chunk(chunk) for chunk in raw_chunks]
    return chunks


def main():
    load_dotenv(HERE / ".env")
    parser = argparse.ArgumentParser(
        description="Run the real OpenRouter agent with hybrid retrieval"
    )
    parser.add_argument("--chunks", type=Path, default=HERE / "chunks_B.json")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.json")
    parser.add_argument(
        "--missing-cases", type=Path, default=HERE / "agent_cases_missing.json"
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--retrieval",
        choices=("hybrid", "tfidf"),
        default="hybrid",
        help="Retrieval pipeline used by the agent",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if min(args.top_k, args.candidate_k, args.budget, args.max_searches) < 1:
        raise ValueError("top-k, candidate-k, budget và max-searches phải lớn hơn 0")
    if args.output is None:
        suffix = "hybrid-minmax" if args.retrieval == "hybrid" else "tfidf"
        args.output = HERE / "results" / f"agent-{args.split}-{suffix}"

    tasks, cases = load_tasks(args.questions, args.missing_cases, args.split)
    chunks = load_chunks(args.chunks)
    api_key = get_api_key()
    embedding_model = get_embedding_model() if args.retrieval == "hybrid" else None
    agent_model = os.environ.get("AGENT_MODEL", DEFAULT_AGENT_MODEL).strip()
    if not agent_model:
        raise ValueError("AGENT_MODEL không được để trống")

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    conn = None
    try:
        db_info = None
        if args.retrieval == "hybrid":
            conn = psycopg.connect(get_connection_url(), row_factory=dict_row)
            db_info, indexed_chunks = verify_index(conn, chunks)
            bm25_index = BM25Index(chunks, k1=0.75, b=0.75)
            retriever = Retriever(
                conn,
                client,
                embedding_model,
                bm25_index,
                candidate_k=args.candidate_k,
            )
        else:
            indexed_chunks = len(chunks)
            retriever = TfidfAgentRetriever(chunks)
        controller = Controller(
            client,
            agent_model,
            max_searches=args.max_searches,
        )
        traces = []
        elapsed_times = []
        for task in tasks:
            started = time.perf_counter()
            trace = run_agent(
                task["question"],
                controller,
                retriever,
                filters=task["filters"] if args.retrieval == "hybrid" else None,
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            elapsed_times.append(elapsed_ms)
            traces.append({
                "query_id": task["id"],
                "split": args.split,
                "retrieval": args.retrieval,
                "question": task["question"],
                "status": trace["status"],
                "answer": trace["answer"],
                "citations": trace["citations"],
                "events": trace["events"],
                "elapsed_ms": elapsed_ms,
            })

        summary = {
            "strategy": (
                "hybrid_minmax_dense_bm25_agent"
                if args.retrieval == "hybrid"
                else "tfidf_agent"
            ),
            "split": args.split,
            "retrieval": args.retrieval,
            "metadata_filters": args.retrieval == "hybrid",
            "n": len(traces),
            "answerable_cases": sum(task["answerable"] for task in tasks),
            "unanswerable_cases": sum(not task["answerable"] for task in tasks),
            "answered": sum(trace["status"] == "answered" for trace in traces),
            "abstained": sum(trace["status"] == "abstain" for trace in traces),
            "error": sum(trace["status"] == "error" for trace in traces),
            "model_abstentions": sum(
                event.get("event") == "model_abstention"
                for trace in traces
                for event in trace["events"]
            ),
            "answer_validation_failures": sum(
                event.get("event") == "answer_validation_failed"
                for trace in traces
                for event in trace["events"]
            ),
            "invalid_model_abstentions": sum(
                event.get("event") == "invalid_model_abstention"
                for trace in traces
                for event in trace["events"]
            ),
            "search_limit_hits": sum(
                event.get("event") == "search_limit"
                for trace in traces
                for event in trace["events"]
            ),
            "parallel_tool_calls_ignored": sum(
                len(event.get("calls", []))
                for trace in traces
                for event in trace["events"]
                if event.get("event") == "parallel_tool_calls_ignored"
            ),
            "agent_model": agent_model,
            "embedding_model": embedding_model,
            "database": db_info["database"] if db_info else None,
            "database_strategy": STRATEGY if db_info else None,
            "indexed_chunks": indexed_chunks,
            "normalization": "per-query min-max" if db_info else None,
            "fusion_weights": {"dense": 0.5, "bm25": 0.5} if db_info else None,
            "top_k": args.top_k,
            "candidate_k_per_method": args.candidate_k if db_info else None,
            "context_budget_chars": args.budget,
            "max_searches": args.max_searches,
            "chat_calls": controller.calls,
            "chat_prompt_tokens": controller.prompt_tokens,
            "chat_completion_tokens": controller.completion_tokens,
            "embedding_calls": retriever.embedding_calls,
            "embedding_tokens": retriever.embedding_tokens,
            "database_elapsed_ms": getattr(retriever, "database_elapsed_ms", 0.0),
            "fusion_elapsed_ms": getattr(retriever, "fusion_elapsed_ms", 0.0),
            "average_elapsed_ms_per_case": statistics.mean(elapsed_times),
            "service_cost_usd": None,
        }
    finally:
        if conn is not None:
            conn.close()
        client.close()

    args.output.mkdir(parents=True, exist_ok=True)
    for name, result in (("cases.json", cases), ("traces.json", traces), ("summary.json", summary)):
        (args.output / name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
