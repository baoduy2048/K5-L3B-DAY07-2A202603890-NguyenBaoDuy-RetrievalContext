import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from psycopg.rows import dict_row

from baseline import hit, pack_context, validate
from convert_day6 import convert_chunk, convert_question


HERE = Path(__file__).resolve().parent
STRATEGY = "B"
EXPECTED_DIMENSIONS = 3072
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SEARCH_SQL = """
SELECT chunk_id, document_id, source, page, text, metadata,
       1 - (embedding <=> %s::vector) AS score
FROM public.ingestion_chunks
WHERE strategy = %s
  AND (%s::jsonb IS NULL OR metadata @> %s::jsonb)
ORDER BY embedding <=> %s::vector
LIMIT %s
"""


def vector_literal(values):
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("embedding phải chứa các số hữu hạn")
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def load_inputs(chunks_path, questions_path, split):
    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(raw_chunks, list) or not isinstance(raw_questions, list):
        raise ValueError("chunks và questions phải là JSON arrays")

    chunks = [convert_chunk(chunk) for chunk in raw_chunks]
    questions = [
        convert_question(question)
        for question in raw_questions
        if question.get("split") == split
    ]
    if not questions:
        raise ValueError(f"không có câu hỏi trong split {split}")
    validate(chunks, questions)
    return chunks, questions


def get_connection_url():
    url = os.environ.get("DAY6_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("Thiếu DAY6_DATABASE_URL (hoặc DATABASE_URL) trong starter/.env")
    return url


def get_api_key():
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Thiếu OPENROUTER_API_KEY trong starter/.env hoặc môi trường")
    return api_key


def get_embedding_model():
    model = os.environ.get(
        "EMBEDDING_MODEL", "openai/text-embedding-3-large"
    ).strip()
    if model == "text-embedding-3-large":
        model = "openai/" + model
    if not model:
        raise ValueError("EMBEDDING_MODEL không được để trống")
    return model


def verify_index(conn, chunks):
    info = conn.execute(
        "SELECT current_database() AS database, current_user AS role"
    ).fetchone()
    db_ids = {
        row["chunk_id"]
        for row in conn.execute(
            "SELECT chunk_id FROM public.ingestion_chunks WHERE strategy = %s",
            (STRATEGY,),
        ).fetchall()
    }
    file_ids = {chunk["id"] for chunk in chunks}
    if db_ids != file_ids:
        raise ValueError(
            "Chunk IDs của strategy B trong DB không khớp chunks_B.json "
            f"(DB={len(db_ids)}, file={len(file_ids)}, "
            f"chỉ DB={len(db_ids - file_ids)}, chỉ file={len(file_ids - db_ids)})"
        )

    dimensions = {
        row["dimensions"]
        for row in conn.execute(
            """
            SELECT DISTINCT vector_dims(embedding) AS dimensions
            FROM public.ingestion_chunks
            WHERE strategy = %s
            """,
            (STRATEGY,),
        ).fetchall()
    }
    if dimensions != {EXPECTED_DIMENSIONS}:
        raise ValueError(
            f"Strategy B cần embedding {EXPECTED_DIMENSIONS} chiều; DB có {sorted(dimensions)}"
        )
    return info, len(db_ids)


def retrieve_and_evaluate(conn, client, model, questions, *, top_k, budget):
    embedding_started = time.perf_counter()
    response = client.embeddings.create(
        model=model,
        input=[question["question"] for question in questions],
    )
    embedding_ms = (time.perf_counter() - embedding_started) * 1000
    embeddings = sorted(response.data, key=lambda item: item.index)
    if len(embeddings) != len(questions):
        raise ValueError("Embedding API trả về số vector không khớp số câu hỏi")
    if any(len(item.embedding) != EXPECTED_DIMENSIONS for item in embeddings):
        raise ValueError(f"Query embedding phải có {EXPECTED_DIMENSIONS} chiều")

    traces = []
    database_times = []
    for question, item in zip(questions, embeddings):
        filters = question.get("filter")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError(f"{question['id']}: filter phải là object hoặc null")
        filter_json = json.dumps(filters, ensure_ascii=False) if filters else None
        vector = vector_literal(item.embedding)

        started = time.perf_counter()
        rows = conn.execute(
            SEARCH_SQL,
            (vector, STRATEGY, filter_json, filter_json, vector, top_k),
        ).fetchall()
        database_ms = (time.perf_counter() - started) * 1000
        database_times.append(database_ms)

        results = []
        for row in rows:
            result = {
                "id": row["chunk_id"],
                "source_id": row["document_id"],
                "source": row["source"],
                "page": row["page"],
                "text": row["text"],
                "metadata": row["metadata"],
                "score": float(row["score"]),
            }
            results.append({**result, "hit": hit(result, question["gold"])})

        context, context_ids = pack_context(results, budget=budget)
        traces.append({
            "query_id": question["id"],
            "split": question["split"],
            "question": question["question"],
            "strategy": "dense_pgvector",
            "database_strategy": STRATEGY,
            "top_k": top_k,
            "budget_chars": budget,
            "gold": question["gold"],
            "filter": filters,
            "results": results,
            "context": context,
            "context_ids": context_ids,
            "context_chars": len(context),
            "database_elapsed_ms": database_ms,
            "hit1": bool(results and results[0]["hit"]),
            "hit3": any(result["hit"] for result in results),
            "context_hit": any(
                result["hit"] and result["id"] in context_ids for result in results
            ),
        })

    return traces, embedding_ms, response.usage.prompt_tokens, database_times


def summarize(traces, *, database, role, model, chunks, embedding_ms,
              embedding_tokens, database_times, top_k, budget):
    count = len(traces)
    total_ms = embedding_ms + sum(database_times)
    return {
        "strategy": "dense_pgvector",
        "database": database,
        "database_role": role,
        "database_strategy": STRATEGY,
        "embedding_model": model,
        "embedding_dimensions": EXPECTED_DIMENSIONS,
        "indexed_chunks": chunks,
        "split": traces[0]["split"],
        "n": count,
        "top_k": top_k,
        "budget_chars": budget,
        "hit1": sum(trace["hit1"] for trace in traces),
        "hit3": sum(trace["hit3"] for trace in traces),
        "context_hit": sum(trace["context_hit"] for trace in traces),
        "avg_context_chars": statistics.mean(trace["context_chars"] for trace in traces),
        "embedding_batch_ms": embedding_ms,
        "median_database_ms": statistics.median(database_times),
        "total_elapsed_ms": total_ms,
        "avg_elapsed_ms_per_question": total_ms / count,
        "embedding_tokens": embedding_tokens,
        "service_cost_usd": None,
    }


def main():
    load_dotenv(HERE / ".env")
    parser = argparse.ArgumentParser(description="Run dense retrieval against the Day 6 pgvector index")
    parser.add_argument("--chunks", type=Path, default=HERE / "chunks_B.json")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "dev-dense-B")
    args = parser.parse_args()
    if args.top_k < 1 or args.budget < 1:
        raise ValueError("top-k và budget phải lớn hơn 0")

    chunks, questions = load_inputs(args.chunks, args.questions, args.split)
    database_url = get_connection_url()
    api_key = get_api_key()
    model = get_embedding_model()

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        db_info, indexed_chunks = verify_index(conn, chunks)
        traces, embedding_ms, embedding_tokens, database_times = retrieve_and_evaluate(
            conn, client, model, questions, top_k=args.top_k, budget=args.budget
        )

    summary = summarize(
        traces,
        database=db_info["database"],
        role=db_info["role"],
        model=model,
        chunks=indexed_chunks,
        embedding_ms=embedding_ms,
        embedding_tokens=embedding_tokens,
        database_times=database_times,
        top_k=args.top_k,
        budget=args.budget,
    )
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
