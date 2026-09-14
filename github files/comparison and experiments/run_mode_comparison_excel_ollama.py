#!/usr/bin/env python3
"""
Run 100-query comparison for three modes using Ollama-backed chatbot:
1) LLM-only
2) Vector RAG
3) Full KG-MCP-RAG

Inputs:
- 100 queries excel.xlsx

Outputs:
- model_comparison_100_queries_ollama.xlsx

This script is a copy of the original comparison script, but it uses
`chatbot_ollama.LiverCirrhosisChatBotOllama` instead of the Gemini-based chatbot.
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import pandas as pd

# Add project root to path for imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from chatbot_ollama import LiverCirrhosisChatBotOllama
from evaluate_chatbot_metrics import check_relevance_score


def load_queries_from_excel(
    path: str,
    max_queries: int = 100,
) -> Tuple[List[str], List[str]]:
    """
    Load queries (and optional reference answers) from an existing Excel file.

    Assumes:
    - First sheet contains the data
    - There is either a 'query'/'question' column OR the first column is queries
    - If a second column exists, it is treated as a reference/expected answer
    """
    df = pd.read_excel(path, sheet_name=0)

    query_col_candidates = [c for c in df.columns if str(c).strip().lower() in {"query", "question"}]
    if query_col_candidates:
        query_col = query_col_candidates[0]
    else:
        query_col = df.columns[0]

    if len(df.columns) > 1:
        ref_col = df.columns[1]
    else:
        ref_col = None

    queries: List[str] = []
    ref_answers: List[str] = []

    for _, row in df.iterrows():
        q = str(row.get(query_col, "")).strip()
        if not q:
            continue
        queries.append(q)
        if ref_col is not None:
            ref_answers.append(str(row.get(ref_col, "")).strip())
        else:
            ref_answers.append("")

        if len(queries) >= max_queries:
            break

    return queries, ref_answers


def compute_keywords(query: str, reference_answer: str) -> List[str]:
    """
    Derive a simple keyword list from query and optional reference answer.
    This is used as input to check_relevance_score (G-EVAL-style relevance).
    """
    base_text = f"{query} {reference_answer}".lower()
    tokens = [t for t in base_text.replace("?", " ").replace(",", " ").split() if len(t) > 3]
    seen = set()
    keywords: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            keywords.append(t)
    return keywords


def run_mode_for_query(
    chatbot: LiverCirrhosisChatBotOllama,
    query: str,
    reference_answer: str,
    mode: str,
) -> Dict[str, Any]:
    """
    Run a single query in a specific mode and return detailed info.

    mode:
      - "llm_only": no RAG, no KG
      - "vector_rag": RAG only
      - "kg_mcp_rag": RAG + KG
    """
    if mode == "llm_only":
        use_rag = False
        use_kg = False
    elif mode == "vector_rag":
        use_rag = True
        use_kg = False
    elif mode == "kg_mcp_rag":
        use_rag = True
        use_kg = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    keywords = compute_keywords(query, reference_answer)

    start_time = time.time()
    response = chatbot.chat(query, use_rag=use_rag, use_kg=use_kg, max_evidence=5)
    latency = time.time() - start_time

    answer = response.get("answer", "")
    sources_list = response.get("sources", []) or []
    sources_str = ", ".join(sources_list)

    relevance = check_relevance_score(answer, query, keywords)

    return {
        "answer": answer,
        "sources": sources_str,
        "latency_sec": latency,
        "answer_length": len(answer),
        "relevance": relevance,
    }


def main() -> None:
    os.chdir(PROJECT_ROOT)

    excel_in_path = os.path.join(THIS_DIR, "100 queries excel.xlsx")
    if not os.path.exists(excel_in_path):
        print(f"Input Excel not found at: {excel_in_path}")
        print("Please make sure '100 queries excel.xlsx' exists in this folder.")
        sys.exit(1)

    print(f"Loading queries from: {excel_in_path}")
    queries, ref_answers = load_queries_from_excel(excel_in_path, max_queries=100)

    if not queries:
        print("No queries loaded from Excel. Aborting.")
        sys.exit(1)

    print(f"Loaded {len(queries)} queries (limit = 100).")

    print("Initializing Ollama-backed chatbot...")
    chatbot = LiverCirrhosisChatBotOllama(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=os.getenv("NEO4J_PASSWORD", "your_neo4j_password_here"),
    )
    print("Chatbot initialized.")

    modes = ["llm_only", "vector_rag", "kg_mcp_rag"]

    rows: List[Dict[str, Any]] = []

    max_queries_attempt = min(100, len(queries))
    max_queries_run = max_queries_attempt

    for idx in range(max_queries_attempt):
        query = queries[idx]
        ref_answer = ref_answers[idx]

        print(f"\n=== Query {idx + 1}/{max_queries_attempt}: {query[:80]} ===")

        row: Dict[str, Any] = {
            "query_index": idx + 1,
            "query": query,
            "reference_answer": ref_answer,
        }

        for mode in modes:
            mode_prefix = {
                "llm_only": "llm_only",
                "vector_rag": "vector_rag",
                "kg_mcp_rag": "kg_mcp_rag",
            }[mode]

            try:
                result = run_mode_for_query(chatbot, query, ref_answer, mode)
                row[f"{mode_prefix}_answer"] = result["answer"]
                row[f"{mode_prefix}_sources"] = result["sources"]
                row[f"{mode_prefix}_latency_sec"] = result["latency_sec"]
                row[f"{mode_prefix}_answer_length"] = result["answer_length"]
                row[f"{mode_prefix}_relevance"] = result["relevance"]
            except Exception as e:
                row[f"{mode_prefix}_answer"] = f"[ERROR] {e}"
                row[f"{mode_prefix}_sources"] = ""
                row[f"{mode_prefix}_latency_sec"] = 0.0
                row[f"{mode_prefix}_answer_length"] = 0
                row[f"{mode_prefix}_relevance"] = 0.0
                print(f"  Error in mode {mode}: {e}")

        rows.append(row)

    rows = rows[:max_queries_run]

    chatbot.close()

    if not rows:
        print("No successful rows collected. Nothing to write.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    summary_rows: List[Dict[str, Any]] = []
    for mode_prefix, label in [
        ("llm_only", "LLM-only"),
        ("vector_rag", "Vector RAG"),
        ("kg_mcp_rag", "Full KG-MCP-RAG"),
    ]:
        mode_df = df[df[f"{mode_prefix}_answer"].notna()]
        if mode_df.empty:
            continue
        summary_rows.append(
            {
                "mode": label,
                "num_queries": len(mode_df),
                "avg_relevance": mode_df[f"{mode_prefix}_relevance"].mean(),
                "avg_answer_length": mode_df[f"{mode_prefix}_answer_length"].mean(),
                "avg_latency_sec": mode_df[f"{mode_prefix}_latency_sec"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    out_name = "model_comparison_100_queries_ollama.xlsx"
    excel_out_path = os.path.join(THIS_DIR, out_name)

    print(f"\nWriting comparison Excel to: {excel_out_path}")
    with pd.ExcelWriter(excel_out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="per_query_results", index=False)
        summary_df.to_excel(writer, sheet_name="summary_metrics", index=False)

    print("Done.")


if __name__ == "__main__":
    main()

