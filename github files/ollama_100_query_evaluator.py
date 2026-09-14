#!/usr/bin/env python3
"""
Evaluate 100 queries against the dataset using Ollama (local LLaMA 3) as a judge.

Inputs:
- comparison and experiments/100 queries excel.xlsx  (first sheet)
- chunked_data/all_chunks_merged.csv                (dataset; must have a 'content' column)

For each query:
- Builds a prompt that includes:
  - The query text
  - A sampled subset of dataset chunks as context
- Asks Ollama (llama3:8b) to return a JSON object with numeric scores:
  f1, mse, accuracy, rmse, rmsle, r2, mape and an overall_relevance score.

Output:
- comparison and experiments/ollama_query_scores.json
  A single JSON file containing a list of per-query results.

This script only talks to Ollama (no Gemini/Groq) and does not modify existing code.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests


QUERIES_XLSX_PATH = Path("comparison and experiments/100 queries excel.xlsx")
DATASET_CSV_PATH = Path("chunked_data/all_chunks_merged.csv")
OUTPUT_JSON_PATH = Path("comparison and experiments/ollama_query_scores.json")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3:8b"


def load_queries() -> List[str]:
    df = pd.read_excel(QUERIES_XLSX_PATH, sheet_name=0)
    candidates = [c for c in df.columns if str(c).strip().lower() in {"query", "question"}]
    col = candidates[0] if candidates else df.columns[0]
    queries = [str(v).strip() for v in df[col].tolist() if str(v).strip()]
    return queries


def load_dataset_sample(max_samples: int = 30) -> List[str]:
    df = pd.read_csv(DATASET_CSV_PATH)
    if "content" in df.columns:
        series = df["content"].dropna()
    else:
        # Fallback: stringify the whole row
        series = df.apply(lambda r: json.dumps(r.to_dict(), ensure_ascii=False), axis=1)
    if len(series) > max_samples:
        sample = series.sample(n=max_samples, random_state=42)
    else:
        sample = series
    return sample.tolist()


def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an automatic evaluation agent.\n"
                    "Given a medical query and a sample of dataset chunks, "
                    "you must estimate how well the dataset can answer the query.\n\n"
                    "VERY IMPORTANT:\n"
                    "- You MUST output ONLY a single JSON object and nothing else.\n"
                    "- Do NOT add explanations, markdown, or backticks around it.\n"
                    "- The JSON object MUST have this exact shape (keys may be null):\n"
                    "{\n"
                    '  \"query\": \"...\",\n'
                    '  \"overall_relevance\": <float in [0,1]>,\n'
                    '  \"f1\": <float or null>,\n'
                    '  \"mse\": <float or null>,\n'
                    '  \"accuracy\": <float or null>,\n'
                    '  \"rmse\": <float or null>,\n'
                    '  \"rmsle\": <float or null>,\n'
                    '  \"r2\": <float or null>,\n'
                    '  \"mape\": <float or null>\n'
                    "}\n"
                    "If you are unsure about a metric, set it to null, "
                    "but STILL return valid JSON.\n"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
            "gpu": True,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "")
    return content


def extract_json(text: str) -> Dict[str, Any]:
    """
    Try to extract the first JSON object from the text.
    Handles cases where the model wraps JSON in markdown fences or adds stray text.
    """
    text = text.strip()

    # Case 1: pure JSON
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Case 2: fenced ```json ... ``` block
    fence_start = text.find("```json")
    if fence_start != -1:
        fence_start = text.find("{", fence_start)
        fence_end = text.find("```", fence_start)
        if fence_start != -1 and fence_end != -1:
            snippet = text[fence_start:fence_end].strip()
            return json.loads(snippet)

    # Case 3: any {...} span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        return json.loads(snippet)

    # If we reach here, we log a bit and raise
    raise ValueError(f"Could not find JSON object in Ollama response: {text[:200]}...")


def main() -> None:
    if not QUERIES_XLSX_PATH.exists():
        print(f"Queries Excel not found: {QUERIES_XLSX_PATH}")
        return
    if not DATASET_CSV_PATH.exists():
        print(f"Dataset CSV not found: {DATASET_CSV_PATH}")
        return

    queries = load_queries()
    dataset_samples = load_dataset_sample(max_samples=30)

    print(f"Loaded {len(queries)} queries from {QUERIES_XLSX_PATH}")
    print(f"Prepared {len(dataset_samples)} dataset samples from {DATASET_CSV_PATH}")

    results: List[Dict[str, Any]] = []

    for idx, q in enumerate(queries, 1):
        print(f"\n=== Evaluating query {idx}/{len(queries)} ===")
        context_preview = "\n---\n".join(dataset_samples)
        prompt = (
            f"QUERY:\n{q}\n\n"
            "DATASET SAMPLES (each is a chunk of text from the corpus):\n"
            f"{context_preview}\n"
        )
        try:
            raw = call_ollama(prompt)
            obj = extract_json(raw)
        except Exception as e:
            print(f"  Error evaluating query {idx}: {e}")
            obj = {
                "query": q,
                "overall_relevance": None,
                "f1": None,
                "mse": None,
                "accuracy": None,
                "rmse": None,
                "rmsle": None,
                "r2": None,
                "mape": None,
                "error": str(e),
            }

        # Ensure query field is set correctly
        obj["query"] = q
        results.append(obj)

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved per-query scores to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()

