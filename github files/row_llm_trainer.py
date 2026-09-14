#!/usr/bin/env python3
"""
Row-wise LLM evaluation pipeline with resume support.

- Reads:
  - Main CSV (e.g. all_chunks_merged.csv) with many rows
  - 100-queries Excel (first sheet) with a 'query' column or first column as queries

- For each row in the CSV:
  - Builds a prompt using row data + all 100 queries
  - Calls LLM with fallback: Gemini -> Groq -> Ollama LLaMA 3 (8B)
  - Saves a JSON file per row in output folder.
  - Optionally computes metrics if label/pred columns are configured.

- Resuming:
  - If run stops, re-running will skip rows that already have a JSON file.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

from llm_fallback_client import LLMFallbackClient


# === CONFIGURATION ===

CSV_PATH = Path("chunked_data/all_chunks_merged.csv")
QUERIES_XLSX_PATH = Path("comparison and experiments/100 queries excel.xlsx")
OUTPUT_DIR = Path("row_training_outputs")

# Optional: set these to numeric columns if you have them in your CSV
Y_TRUE_COL: Optional[str] = None
Y_PRED_COL: Optional[str] = None


def compute_metrics(
    y_true: Optional[float],
    y_pred: Optional[float],
) -> Dict[str, Optional[float]]:
    """Compute basic metrics for a single pair; returns None where not applicable."""
    if y_true is None or y_pred is None:
        return {
            "f1": None,
            "mse": None,
            "accuracy": None,
            "rmse": None,
            "rmsle": None,
            "r2": None,
            "mape": None,
        }

    y_true_arr = np.array([y_true])
    y_pred_arr = np.array([y_pred])

    metrics: Dict[str, Optional[float]] = {}

    # Classification-like
    try:
        metrics["f1"] = float(f1_score(y_true_arr, y_pred_arr, average="binary"))
        metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
    except Exception:
        metrics["f1"] = None
        metrics["accuracy"] = None

    # Regression-like
    try:
        mse = float(mean_squared_error(y_true_arr, y_pred_arr))
        metrics["mse"] = mse
        metrics["rmse"] = float(math.sqrt(mse))
    except Exception:
        metrics["mse"] = None
        metrics["rmse"] = None

    # RMSLE
    try:
        if y_true > 0 and y_pred > 0:
            rmsle = float(math.sqrt(mean_squared_log_error(y_true_arr, y_pred_arr)))
        else:
            rmsle = None
        metrics["rmsle"] = rmsle
    except Exception:
        metrics["rmsle"] = None

    # R² and MAPE
    try:
        metrics["r2"] = float(r2_score(y_true_arr, y_pred_arr))
    except Exception:
        metrics["r2"] = None

    try:
        metrics["mape"] = float(
            mean_absolute_percentage_error(y_true_arr, y_pred_arr)
        )
    except Exception:
        metrics["mape"] = None

    return metrics


def load_queries() -> List[str]:
    df = pd.read_excel(QUERIES_XLSX_PATH, sheet_name=0)
    candidates = [c for c in df.columns if str(c).strip().lower() in {"query", "question"}]
    col = candidates[0] if candidates else df.columns[0]
    queries = [str(v).strip() for v in df[col].tolist() if str(v).strip()]
    return queries


def build_prompt_for_row(row: pd.Series, queries: List[str]) -> str:
    row_json = row.to_dict()
    parts = [
        "You are a medical research assistant working with chunked evidence from liver cirrhosis papers.",
        "Below is a single data row from our CSV (chunk-level information).",
        "Then you will see a list of up to 100 queries we care about.",
        "",
        "=== ROW DATA (JSON) ===",
        json.dumps(row_json, ensure_ascii=False, indent=2),
        "",
        "=== QUERIES ===",
    ]
    for i, q in enumerate(queries, 1):
        parts.append(f"{i}. {q}")
    parts.append("")
    parts.append(
        "For this row, generate any useful structured representation or summary "
        "that helps answer these queries (e.g., key entities, relations, findings). "
        "Keep the answer concise but information-dense."
    )
    return "\n".join(parts)


def get_resume_start_index() -> int:
    if not OUTPUT_DIR.exists():
        return 0
    existing = sorted(OUTPUT_DIR.glob("row_*.json"))
    if not existing:
        return 0
    max_idx = 0
    for p in existing:
        name = p.stem
        try:
            idx = int(name.split("_", 1)[1])
            max_idx = max(max_idx, idx)
        except Exception:
            continue
    return max_idx + 1


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return
    if not QUERIES_XLSX_PATH.exists():
        print(f"Queries Excel not found: {QUERIES_XLSX_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    queries = load_queries()

    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Loaded {len(queries)} queries from {QUERIES_XLSX_PATH}")

    start_idx = get_resume_start_index()
    print(f"Resuming from row index: {start_idx}")

    llm = LLMFallbackClient()

    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        out_path = OUTPUT_DIR / f"row_{idx:05d}.json"
        if out_path.exists():
            continue

        print(f"\n=== Processing row {idx}/{len(df)-1} ===")
        prompt = build_prompt_for_row(row, queries)

        try:
            response_text, backend = llm.generate(prompt)
        except Exception as e:
            print(f"  LLM call failed for row {idx}: {e}")
            print("  Stopping; existing JSON files are preserved. Re-run to resume.")
            break

        y_true = None
        y_pred = None
        if Y_TRUE_COL and Y_TRUE_COL in df.columns:
            try:
                y_true = float(row[Y_TRUE_COL])
            except Exception:
                y_true = None
        if Y_PRED_COL and Y_PRED_COL in df.columns:
            try:
                y_pred = float(row[Y_PRED_COL])
            except Exception:
                y_pred = None

        metrics = compute_metrics(y_true, y_pred)

        out_obj: Dict[str, Any] = {
            "row_index": idx,
            "backend": backend,
            "prompt": prompt,
            "response": response_text,
            "metrics": metrics,
            "row_primary_keys": {
                k: row[k]
                for k in ["chunk_id", "document_id", "chunk_index"]
                if k in df.columns
            },
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print(f"  Saved {out_path}")

    print("\nDone. You can re-run this script; it will continue from the next unfinished row.")


if __name__ == "__main__":
    main()

