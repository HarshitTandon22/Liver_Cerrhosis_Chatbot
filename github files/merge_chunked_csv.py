#!/usr/bin/env python3
"""
Merge all CSV files in `chunked_data` into a single CSV.

Assumes you already ran `chunked_to_csv.py` so that each JSON file in
`chunked_data` has a corresponding CSV.

Output:
- `chunked_data/all_chunks_merged.csv`

No existing code is modified.
"""

import csv
from pathlib import Path
from typing import Dict, Any, List


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Read all rows from a CSV file into a list of dicts."""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    chunked_dir = base_dir / "chunked_data"

    if not chunked_dir.exists():
        print(f"'chunked_data' folder not found at: {chunked_dir}")
        return

    csv_files = sorted(
        f for f in chunked_dir.glob("*.csv")
        if f.name != "all_chunks_merged.csv"
    )
    if not csv_files:
        print(f"No CSV files found in {chunked_dir}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {chunked_dir}")

    all_rows: List[Dict[str, Any]] = []
    all_fields = set()

    for path in csv_files:
        print(f"Reading {path.name}...")
        rows = read_csv_rows(path)
        all_rows.extend(rows)
        for r in rows:
            all_fields.update(r.keys())

    if not all_rows:
        print("No rows found across CSV files; nothing to merge.")
        return

    fieldnames = sorted(all_fields)
    out_path = chunked_dir / "all_chunks_merged.csv"

    print(f"Writing merged CSV: {out_path}")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print("Merge complete.")


if __name__ == "__main__":
    main()

