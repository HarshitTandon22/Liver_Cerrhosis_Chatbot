#!/usr/bin/env python3
"""
Convert JSON chunk files in `chunked_data` to CSV.

For each *.json file in the `chunked_data` folder, this script:
- Loads the JSON (expecting a list of objects / dicts)
- Flattens top-level keys into columns
- Writes a CSV with the same base name in the same folder.

It does NOT modify any existing code or data.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any


def json_list_to_csv_rows(data: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize a JSON list into a list of flat dicts suitable for CSV.
    - If an element is a dict, use it as-is.
    - Otherwise, wrap it into a dict with a single 'value' column.
    """
    rows: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
        else:
            rows.append({"value": item})
    return rows


def write_csv(path_json: Path) -> None:
    """Convert a single JSON file to CSV next to it."""
    with path_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        # If the JSON is not a list, wrap it so we can still write a CSV
        data = [data]

    rows = json_list_to_csv_rows(data)
    if not rows:
        # Nothing to write
        return

    # Collect all keys across rows to form the header
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    header = sorted(fieldnames)

    csv_path = path_json.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Written CSV: {csv_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    chunked_dir = base_dir / "chunked_data"

    if not chunked_dir.exists():
        print(f"'chunked_data' folder not found at: {chunked_dir}")
        return

    json_files = sorted(chunked_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {chunked_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s) in {chunked_dir}")
    for jf in json_files:
        print(f"Converting {jf.name} -> CSV")
        try:
            write_csv(jf)
        except Exception as e:
            print(f"  Error converting {jf.name}: {e}")


if __name__ == "__main__":
    main()

