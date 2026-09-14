exitimport os
import sys
from typing import List

import httpx


SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8000").rstrip("/")


def ask(question: str) -> str:
    with httpx.Client(timeout=90) as client:
        resp = client.post(f"{SERVER_URL}/rag.answer_with_evidence", json={"question": question, "filters": {}})
        if resp.status_code >= 400:
            return f"HTTP {resp.status_code}: {resp.text[:500]}"
        data = resp.json()
        return data.get("answer", "")


def main() -> None:
    questions: List[str] = [
        "What are the common complications of decompensated liver cirrhosis?",
        "How is spontaneous bacterial peritonitis diagnosed and managed?",
        "Does non-selective beta-blocker therapy reduce variceal bleeding risk?",
    ]
    for i, q in enumerate(questions, 1):
        print("=" * 80)
        print(f"Q{i}: {q}")
        try:
            a = ask(q)
        except Exception as e:
            a = f"[Error querying server: {e}]"
        print("-" * 80)
        print(a)
        print()


if __name__ == "__main__":
    main()


