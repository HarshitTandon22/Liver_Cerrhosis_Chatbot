#!/usr/bin/env python3
"""Quick verification harness for the MCP server stack."""

from pprint import pprint

from mcp_server import MCPServer


def pick_two_entities(results):
    names = []
    for item in results:
        name = item.get("name") or item.get("normalized_name")
        if name and name not in names:
            names.append(name)
        if len(names) == 2:
            break
    return names if len(names) == 2 else None


def main() -> None:
    server = MCPServer()
    try:
        print("Checking kg.search_graph for 'cirrhosis'...")
        kg = server.kg_search_graph("cirrhosis", limit=10)
        print(f"entities_found={kg.get('total')}, error={kg.get('error')}")
        pprint(kg.get('results', [])[:3])

        pair = pick_two_entities(kg.get("results", []))
        if pair:
            print(f"\nAttempting path between: {pair[0]} and {pair[1]}")
            try:
                path_result = server.kg_path_explain(pair[0], pair[1], max_depth=3)
            except Exception as exc:
                print(f"path_explain error: {exc}")
            else:
                print(f"paths_found={path_result.get('total_paths')}, error={path_result.get('error')}")
                pprint(path_result.get('paths', [])[:2])
        else:
            print("Not enough distinct entities returned to attempt path search.")

        print("\nCalling rag.answer_with_evidence...")
        rag = server.rag_answer_with_evidence("What are complications of cirrhosis?", top_k=3)
        print(f"evidence_items={len(rag.get('evidence', []))}")
        print(rag.get('answer', '')[:400])
    finally:
        server.close()


if __name__ == "__main__":
    main()










