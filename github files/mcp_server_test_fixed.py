#!/usr/bin/env python3
"""Safe MCP server smoke test that avoids KeyError when Neo4j is offline."""

import pprint

from mcp_server import MCPServer


def safe_get(mapping, key, default=None):
    try:
        return mapping[key]
    except KeyError:
        return default


def main() -> None:
    print("MCP Server Safe Test")
    print("=" * 50)

    server = MCPServer()
    try:
        print("\n1. kg.search_graph — top 5 for 'cirrhosis'")
        kg_search = server.kg_search_graph("cirrhosis", limit=5)
        print(f"entities_found = {kg_search.get('total')}, error = {kg_search.get('error')}")

        print("\n2. kg.find_relations — 'cirrhosis'")
        relations = server.kg_find_relations("cirrhosis", limit=5)
        print(f"relations_found = {relations.get('total')}, error = {relations.get('error')}")

        print("\n3. kg.path_explain — 'cirrhosis' -> 'portal vein thrombosis'")
        try:
            paths = server.kg_path_explain("cirrhosis", "portal vein thrombosis", max_depth=2)
        except Exception as exc:
            print(f"paths query error: {exc}")
        else:
            total_paths = safe_get(paths, "total_paths", 0)
            error = paths.get("error")
            print(f"paths_found = {total_paths}, error = {error}")
            if total_paths and not error:
                pprint.pprint(paths.get("paths", [])[:2])

        print("\n4. rag.answer_with_evidence — question")
        rag = server.rag_answer_with_evidence("What is portal vein thrombosis?", top_k=3)
        evidence_count = len(rag.get("evidence", []))
        print(f"evidence_items = {evidence_count}")
    finally:
        server.close()


if __name__ == "__main__":
    main()


