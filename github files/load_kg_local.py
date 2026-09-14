#!/usr/bin/env python3
"""Load existing KG JSON into the local Neo4j instance."""

from neo4j_kg_store import Neo4jKGStore


def main() -> None:
    store = Neo4jKGStore(uri="bolt://localhost:7687", user="neo4j", password=None)
    try:
        store.create_constraints()
        entity_map = store.load_entities_from_extractions("all_entities.json")
        store.load_relations_from_extractions("all_relations.json", entity_map)
        stats = store.get_statistics()
        print("Loaded KG:", stats)
    finally:
        store.close()


if __name__ == "__main__":
    main()










