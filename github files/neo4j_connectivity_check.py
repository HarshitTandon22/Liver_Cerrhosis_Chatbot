#!/usr/bin/env python3
"""Quick diagnostic for Neo4j connectivity."""

from neo4j import GraphDatabase


def check(uri: str, user: str = "neo4j", password: str = None) -> None:
    print(f"\n== Checking {uri} ==")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except Exception as exc:
        print(f"Connectivity FAILED: {exc}")
    else:
        print("Connectivity OK")
    finally:
        try:
            driver.close()  # type: ignore[name-defined]
        except Exception:
            pass


def main() -> None:
    for candidate in ("neo4j://localhost:7687", "bolt://localhost:7687"):
        check(candidate)


if __name__ == "__main__":
    main()










