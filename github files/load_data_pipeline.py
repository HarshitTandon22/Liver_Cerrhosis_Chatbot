#!/usr/bin/env python3
"""
Complete Data Loading Pipeline
Loads data into Neo4j and Pinecone, then tests chatbot.
"""

import os
from dotenv import load_dotenv

load_dotenv()
import logging
from pathlib import Path
from neo4j_kg_store import Neo4jKGStore
from pinecone_enhanced import EnhancedPineconeManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set credentials

# Neo4j connection - try different formats
NEO4J_URIS = [
    "neo4j://localhost:7687",
    "neo4j://localhost:7687",
    "bolt://localhost:7687",
]
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_neo4j_password_here")

def try_neo4j_connection():
    """Try connecting to Neo4j with different URI formats."""
    for uri in NEO4J_URIS:
        try:
            logger.info(f"Trying Neo4j URI: {uri}")
            kg_store = Neo4jKGStore(uri=uri, user=NEO4J_USER, password=NEO4J_PASSWORD)
            # Test connection
            stats = kg_store.get_statistics()
            logger.info(f"✓ Connected successfully with URI: {uri}")
            kg_store.close()
            return uri
        except Exception as e:
            logger.warning(f"Failed with {uri}: {e}")
            continue
    return None

def main():
    print("=" * 60)
    print("Complete Data Loading Pipeline")
    print("=" * 60)
    
    # Step 1: Try Neo4j connection
    print("\n[Step 1] Testing Neo4j connection...")
    neo4j_uri = try_neo4j_connection()
    if not neo4j_uri:
        print("Warning: Could not connect to Neo4j. Skipping KG loading.")
        print("  You may need to verify the database ID and credentials.")
        neo4j_loaded = False
    else:
        # Load into Neo4j
        print("\n[Step 2] Loading data into Neo4j...")
        try:
            kg_store = Neo4jKGStore(uri=neo4j_uri, user=NEO4J_USER, password=NEO4J_PASSWORD)
            kg_store.create_constraints()
            entity_map = kg_store.load_entities_from_extractions("all_entities.json")
            kg_store.load_relations_from_extractions("all_relations.json", entity_map)
            stats = kg_store.get_statistics()
            print(f"✓ Loaded {stats['total_entities']} entities and {stats['total_relations']} relations into Neo4j")
            kg_store.close()
            neo4j_loaded = True
        except Exception as e:
            logger.error(f"Failed to load into Neo4j: {e}")
            neo4j_loaded = False
    
    # Step 3: Upload to Pinecone
    print("\n[Step 3] Uploading to Pinecone...")
    try:
        pinecone_manager = EnhancedPineconeManager()
        pinecone_manager.create_index(recreate=False)
        
        print("  Uploading papers...")
        pinecone_manager.upload_papers()
        
        print("  Uploading triples...")
        pinecone_manager.upload_triples()
        
        print("  Uploading claims...")
        pinecone_manager.upload_claims()
        
        print("✓ Pinecone upload complete")
    except Exception as e:
        logger.error(f"Failed to upload to Pinecone: {e}")
        print(f"✗ Pinecone upload failed: {e}")
    
    # Step 4: Test chatbot
    print("\n[Step 4] Testing chatbot...")
    print("=" * 60)
    print("You can now run the chatbot with:")
    print("  python chatbot.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

