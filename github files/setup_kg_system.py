#!/usr/bin/env python3
"""
Setup Script for Knowledge Graph System
Runs the complete pipeline: NER/RE extraction -> KG creation -> Pinecone indexing
"""

import logging
from pathlib import Path
from ner_re_extractor import NERREExtractor
from neo4j_kg_store import Neo4jKGStore
from pinecone_enhanced import EnhancedPineconeManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_kg_system():
    """Setup complete knowledge graph system."""
    print("=" * 60)
    print("Knowledge Graph System Setup")
    print("=" * 60)
    
    # Step 1: NER/RE Extraction
    print("\n[Step 1/4] Running NER/RE Extraction...")
    print("-" * 60)
    try:
        extractor = NERREExtractor()
        entities, relations, document_extractions = extractor.process_all_documents()
        print(f"✓ Extracted {len(entities)} entities and {len(relations)} relations")
    except Exception as e:
        logger.error(f"NER/RE extraction failed: {e}")
        print(f"✗ NER/RE extraction failed: {e}")
        return False
    
    # Step 2: Create Neo4j Knowledge Graph
    print("\n[Step 2/4] Creating Neo4j Knowledge Graph...")
    print("-" * 60)
    try:
        kg_store = Neo4jKGStore(
            uri="neo4j://localhost:7687",
            user="neo4j",
            password=None
        )
        
        kg_store.create_constraints()
        entity_map = kg_store.load_entities_from_extractions("all_entities.json")
        kg_store.load_relations_from_extractions("all_relations.json", entity_map)
        
        stats = kg_store.get_statistics()
        print(f"✓ Created KG with {stats['total_entities']} entities and {stats['total_relations']} relations")
        kg_store.close()
    except Exception as e:
        logger.error(f"KG creation failed: {e}")
        print(f"✗ KG creation failed: {e}")
        print("  Make sure Neo4j is running and credentials are correct")
        return False
    
    # Step 3: Setup Pinecone with namespaces
    print("\n[Step 3/4] Setting up Pinecone with namespaces...")
    print("-" * 60)
    try:
        pinecone_manager = EnhancedPineconeManager()
        pinecone_manager.create_index(recreate=False)
        
        print("  Uploading papers...")
        pinecone_manager.upload_papers()
        
        print("  Uploading triples...")
        pinecone_manager.upload_triples()
        
        print("  Uploading claims...")
        pinecone_manager.upload_claims()
        
        print("✓ Pinecone setup complete")
    except Exception as e:
        logger.error(f"Pinecone setup failed: {e}")
        print(f"✗ Pinecone setup failed: {e}")
        print("  Make sure PINECONE_API_KEY is set in .env file")
        return False
    
    # Step 4: Verify setup
    print("\n[Step 4/4] Verifying setup...")
    print("-" * 60)
    
    kg_dir = Path("kg_data")
    if kg_dir.exists():
        entity_file = kg_dir / "all_entities.json"
        relation_file = kg_dir / "all_relations.json"
        if entity_file.exists() and relation_file.exists():
            print("✓ KG data files exist")
        else:
            print("✗ KG data files missing")
            return False
    else:
        print("✗ KG data directory missing")
        return False
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nYou can now:")
    print("  1. Run the chatbot: python chatbot.py")
    print("  2. Test MCP server: python mcp_server.py")
    print("  3. Use RAG orchestrator: python rag_orchestrator.py")
    print("\nMake sure to:")
    print("  - Set GEMINI_API_KEY in environment for chatbot")
    print("  - Set PINECONE_API_KEY in .env file")
    print("  - Configure Neo4j connection in scripts")
    
    return True

if __name__ == "__main__":
    success = setup_kg_system()
    if not success:
        exit(1)

