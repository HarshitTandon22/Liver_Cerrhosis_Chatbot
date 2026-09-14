#!/usr/bin/env python3
"""
Complete Data Loading and ChatBot Runner
Loads data into Neo4j and Pinecone, then runs the chatbot.
"""

import os
import logging
from pathlib import Path
from ner_re_extractor import NERREExtractor
from neo4j_kg_store import Neo4jKGStore
from pinecone_enhanced import EnhancedPineconeManager
from chatbot import LiverCirrhosisChatBot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("=" * 70)
    print("Complete Data Loading and ChatBot Setup")
    print("=" * 70)
    
    # Step 1: NER/RE Extraction
    print("\n[Step 1/5] Running NER/RE Extraction...")
    print("-" * 70)
    try:
        extractor = NERREExtractor()
        entities, relations, document_extractions = extractor.process_all_documents()
        print(f"SUCCESS: Extracted {len(entities)} entities and {len(relations)} relations")
    except Exception as e:
        logger.error(f"NER/RE extraction failed: {e}")
        print(f"ERROR: NER/RE extraction failed: {e}")
        return False
    
    # Step 2: Load into Neo4j
    print("\n[Step 2/5] Loading data into Neo4j...")
    print("-" * 70)
    try:
        kg_store = Neo4jKGStore()
        kg_store.create_constraints()
        entity_map = kg_store.load_entities_from_extractions("all_entities.json")
        kg_store.load_relations_from_extractions("all_relations.json", entity_map)
        
        stats = kg_store.get_statistics()
        print(f"✓ Loaded {stats['total_entities']} entities and {stats['total_relations']} relations into Neo4j")
        kg_store.close()
    except Exception as e:
        logger.error(f"Neo4j loading failed: {e}")
        print(f"✗ Neo4j loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Setup Pinecone
    print("\n[Step 3/5] Setting up Pinecone with namespaces...")
    print("-" * 70)
    try:
        pinecone_manager = EnhancedPineconeManager()
        pinecone_manager.create_index(recreate=False)
        
        print("  Uploading papers to Pinecone...")
        pinecone_manager.upload_papers()
        
        print("  Uploading triples to Pinecone...")
        pinecone_manager.upload_triples()
        
        print("  Uploading claims to Pinecone...")
        pinecone_manager.upload_claims()
        
        print("SUCCESS: Pinecone setup complete")
    except Exception as e:
        logger.error(f"Pinecone setup failed: {e}")
        print(f"ERROR: Pinecone setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Verify setup
    print("\n[Step 4/5] Verifying setup...")
    print("-" * 70)
    kg_dir = Path("kg_data")
    if kg_dir.exists() and (kg_dir / "all_entities.json").exists() and (kg_dir / "all_relations.json").exists():
        print("SUCCESS: All data files verified")
    else:
        print("ERROR: Data files missing")
        return False
    
    # Step 5: Run ChatBot
    print("\n[Step 5/5] Starting ChatBot...")
    print("-" * 70)
    print("\n" + "=" * 70)
    print("ChatBot is ready! Ask questions about liver cirrhosis.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 70)
    
    try:
        chatbot = LiverCirrhosisChatBot()
        chatbot.interactive_chat()
    except Exception as e:
        logger.error(f"ChatBot error: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            chatbot.close()
        except:
            pass
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)

